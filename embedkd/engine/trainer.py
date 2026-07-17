"""Training loop.

Numerical-safety design: the backbone forward pass runs under autocast when
AMP is enabled, but every loss (task and distillation) is computed in fp32 on
upcast embeddings OUTSIDE the autocast region. Geometric losses such as RKD
underflow in fp16 (tiny pairwise distances -> inf gradients -> GradScaler
silently skips steps and training freezes), so fp32 loss computation is the
default for everything, not an opt-in.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.samplers import PKSampler
from ..evaluation import evaluate_model
from .seed import set_seed, worker_init_fn


def _build_param_groups(student, task_loss, train_cfg: dict) -> list[dict]:
    """Two-tier learning rate: pretrained backbone runs ~10x cooler than the
    fresh head/classifier/loss parameters. Single-LR fine-tuning burns the
    pretrained features and collapses open-set retrieval (inherited, proven
    remedy from the authors' prior experiments)."""
    lr_head = float(train_cfg["lr"])
    lr_backbone = train_cfg.get("lr_backbone")
    lr_backbone = lr_head / 10.0 if lr_backbone is None else float(lr_backbone)
    head_params = [p for name, p in student.named_parameters()
                   if not name.startswith("backbone.")]
    return [
        {"params": list(student.backbone.parameters()), "lr": lr_backbone},
        {"params": head_params + list(task_loss.parameters()), "lr": lr_head},
    ]


def _build_optimizer(param_groups, train_cfg: dict) -> torch.optim.Optimizer:
    name = train_cfg["optimizer"]
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=train_cfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9,
                               weight_decay=train_cfg["weight_decay"])
    raise ValueError(f"Unknown optimizer '{name}' (use adamw | sgd)")


def _lr_scale(epoch: int, train_cfg: dict) -> float:
    warmup = int(train_cfg.get("warmup_epochs", 0))
    total = int(train_cfg["epochs"])
    if warmup and epoch < warmup:
        return (epoch + 1) / warmup
    if train_cfg["scheduler"] == "cosine":
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    if train_cfg["scheduler"] == "step":
        return 0.1 ** (epoch // max(1, total // 3))
    return 1.0


class Trainer:
    def __init__(self, cfg, student, teacher, objective, task_loss, bundle, out_dir, device=None):
        self.cfg = cfg
        self.student = student
        self.teacher = teacher
        self.objective = objective
        self.task_loss = task_loss
        self.bundle = bundle
        self.out_dir = Path(out_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.history: list[dict] = []
        self.best = {"metric": -1.0, "epoch": -1}

    def _train_loader(self, generator: torch.Generator) -> DataLoader:
        data_cfg = self.cfg["data"]
        common = dict(num_workers=data_cfg["num_workers"], worker_init_fn=worker_init_fn)
        if data_cfg["sampler"] == "pk":
            self.pk_sampler = PKSampler(
                self.bundle.train_labels, data_cfg["p_classes"], data_cfg["k_samples"],
                seed=self.cfg["train"]["seed"],
            )
            return DataLoader(self.bundle.train, batch_sampler=self.pk_sampler, **common)
        self.pk_sampler = None
        return DataLoader(
            self.bundle.train, batch_size=self.cfg["train"]["batch_size"], shuffle=True,
            generator=generator, drop_last=True, **common,
        )

    def _relational_scale(self, epoch: int) -> float:
        """Maturation ramp for relational objectives (RKD-style), inherited:
        relational terms spike on an immature student and can collapse
        training, so they stay OFF until warmup ends, then ramp 0 -> 1."""
        ramp_cfg = self.cfg["distill"].get("relational_ramp") or {}
        start = ramp_cfg.get("start_epoch")
        if start is None:
            start = int(self.cfg["train"].get("warmup_epochs", 0))
        ramp_epochs = max(1, int(ramp_cfg.get("epochs", 5)))
        if epoch < start:
            return 0.0
        return min(1.0, (epoch - start + 1) / ramp_epochs)

    def _forward_losses(self, images: torch.Tensor, labels: torch.Tensor, amp: bool,
                        relational_scale: float = 1.0):
        alpha = float(self.cfg["distill"]["alpha"])
        autocast_ctx = torch.autocast(self.device.type, enabled=amp)
        with autocast_ctx:
            s_emb, s_logits = self.student(images, return_logits=True)
        # fp32 island: all losses outside autocast on upcast tensors.
        s_emb32 = s_emb.float()
        s_logits32 = s_logits.float() if s_logits is not None else None
        task = self.task_loss(s_emb32, s_logits32, labels)
        if alpha == 0.0:
            # Standalone training (e.g. preparing a teacher): skip the teacher
            # forward pass entirely instead of computing an unused objective.
            return task, task, s_emb.new_zeros(())
        with torch.no_grad(), autocast_ctx:
            t_emb, t_logits = self.teacher(images, return_logits=True)
        t_emb32 = t_emb.float()
        t_logits32 = t_logits.float() if t_logits is not None else None
        distill = self.objective(s_emb32, t_emb32, s_logits=s_logits32, t_logits=t_logits32,
                                 relational_scale=relational_scale)
        total = task + alpha * distill
        return total, task, distill

    def fit(self) -> dict:
        train_cfg = self.cfg["train"]
        generator = set_seed(train_cfg["seed"])
        self.student.to(self.device)
        if float(self.cfg["distill"]["alpha"]) != 0.0:
            self.teacher.to(self.device)
        self.teacher.eval()
        self.task_loss.to(self.device)
        self.objective.to(self.device)

        param_groups = _build_param_groups(self.student, self.task_loss, train_cfg)
        optimizer = _build_optimizer(param_groups, train_cfg)
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        amp = bool(train_cfg["amp"]) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=amp)
        loader = self._train_loader(generator)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.out_dir / "log.jsonl"
        patience_cfg = train_cfg.get("early_stopping") or None
        stale = 0

        with open(log_path, "a", encoding="utf-8") as log_file:
            for epoch in range(train_cfg["epochs"]):
                if self.pk_sampler is not None:
                    self.pk_sampler.set_epoch(epoch)
                scale = _lr_scale(epoch, train_cfg)
                for group, base in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base * scale
                self.student.train()
                sums = {"total": 0.0, "task": 0.0, "distill": 0.0}
                steps = 0
                rel_scale = self._relational_scale(epoch)
                grad_clip = float(train_cfg.get("grad_clip") or 0.0)
                trained_params = [p for group in optimizer.param_groups for p in group["params"]]
                for images, labels in loader:
                    images = images.to(self.device)
                    labels = torch.as_tensor(labels).to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    total, task, distill = self._forward_losses(
                        images, labels, amp, relational_scale=rel_scale)
                    scaler.scale(total).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trained_params, max_norm=grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    sums["total"] += float(total.detach())
                    sums["task"] += float(task.detach())
                    sums["distill"] += float(distill.detach())
                    steps += 1
                record = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{k: v / max(1, steps) for k, v in sums.items()},
                }
                if rel_scale < 1.0 and getattr(self.objective, "relational", False):
                    record["relational_scale"] = round(rel_scale, 4)
                nonfinite = getattr(self.objective, "nonfinite_count", 0)
                if nonfinite:
                    record["distill_nonfinite_skips"] = nonfinite
                eval_every = max(1, int(train_cfg.get("eval_every", 1)))
                is_eval_epoch = (epoch + 1) % eval_every == 0 or epoch == train_cfg["epochs"] - 1
                if is_eval_epoch:
                    metrics = self.evaluate()
                    record.update({f"val_{k}": v for k, v in metrics.items()})
                    monitored = metrics.get("map", 0.0)
                    if monitored > self.best["metric"]:
                        self.best = {"metric": monitored, "epoch": epoch}
                        self._save("best.pth", epoch)
                        stale = 0
                    else:
                        stale += 1
                self.history.append(record)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                self._save("last.pth", epoch)
                if patience_cfg and stale >= int(patience_cfg.get("patience", 10)):
                    break

        return {
            "history": self.history,
            "best": self.best,
            "checkpoints": {name: str(self.out_dir / f"{name}.pth") for name in ("best", "last")},
        }

    def evaluate(self, target: bool = False) -> dict[str, float]:
        gallery = self.bundle.target_gallery if target else self.bundle.gallery
        query = self.bundle.target_query if target else self.bundle.query
        if gallery is None or query is None:
            raise ValueError("No target split available; configure data.protocol: cross_domain")
        return evaluate_model(
            self.student, gallery, query,
            batch_size=self.cfg["eval"]["batch_size"], device=self.device,
        )

    def _save(self, name: str, epoch: int) -> None:
        torch.save(
            {
                "state_dict": self.student.state_dict(),
                "task_loss_state": self.task_loss.state_dict(),
                "config": self.cfg,
                "epoch": epoch,
            },
            self.out_dir / name,
        )
