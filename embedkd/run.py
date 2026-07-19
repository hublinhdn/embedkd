"""DistillationRun: the single entry point shared by the Python API and the CLI.

The CLI is a thin shell around this class, so numbers can never differ
between the two ways of using the toolkit.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .config import resolve
from .data import ImageTransform, build_bundle
from .diagnostics import compatibility_report
from .engine import Trainer, set_seed, write_fingerprint
from .evaluation import evaluate_model
from .losses import build_task_loss
from .models import EmbedHead, EmbeddingModel, create_backbone
from .objectives import build_objective


def _load_teacher_weights(model: EmbeddingModel, spec: str) -> None:
    if spec in ("random", "pretrained"):
        return  # 'pretrained' is handled at backbone creation time
    state = torch.load(spec, map_location="cpu", weights_only=True)
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(
            f"Teacher checkpoint '{spec}' does not match the configured architecture. "
            f"Missing keys (first 5): {missing[:5]}"
        )
    # A checkpoint may carry a classifier the current objective does not need
    # (e.g. teacher trained with sce, distilled with cosine): ignore it.
    leftover = [k for k in unexpected if not k.startswith("classifier.")]
    if leftover:
        raise RuntimeError(
            f"Teacher checkpoint '{spec}' has unexpected keys (first 5): {leftover[:5]}"
        )


class DistillationRun:
    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg = cfg
        set_seed(cfg["train"]["seed"])
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        data_cfg = cfg["data"]
        train_tf = ImageTransform(data_cfg["input_size"], train=True)
        eval_tf = ImageTransform(data_cfg["input_size"], train=False)
        self.bundle = build_bundle(data_cfg, train_tf, eval_tf)

        policy = cfg["backbone_policy"]
        t_backbone, t_dim = create_backbone(
            cfg["teacher"]["backbone"],
            pretrained=(cfg["teacher"]["weights"] == "pretrained"),
            policy=policy,
            output_stride=cfg["teacher"].get("output_stride"),
        )
        s_backbone, s_dim = create_backbone(
            cfg["student"]["backbone"], pretrained=cfg["student"]["pretrained"], policy=policy,
            output_stride=cfg["student"].get("output_stride"),
        )
        head_cfg = cfg["head"]

        def make_head(in_dim: int, embed_dim: int) -> EmbedHead:
            return EmbedHead(
                in_dim, embed_dim,
                pooling=head_cfg["pooling"], gem_p=head_cfg["gem_p"],
                gem_p_trainable=head_cfg["gem_p_trainable"], normalize=head_cfg["normalize"],
            )

        # Classifiers are built only when an active component needs logits;
        # at SOP-scale class counts an unused classifier head wastes millions
        # of parameters.
        objective_spec = cfg["distill"]["objective"]
        objective_names = (
            {objective_spec} if isinstance(objective_spec, str) else set(objective_spec)
        )
        needs_logits = "kl" in objective_names and float(cfg["distill"]["alpha"]) != 0.0
        num_classes = self.bundle.num_classes
        student_classes = num_classes if ("sce" in head_cfg["losses"] or needs_logits) else None
        teacher_classes = num_classes if needs_logits else None

        logit_scale = float(head_cfg.get("logit_scale", 64.0))
        self.teacher = EmbeddingModel(
            t_backbone, make_head(t_dim, cfg["teacher"]["embed_dim"]), teacher_classes,
            logit_scale=logit_scale,
        )
        if float(cfg["distill"]["alpha"]) != 0.0:
            _load_teacher_weights(self.teacher, cfg["teacher"]["weights"])
        # alpha == 0 is standalone training: the teacher is unused, so a
        # placeholder weights path in the config must not be an error.
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.student = EmbeddingModel(
            s_backbone, make_head(s_dim, cfg["student"]["embed_dim"]), student_classes,
            logit_scale=logit_scale,
        )
        self.objective = build_objective(cfg["distill"])
        self.task_loss = build_task_loss(head_cfg, cfg["student"]["embed_dim"], num_classes)

        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{cfg['run']['tag']}"
        self.out_dir = Path(cfg["run"]["output_dir"]) / run_id
        self._trainer: Trainer | None = None

    @classmethod
    def from_config(cls, path: str | Path, overrides: list[str] | dict | None = None,
                    device: str | None = None) -> "DistillationRun":
        return cls(resolve(path, overrides), device=device)

    def fit(self) -> dict:
        write_fingerprint(self.cfg, self.out_dir)
        self._trainer = Trainer(
            self.cfg, self.student, self.teacher, self.objective, self.task_loss,
            self.bundle, self.out_dir, device=self.device,
        )
        return self._trainer.fit()

    def evaluate(self, checkpoint: str | Path | None = None, target: bool = False) -> dict:
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.student.load_state_dict(state["state_dict"])
        gallery = self.bundle.target_gallery if target else self.bundle.gallery
        query = self.bundle.target_query if target else self.bundle.query
        if gallery is None or query is None:
            raise ValueError("No target split configured (data.protocol: cross_domain)")
        metrics = evaluate_model(
            self.student.to(self.device), gallery, query,
            batch_size=self.cfg["eval"]["batch_size"], device=self.device,
        )
        if self.cfg["eval"].get("report_retention"):
            teacher_metrics = evaluate_model(
                self.teacher.to(self.device), gallery, query,
                batch_size=self.cfg["eval"]["batch_size"], device=self.device,
            )
            if teacher_metrics["map"] > 0:
                metrics["teacher_map"] = teacher_metrics["map"]
                metrics["retention"] = round(metrics["map"] / teacher_metrics["map"], 4)
        return metrics

    def diagnose(self) -> dict:
        return compatibility_report(
            self.teacher.to(self.device), self.student.to(self.device),
            self.bundle.query, batch_size=self.cfg["eval"]["batch_size"], device=self.device,
        )
