#!/usr/bin/env python
"""Render a qualitative retrieval figure: query images next to their top-k
gallery neighbours, one row per checkpoint, correct hits framed in blue and
misses in vermillion (Okabe-Ito, colorblind-safe, matching the paper
figures).

Typical use, comparing the no-KD baseline against the distilled student:

    python scripts/make_qualitative_figure.py \\
        --config configs/d2_cars196_cosine.yaml \\
        --checkpoint "no KD=runs/<id_nokd>/best.pth" \\
        --checkpoint "distilled=runs/<id_cosine>/best.pth" \\
        --out fig_qualitative.pdf

Both checkpoints must share the student architecture of --config. Query
selection is deterministic given --seed. --select improved (default) picks,
in shuffled order, queries where the LAST named checkpoint ranks a correct
image first and the FIRST named one does not. --select happy draws from the
same improved pool but orders it to show the cleanest, most eye-catching
cases first (distilled row correct across the most of top-k, no-KD row wrong
across the most of top-k); use it for illustration slides and captions where
a vivid before/after helps. The printed summary reports how common the
improved case is, so the caption can still state the selection honestly.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from embedkd.config import resolve
from embedkd.evaluation.retrieval import extract_embeddings
from embedkd.run import DistillationRun

BLUE = "#0072B2"       # correct neighbour (Okabe-Ito blue, as in the paper)
VERMILLION = "#D55E00"  # wrong neighbour
INK = "#20303C"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", action="append", required=True,
                   metavar="NAME=PATH",
                   help="repeatable; row order in the figure follows CLI order")
    p.add_argument("--out", default="fig_qualitative.pdf")
    p.add_argument("--num-queries", type=int, default=3)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--select", choices=["improved", "happy", "random", "first"],
                   default="improved")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--thumb", type=int, default=196, help="thumbnail size in px")
    p.add_argument("--set", action="append", default=[], help="config overrides")
    return p.parse_args()


def load_thumb(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size))


def pretty_label(name: str) -> str:
    # "001.Black_footed_Albatross" -> "Black footed Albatross"
    name = name.split(".", 1)[-1] if name.split(".", 1)[0].isdigit() else name
    return name.replace("_", " ")


def main() -> int:
    args = parse_args()
    names, paths = [], []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint expects NAME=PATH, got: {spec}")
        name, _, path = spec.partition("=")
        names.append(name)
        paths.append(path)

    # Student-only workload: alpha=0 makes DistillationRun skip loading the
    # teacher checkpoint (same mechanism as `reproduce --eval-only`), so the
    # config's teacher.weights path does not need to exist.
    overrides = list(args.set) + ["distill.alpha=0"]
    run = DistillationRun(resolve(args.config, overrides), device=args.device)
    gallery, query = run.bundle.gallery, run.bundle.query
    for ds, split in ((gallery, "gallery"), (query, "query")):
        if not hasattr(ds, "items"):
            raise SystemExit(f"{split} split has no file-backed items; "
                             "qualitative rendering needs an on-disk dataset")

    # Index -> class name, for titles. Folder names only identify the class in
    # class-per-folder layouts (CUB, image_folder); flat layouts such as
    # Cars196's car_ims/ give every query the same meaningless folder title,
    # so fall back to a plain label when folder names collide across classes.
    idx_to_name = {}
    for path, idx in gallery.items + query.items:
        idx_to_name.setdefault(idx, Path(path).parent.name)
    if len(set(idx_to_name.values())) != len(idx_to_name):
        idx_to_name = {idx: "" for idx in idx_to_name}

    g_labels = q_labels = None
    orders = []
    for name, ckpt in zip(names, paths):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        run.student.load_state_dict(state["state_dict"])
        model = run.student.to(args.device)
        g_emb, g_labels = extract_embeddings(model, gallery, args.batch_size, args.device)
        q_emb, q_labels = extract_embeddings(model, query, args.batch_size, args.device)
        orders.append((q_emb @ g_emb.t()).argsort(dim=1, descending=True))
        print(f"embedded {len(q_labels)} queries / {len(g_labels)} gallery with '{name}'")

    correct1 = [g_labels[o[:, 0]] == q_labels for o in orders]
    # Per-query count of correct neighbours within top-k, per model; used to
    # rank the 'happy' selection toward clean distilled rows and wide gaps.
    topk_hits = [(g_labels[o[:, :args.topk]] == q_labels.unsqueeze(1)).sum(dim=1)
                 for o in orders]
    n_q = len(q_labels)
    for name, c in zip(names, correct1):
        print(f"top-1 correct with '{name}': {int(c.sum())}/{n_q} ({c.float().mean():.1%})")

    rng = random.Random(args.seed)
    if args.select in ("improved", "happy") and len(orders) >= 2:
        eligible = torch.nonzero(~correct1[0] & correct1[-1]).flatten().tolist()
        print(f"queries where '{names[-1]}' fixes the top-1 of '{names[0]}': "
              f"{len(eligible)}/{n_q}")
    else:
        eligible = list(range(n_q))
    if args.select == "happy" and len(orders) >= 2:
        # Cleanest distilled row first, then widest before/after gap. Deterministic.
        eligible.sort(key=lambda i: (int(topk_hits[-1][i]), -int(topk_hits[0][i])),
                      reverse=True)
    elif args.select != "first":
        rng.shuffle(eligible)

    # Prefer class-diverse queries so the figure is not three birds of a
    # feather; top up from the remaining pool if the eligible set runs short.
    chosen, seen = [], set()

    def pick(pool: list[int]) -> None:
        for prefer_unseen in (True, False):
            for i in pool:
                if len(chosen) == args.num_queries:
                    return
                if i in chosen or (prefer_unseen and int(q_labels[i]) in seen):
                    continue
                chosen.append(i)
                seen.add(int(q_labels[i]))

    pick(eligible)
    if len(chosen) < args.num_queries:
        rest = [i for i in range(n_q) if i not in eligible]
        if args.select != "first":
            rng.shuffle(rest)
        pick(rest)
    if len(chosen) < args.num_queries:
        raise SystemExit(f"only {len(chosen)} queries available, "
                         f"asked for {args.num_queries}")

    from matplotlib.gridspec import GridSpec

    rows_per_query = len(names)
    n_rows = args.num_queries * rows_per_query
    n_cols = args.topk + 1
    cell = 1.15
    group_gap = 0.5  # blank vertical band between query groups (fraction of a cell)

    # A thin empty spacer row between consecutive query groups makes each
    # query's rows read as one block, set apart from the next query.
    height_ratios, row_map = [], []
    for qi in range(args.num_queries):
        for _ in range(rows_per_query):
            row_map.append(len(height_ratios))
            height_ratios.append(1.0)
        if qi < args.num_queries - 1:
            height_ratios.append(group_gap)

    fig_h = sum(height_ratios) * cell + 0.3
    fig = plt.figure(figsize=(n_cols * cell + 1.0, fig_h))
    gs = GridSpec(len(height_ratios), n_cols, figure=fig,
                  height_ratios=height_ratios, hspace=0.1, wspace=0.08)
    axes = [[fig.add_subplot(gs[row_map[r], c]) for c in range(n_cols)]
            for r in range(n_rows)]

    for qi, q_idx in enumerate(chosen):
        q_path, q_cls = query.items[q_idx]
        for mi, (name, order) in enumerate(zip(names, orders)):
            r = qi * rows_per_query + mi
            ax = axes[r][0]
            ax.imshow(load_thumb(q_path, args.thumb))
            ax.set_ylabel(name, fontsize=9, color=INK)
            if mi == 0:
                label = pretty_label(idx_to_name[q_cls])
                ax.set_title(f"query: {label}" if label else "query",
                             fontsize=9, loc="left", color=INK)
            for k in range(args.topk):
                g_idx = int(order[q_idx, k])
                g_path, g_cls = gallery.items[g_idx]
                axk = axes[r][k + 1]
                axk.imshow(load_thumb(g_path, args.thumb))
                good = g_cls == q_cls
                for spine in axk.spines.values():
                    spine.set_edgecolor(BLUE if good else VERMILLION)
                    spine.set_linewidth(2.4)
                axk.text(0.04, 0.04, "✓" if good else "✗",
                         transform=axk.transAxes, fontsize=10, fontweight="bold",
                         color=BLUE if good else VERMILLION, va="bottom")
                if mi == 0 and qi == 0:
                    axk.set_title(f"top-{k + 1}", fontsize=8, color=INK)

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    for row in axes:  # query column carries no verdict frame
        for spine in row[0].spines.values():
            spine.set_edgecolor("#C7D8E6")
            spine.set_linewidth(0.8)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}  ({args.num_queries} queries x "
          f"{len(names)} models x top-{args.topk}, select={args.select}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
