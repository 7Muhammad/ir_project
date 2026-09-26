#!/usr/bin/env python3
"""
scripts/make_heatmap_matrix.py
================================
Standalone script: stitch existing per-attack heatmap PNGs into one
big matrix figure.

  Rows    = unique (token, position) e.g. "relevant_start"
            each row has 3 sub-rows: forward / reverse / combined
  Columns = repetition count 1, 2, 3, 4, 5

Run (no heavy deps, just matplotlib):
    conda run -n advseq2seq python scripts/make_heatmap_matrix.py
"""

from __future__ import annotations

import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
ATTACKS_DIR   = pathlib.Path(__file__).parent.parent / "outputs" / "attacks"
OUTPUT_PATH   = pathlib.Path(__file__).parent.parent / "outputs" / "attack_comparison" / "heatmap_matrix.png"

DIRECTIONS = [
    ("forward",  "forward_layer_component_heatmap.png",  "#2166ac"),
    ("reverse",  "reverse_layer_component_heatmap.png",  "#d6604d"),
    ("combined", "combined_layer_component_heatmap.png", "#1a9641"),
]
# ---------------------------------------------------------------------------

def main() -> None:
    # --- discover all attack folders that have plots -----------------------
    attack_dirs = sorted(p for p in ATTACKS_DIR.iterdir() if p.is_dir())

    # parse name -> (token, position, rep)
    pattern = re.compile(r"^(.+)_(start|end|random)_(\d+)$")
    entries = []
    for d in attack_dirs:
        m = pattern.match(d.name)
        if m:
            token, position, rep = m.group(1), m.group(2), int(m.group(3))
            entries.append((token, position, rep, d))

    if not entries:
        raise RuntimeError(f"No attack folders found in {ATTACKS_DIR}")

    rep_counts = sorted({e[2] for e in entries})
    row_keys   = sorted({f"{e[0]}_{e[1]}" for e in entries})

    # index: (row_key, rep) -> attack Path
    idx = {}
    for token, position, rep, d in entries:
        idx[(f"{token}_{position}", rep)] = d

    n_groups = len(row_keys)
    n_dirs   = len(DIRECTIONS)
    n_cols   = len(rep_counts)
    n_rows   = n_groups * n_dirs

    cell_w, cell_h = 3.0, 2.4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w + 1.2, n_rows * cell_h + 0.9),
        squeeze=False,
    )

    for g_idx, row_key in enumerate(row_keys):
        for d_idx, (dir_label, png_name, color) in enumerate(DIRECTIONS):
            r_idx = g_idx * n_dirs + d_idx

            for c_idx, rep in enumerate(rep_counts):
                ax = axes[r_idx][c_idx]
                attack_dir = idx.get((row_key, rep))
                png_path   = attack_dir / "plots" / png_name if attack_dir else None

                if png_path is None or not png_path.exists():
                    ax.set_facecolor("#eeeeee")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="grey")
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    img = mpimg.imread(str(png_path))
                    ax.imshow(img, aspect="auto", interpolation="lanczos")
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.4)

                # column header (top row only)
                if r_idx == 0:
                    ax.set_title(f"rep = {rep}", fontsize=9, pad=4)

                # direction label (left column only)
                if c_idx == 0:
                    ax.set_ylabel(dir_label, fontsize=7, labelpad=3,
                                  color=color, fontweight="bold")

        # token_position label: rotated, centred vertically on the middle
        # direction sub-row, placed to the left of the axes
        mid_r = g_idx * n_dirs + 1
        axes[mid_r][0].annotate(
            row_key,
            xy=(-0.52, 0.5),
            xycoords="axes fraction",
            fontsize=7.5,
            ha="right", va="center",
            rotation=90,
            annotation_clip=False,
            fontweight="bold",
        )

        # thick separator between groups
        if g_idx < n_groups - 1:
            sep_r = (g_idx + 1) * n_dirs
            for c_idx in range(n_cols):
                axes[sep_r][c_idx].spines["top"].set_linewidth(2.0)
                axes[sep_r][c_idx].spines["top"].set_color("#333333")

    fig.suptitle(
        "Patching Heatmaps — Rows: token×position (fwd / rev / combined)   Cols: repetitions",
        fontsize=11, y=1.002,
    )
    plt.tight_layout(h_pad=0.15, w_pad=0.15)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
