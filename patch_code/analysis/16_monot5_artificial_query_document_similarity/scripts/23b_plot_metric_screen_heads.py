"""
scripts/23b_plot_metric_screen_heads.py
========================================
Plots only (no model, no recomputation): the fig_paired_heads_all_layers.png layout —
relevant / non-relevant panels, padded control vs successful attack, all 144 heads,
instance-level mean +-1 SE (query clusters) — once per stage-23 metric, on the stage-22
token sample. Full-document region; values from 23_metric_screen/metrics.npz.

Outputs: 23_metric_screen/plots/heads_control_vs_attack_{metric}.png
         23_metric_screen/heads_control_vs_attack.csv
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib import paired as P  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.heads import encoder_heads  # noqa: E402
from exp16lib.run_utils import is_already_successful, stage_argparser  # noqa: E402
from exp16lib.token_sample import TokenSample  # noqa: E402

_spec = importlib.util.spec_from_file_location("p21", EXP_DIR / "scripts" / "21_plot_paired.py")
P21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P21)
_spec = importlib.util.spec_from_file_location("p23", EXP_DIR / "scripts" / "23_screen_token_metrics.py")
P23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P23)
P5, P10 = P21.P5, P21.P10


def main():
    args = stage_argparser("Exp 16 stage 23b: per-metric control-vs-attack head plots").parse_args()
    cfg = load_config(args.config)
    P.configure("all_layers")
    out = output_dir(cfg)
    st = out / P23.STAGE
    if not is_already_successful(st, ["metrics.npz"]):
        raise FileNotFoundError(f"run stage 23 first ({st})")
    z = np.load(st / "metrics.npz")
    M, cols, labels = z["M"], list(z["columns"]), list(z["heads"])
    meta = TokenSample(out / "22_token_sample").meta
    assert (z["seq_id"] == meta.seq_id.values).all() and labels == P.head_labels()
    canon = {h.label for h in encoder_heads()}

    succ = meta[(meta.kind == "attack") & (meta.successful == True)]  # noqa: E712
    ctl = meta[meta.kind == "control"].set_index(["attack_name", "pair_id"])
    ia = succ.seq_id.values
    ic = ctl.loc[list(zip(succ.attack_name, succ.pair_id)), "seq_id"].values
    assert (meta.pair_id.values[ic] == succ.pair_id.values).all()
    rows = []
    for m in P23.MAIN:
        c = cols.index(P23.col(m))
        fig, axes = plt.subplots(2, 1, figsize=(28, 9.6), squeeze=False)
        for r, g in enumerate(P.GROUPS):
            k = succ.relevance_group.values == g
            q = succ.qid.values[k]
            tab = {"head_label": labels}
            for side, idx in (("control", ic[k]), ("attack", ia[k])):
                X = M[idx, c].astype(np.float64)
                tab[f"{side}_mean"] = X.mean(0)
                tab[f"{side}_se"] = P23.cluster_se_mat(X, q)
            t = pd.DataFrame(tab)
            rows.append(t.assign(metric=m, relevance_group=g, n_instances=int(k.sum()),
                                 n_docs=succ.pair_id[k].nunique(), n_queries=len(set(q))))
            ax = axes[r, 0]
            P10.draw(ax, t, P21.SIDES)
            P21.head_axis(ax, labels, canon)
            ax.set_ylabel(P23.NAMES[m])
            ax.legend(loc="lower left", fontsize=8)
            ax.set_title(f"{P.GROUP_LABEL[g]}: {int(k.sum())} successful instances / {succ.pair_id[k].nunique()} docs "
                         f"/ {len(set(q))} queries; ±1 SE (query clusters)", loc="left", fontsize=9, pad=16)
        axes[-1, 0].set_xlabel("encoder head (bold = one of the 18 canonical important heads)")
        fig.suptitle(f"{P23.NAMES[m]} across all encoder heads, layers 0–11 (144 heads): padded control vs successful attack "
                     f"— stage-22 token sample, full document", x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        P5._save(fig, st / "plots" / f"heads_control_vs_attack_{m}.png")
        print(f"[23b] {m}")
    pd.concat(rows, ignore_index=True).to_csv(st / "heads_control_vs_attack.csv", index=False)
    print(f"[23b] -> {st / 'plots'}")


if __name__ == "__main__":
    main()
