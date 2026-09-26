"""
scripts/12_plot_decoder_probe_similarity.py
============================================
Decoder counterpart of the encoder four-population level plot (scripts/06),
no model; reads 11_decoder_probe/ only.

y = cos(decoder state with cross-attention restricted to the query text,
        decoder state with cross-attention restricted to the document)
at 37 decoder checkpoints (embedding + post self-attn / post cross-attn /
post MLP per layer), for:
  clean, genuinely relevant (qrel 2/3) and clean, non-relevant (qrel 0):
      per-query mean, equal query weight, +-1 SE across queries
  attacked successful (example-level delta_score > 0) / unsuccessful (<= 0):
      attacked input; mean within attack, equal attack weight, +-1 SE across attacks

Outputs: plots/fig_decoder_probe_similarity_four_populations.png
         04_analysis/decoder_probe_similarity_four_populations.csv
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

from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.decoder_probe import DECODER_PROBE_CHECKPOINTS as CK  # noqa: E402
from exp16lib.run_utils import is_already_successful, stage_argparser  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp16_05_plot", EXP_DIR / "scripts" / "05_plot.py")
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)

DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str}
X = np.arange(len(CK))
LABELS = ["emb"] + [f"D{int(c[1:3])} {c.split('_post_')[1].replace('self_attn', 'self').replace('cross_attn', 'x-attn')}"
                    for c in CK[1:]]


def _se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def main():
    args = stage_argparser("Exp 16: decoder probe similarity, four populations").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    st = out / "11_decoder_probe"
    if not is_already_successful(st, ["qrel_decoder_probe.csv.gz", "attack_decoder_probe.csv.gz"]):
        raise FileNotFoundError(f"run stage 11 first ({st})")
    q = pd.read_csv(st / "qrel_decoder_probe.csv.gz", dtype=DT)
    a = pd.read_csv(st / "attack_decoder_probe.csv.gz", dtype=DT)
    if not np.allclose(a.delta_score, a.score_attack - a.score_control, rtol=0, atol=1e-9):
        raise RuntimeError("delta_score != score_attack - score_control")
    succ = a.delta_score > 0

    rows = []
    pq = q.groupby(["qid", "relevance_group"])[CK].mean().reset_index()
    pa_s = a[succ].groupby("attack_name")[CK].mean()
    pa_u = a[~succ].groupby("attack_name")[CK].mean()
    for i, c in enumerate(CK):
        r = {"checkpoint_index": i, "checkpoint": c}
        for name, grp in (("genuine", "relevant"), ("clean_nonrelevant", "nonrelevant")):
            v = pq[pq.relevance_group == grp][c]
            r[f"{name}_mean"], r[f"{name}_se"] = float(v.mean()), _se(v)
        for name, pa in (("successful_attack", pa_s), ("unsuccessful_attack", pa_u)):
            r[f"{name}_mean"], r[f"{name}_se"] = float(pa[c].mean()), _se(pa[c])
        rows.append(r)
    t = pd.DataFrame(rows)
    t["n_queries"] = pq.qid.nunique()
    t["n_successful_attacks"], t["n_successful_attack_examples"] = len(pa_s), int(succ.sum())
    t["n_unsuccessful_attacks"], t["n_unsuccessful_attack_examples"] = len(pa_u), int((~succ).sum())
    csv_path = out / "04_analysis" / "decoder_probe_similarity_four_populations.csv"
    t.to_csv(csv_path, index=False)

    nq = int(t.n_queries.iloc[0])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    specs = [("genuine", f"clean, genuinely relevant (qrel 2/3; {nq} queries)", P5.BLUE, "o", "-"),
             ("clean_nonrelevant", f"clean, non-relevant (qrel 0; {nq} queries)", P5.BLUE, "D", "--"),
             ("successful_attack", f"attacked, successful (Δscore > 0; {int(succ.sum()):,} ex., {len(pa_s)} attacks)",
              P5.ORANGE, "s", "-"),
             ("unsuccessful_attack", f"attacked, unsuccessful (Δscore ≤ 0; {int((~succ).sum()):,} ex., {len(pa_u)} attacks)",
              P5.ORANGE, "^", "--")]
    for name, label, color, marker, ls in specs:
        y, se = t[f"{name}_mean"].values, t[f"{name}_se"].values
        ax.fill_between(X, y - se, y + se, color=color, alpha=0.15, lw=0)
        ax.plot(X, y, color=color, ls=ls, lw=2, label=label, zorder=3)
        ax.scatter(X, y, s=20, color=color, edgecolor="white", linewidth=1, zorder=4, marker=marker)
    for L in range(12):                                       # thin separators between decoder layers
        ax.axvline(3 * L + 0.5, color=P5.GRID, lw=1.2, zorder=1)
    ax.set_xticks(X)
    ax.set_xticklabels(LABELS, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(X) - 0.4)
    ax.set_xlabel("decoder checkpoint (first decoder step)")
    ax.set_ylabel("cos(decoder query repr., decoder document repr.)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Decoder query-document similarity: genuine relevance vs attacks\n"
                 "query repr. = decoder state with cross-attention restricted to the query text; document repr. = "
                 "restricted to the document; ±1 SE", loc="left", fontsize=9)
    png = out / "plots" / "fig_decoder_probe_similarity_four_populations.png"
    P5._save(fig, png)
    print(f"[12] -> {png}\n  -> {csv_path}")


if __name__ == "__main__":
    main()
