"""
scripts/08_plot_decoder_message_similarity.py
==============================================
Decoder counterpart of scripts/06 (no model; reads 07_decoder/ only).

Per decoder layer, the cosine between the cross-attention message read from
query-text positions and the message read from document positions
(exp16lib.decoder), for four populations:
  1. clean, genuinely relevant (qrel 2/3)   per-query mean, equal query weight, +-1 SE across queries
  2. clean, non-relevant (qrel 0)           same
  3. attacked, successful   (example-level delta_score > 0)   attacked input only (a_msg_cos);
  4. attacked, unsuccessful (delta_score <= 0)                mean within attack, equal attack weight,
                                                              +-1 SE across attacks
Outputs: plots/fig_decoder_message_similarity_four_populations.png
         04_analysis/decoder_message_similarity_four_populations.csv
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
from exp16lib.decoder import DECODER_CHECKPOINTS  # noqa: E402
from exp16lib.run_utils import is_already_successful, stage_argparser  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp16_05_plot", EXP_DIR / "scripts" / "05_plot.py")
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)

DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str}
X = np.arange(len(DECODER_CHECKPOINTS))


def _se(x: pd.Series) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def _line(ax, y, se, color, label, marker, ls):
    ax.fill_between(X, y - se, y + se, color=color, alpha=0.15, lw=0)
    ax.plot(X, y, color=color, ls=ls, lw=2, label=label, zorder=3)
    ax.scatter(X, y, s=24, color=color, edgecolor="white", linewidth=1, zorder=4, marker=marker)


def main():
    args = stage_argparser("Exp 16: decoder message similarity, four populations").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    st = out / "07_decoder"
    if not is_already_successful(st, ["qrel_decoder.csv.gz", "attack_decoder.csv.gz"]):
        raise FileNotFoundError(f"run stage 07 first ({st})")
    q = pd.read_csv(st / "qrel_decoder.csv.gz", dtype=DT)
    a = pd.read_csv(st / "attack_decoder.csv.gz", dtype=DT)
    if not np.allclose(a.delta_score, a.score_attack - a.score_control, rtol=0, atol=1e-9):
        raise RuntimeError("delta_score != score_attack - score_control")

    def query_level(grp, name):
        per_q = q[q.relevance_group == grp].groupby(["qid", "decoder_layer"]).msg_cos.mean().reset_index()
        return per_q.groupby("decoder_layer").msg_cos.agg(**{f"{name}_mean": "mean", f"{name}_se": _se,
                                                             f"n_queries_{name}": "size"})

    def attack_level(mask, name):
        per_a = a[mask].groupby(["attack_name", "decoder_layer"]).a_msg_cos.agg(["mean", "size"]).reset_index()
        return per_a.groupby("decoder_layer").agg(**{f"{name}_mean": ("mean", "mean"), f"{name}_se": ("mean", _se),
                                                     f"n_{name}s": ("attack_name", "nunique"),
                                                     f"n_{name}_examples": ("size", "sum")})

    parts = [query_level("relevant", "genuine"), query_level("nonrelevant", "clean_nonrelevant"),
             attack_level(a.delta_score > 0, "successful_attack"), attack_level(a.delta_score <= 0, "unsuccessful_attack")]
    tab = parts[0].join(parts[1:]).reset_index()
    tab = tab.rename(columns={"n_queries_genuine": "n_queries"}).drop(columns=["n_queries_clean_nonrelevant"])
    tab.insert(1, "checkpoint", [DECODER_CHECKPOINTS[i] for i in tab.decoder_layer])
    csv_path = out / "04_analysis" / "decoder_message_similarity_four_populations.csv"
    tab.to_csv(csv_path, index=False)

    nq = int(tab.n_queries.iloc[0])
    ns, nse = int(tab.n_successful_attacks.iloc[0]), int(tab.n_successful_attack_examples.iloc[0])
    nu, nue = int(tab.n_unsuccessful_attacks.iloc[0]), int(tab.n_unsuccessful_attack_examples.iloc[0])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    _line(ax, tab.genuine_mean.values, tab.genuine_se.values, P5.BLUE,
          f"clean, genuinely relevant (qrel 2/3; {nq} queries)", "o", "-")
    _line(ax, tab.clean_nonrelevant_mean.values, tab.clean_nonrelevant_se.values, P5.BLUE,
          f"clean, non-relevant (qrel 0; {nq} queries)", "D", "--")
    _line(ax, tab.successful_attack_mean.values, tab.successful_attack_se.values, P5.ORANGE,
          f"attacked, successful (Δscore > 0; {nse:,} ex., {ns} attacks)", "s", "-")
    _line(ax, tab.unsuccessful_attack_mean.values, tab.unsuccessful_attack_se.values, P5.ORANGE,
          f"attacked, unsuccessful (Δscore ≤ 0; {nue:,} ex., {nu} attacks)", "^", "--")
    ax.set_xticks(X)
    ax.set_xticklabels([f"D{i} x-attn" for i in X], rotation=90, fontsize=7)
    ax.set_xlim(-0.5, len(X) - 0.5)
    ax.set_xlabel("decoder layer (cross-attention output)")
    ax.set_ylabel("cos(query-sourced msg, document-sourced msg)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Decoder: query vs document information read through cross-attention\n"
                 "±1 SE across queries (clean qrel docs) / across attack configurations (attacked docs)",
                 loc="left", fontsize=9)
    png = out / "plots" / "fig_decoder_message_similarity_four_populations.png"
    P5._save(fig, png)
    print(f"[08] successful {nse} ex / {ns} attacks; unsuccessful {nue} ex / {nu} attacks; {nq} queries\n"
          f"  -> {png}\n  -> {csv_path}")


if __name__ == "__main__":
    main()
