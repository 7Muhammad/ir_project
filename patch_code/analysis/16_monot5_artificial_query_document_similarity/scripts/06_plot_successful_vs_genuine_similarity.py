"""
scripts/06_plot_successful_vs_genuine_similarity.py
====================================================
Supplementary figure (no model; reads existing Exp 16 outputs only): the
absolute query-document similarity LEVEL at the 25 encoder checkpoints for

  1. genuine relevant inputs: clean prompts with TREC DL19 qrel 2/3 documents
     (02_qrel); mean within query, then equal weight per query; +-1 SE across queries;
  2. clean non-relevant inputs: clean prompts with the balanced qrel 0 documents
     of the same queries (02_qrel); same weighting and band;
  3. successful attacked inputs: example-level delta_score = score_attack -
     score_control > 0 (03_attack; every attack configuration contributes its
     successful examples, including configurations negative on average);
  4. unsuccessful attacked inputs: delta_score <= 0.
     For 3/4: sim_attack only (injected tokens are in the document mean, the
     Exp 16 convention); mean within attack, then equal weight per attack;
     +-1 SE across attacks.

This is a level comparison, NOT a gap (no rel - nonrel, no attack - control).
The success filter is applied only here, for this descriptive figure; the
primary Exp 16 analyses remain unfiltered (DECISIONS 20).

Outputs: plots/fig_successful_attacks_vs_genuine_similarity.png
         04_analysis/successful_attacks_vs_genuine_similarity.csv
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

from exp16lib.checkpoints import CHECKPOINTS  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.run_utils import is_already_successful, stage_argparser  # noqa: E402

_spec = importlib.util.spec_from_file_location("exp16_05_plot", EXP_DIR / "scripts" / "05_plot.py")
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)   # shared style + helpers (_line, _axis, palette)

DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str}


def _se(x: pd.Series) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def main():
    args = stage_argparser("Exp 16: genuine / non-relevant / successful / unsuccessful similarity levels").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    for d, f in ((out / "02_qrel", "qrel_similarity.csv.gz"), (out / "03_attack", "attack_similarity.csv.gz")):
        if not is_already_successful(d, [f]):
            raise FileNotFoundError(f"upstream stage incomplete: {d / f}")

    # 1/2. clean qrel inputs: per-query mean within group, then equal query weight
    q = pd.read_csv(out / "02_qrel" / "qrel_similarity.csv.gz", dtype=DT)
    for grp, grades in (("relevant", [2, 3]), ("nonrelevant", [0])):
        if not q[q.relevance_group == grp].qrel_grade.isin(grades).all():
            raise RuntimeError(f"{grp} group contains grades outside {grades}")

    def query_level(grp: str, name: str) -> pd.DataFrame:
        per_q = q[q.relevance_group == grp].groupby(["qid", "checkpoint_index"]).similarity.mean().reset_index()
        return per_q.groupby("checkpoint_index").similarity.agg(
            **{f"{name}_mean": "mean", f"{name}_se": _se, f"n_queries_{name}": "size"})

    # 3/4. attacked inputs split by example-level success, attack-level then equal attack weight
    a = pd.read_csv(out / "03_attack" / "attack_similarity.csv.gz", dtype=DT)
    if not np.allclose(a.delta_score, a.score_attack - a.score_control, rtol=0, atol=1e-9):
        raise RuntimeError("delta_score != score_attack - score_control")

    def attack_level(mask: pd.Series, name: str) -> pd.DataFrame:
        per_a = a[mask].groupby(["attack_name", "checkpoint_index"]).sim_attack.agg(["mean", "size"]).reset_index()
        return per_a.groupby("checkpoint_index").agg(
            **{f"{name}_mean": ("mean", "mean"), f"{name}_se": ("mean", _se),
               f"n_{name}s": ("attack_name", "nunique"), f"n_{name}_examples": ("size", "sum")})

    parts = [query_level("relevant", "genuine"), query_level("nonrelevant", "clean_nonrelevant"),
             attack_level(a.delta_score > 0, "successful_attack"), attack_level(a.delta_score <= 0, "unsuccessful_attack")]
    tab = parts[0].join(parts[1:]).reset_index()
    tab = tab.rename(columns={"n_queries_genuine": "n_queries"}).drop(columns=["n_queries_clean_nonrelevant"])
    tab.insert(1, "checkpoint", [CHECKPOINTS[i] for i in tab.checkpoint_index])
    if tab.checkpoint.tolist() != CHECKPOINTS:
        raise RuntimeError("checkpoint set/order mismatch")
    csv_path = out / "04_analysis" / "successful_attacks_vs_genuine_similarity.csv"
    tab.to_csv(csv_path, index=False)

    nq = int(tab.n_queries.iloc[0])
    ns, nse = int(tab.n_successful_attacks.iloc[0]), int(tab.n_successful_attack_examples.iloc[0])
    nu, nue = int(tab.n_unsuccessful_attacks.iloc[0]), int(tab.n_unsuccessful_attack_examples.iloc[0])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    # colour = population (blue: clean judged qrel docs, orange: attacked docs); line style = outcome
    P5._line(ax, tab.genuine_mean.values, P5.BLUE, f"clean, genuinely relevant (qrel 2/3; {nq} queries)",
             se=tab.genuine_se.values)
    P5._line(ax, tab.clean_nonrelevant_mean.values, P5.BLUE, f"clean, non-relevant (qrel 0; {nq} queries)",
             se=tab.clean_nonrelevant_se.values, marker="D", ls="--")
    P5._line(ax, tab.successful_attack_mean.values, P5.ORANGE,
             f"attacked, successful (Δscore > 0; {nse:,} ex., {ns} attacks)",
             se=tab.successful_attack_se.values, marker="s")
    P5._line(ax, tab.unsuccessful_attack_mean.values, P5.ORANGE,
             f"attacked, unsuccessful (Δscore ≤ 0; {nue:,} ex., {nu} attacks)",
             se=tab.unsuccessful_attack_se.values, marker="^", ls="--")
    P5._axis(ax, "cosine(query representation, document representation)", zero=False)
    ax.set_xlabel("encoder checkpoint")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Query-document similarity: genuine relevance vs attacks\n"
                 "±1 SE across queries (clean qrel docs) / across attack configurations (attacked docs)",
                 loc="left", fontsize=9)
    png = out / "plots" / "fig_successful_attacks_vs_genuine_similarity.png"
    P5._save(fig, png)
    print(f"[06] successful {nse} ex / {ns} attacks; unsuccessful {nue} ex / {nu} attacks; {nq} queries\n"
          f"  -> {png}\n  -> {csv_path}")


if __name__ == "__main__":
    main()
