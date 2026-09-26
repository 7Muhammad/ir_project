"""
scripts/04_analyze.py
======================
Stage 04 — all analyses (no model). Output directory: 04_analysis/

A  clean_correlation.csv            Spearman(sim_clean(c), score_clean) per checkpoint (primary);
                                    Pearson + BH over the 25 Spearman p-values (secondary)
B  qrel_per_query_trajectory.csv    per query: sim_rel_q, sim_nonrel_q, delta_sim_real_q (+ steps)
   genuine_trajectory.csv           equal-query-weight global trajectories + delta_step_real
C  attack_per_attack_means.csv      per attack: mean sim_attack/sim_control/delta_sim, mean delta_score
   attack_global_trajectory.csv     equal-attack-weight global trajectories
   attack_accumulation.csv          step_attack, step_control, delta_step (equal attack weight)
   attack_signflip_fdr.csv          one-sided sign-flip over the K attack means + BH over 25 checkpoints
D  delta_sim_vs_delta_score.csv     Spearman across attacks (PRIMARY) + pooled example-level (SECONDARY)
E  genuine_vs_attack.csv            delta_sim_real vs delta_sim_attack, delta_step_real vs delta_step_attack
   attack_breakdowns.csv            descriptive: equal-weight delta_sim by token / position / repetitions
   summary.json                     headline numbers
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib.aggregate import KEY, add_steps, two_level_mean  # noqa: E402
from exp16lib.checkpoints import CHECKPOINTS  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.run_utils import is_already_successful, load_status, now, stage_argparser, write_status  # noqa: E402
from exp16lib.stats import benjamini_hochberg, pearson, sign_flip_test, spearman  # noqa: E402

DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str, "attack_position": str}


def _need(stage_dir: pathlib.Path, f: str) -> pathlib.Path:
    if not is_already_successful(stage_dir, [f]):
        raise FileNotFoundError(f"upstream stage incomplete: {stage_dir / f}")
    return stage_dir / f


def _check_checkpoints(df: pd.DataFrame, name: str, unit: str):
    ck = df.drop_duplicates("checkpoint_index").sort_values("checkpoint_index")["checkpoint_name"].tolist()
    if ck != CHECKPOINTS:
        raise RuntimeError(f"{name}: checkpoints {ck} != the 25 primary checkpoints")
    if (df.groupby(unit).size() % 25 != 0).any():
        raise RuntimeError(f"{name}: some {unit} lack a full set of 25 checkpoints")


def analyze_clean(clean: pd.DataFrame, alpha: float):
    rows = []
    for c, g in clean.groupby("checkpoint_index", sort=True):
        s = spearman(g["similarity"], g["score_clean"])
        pr = pearson(g["similarity"], g["score_clean"])
        rows.append({**g.iloc[0][KEY].to_dict(), "n": s["n"], "spearman_rho": s["rho"], "spearman_p": s["p"],
                     "pearson_r": pr["r"], "pearson_p": pr["p"], "mean_similarity": g["similarity"].mean(),
                     "sd_similarity": g["similarity"].std(ddof=1)})
    out = pd.DataFrame(rows)
    bh = benjamini_hochberg(out["spearman_p"], alpha)
    out["spearman_q_bh_secondary"] = bh["q"]
    out["spearman_reject_bh_secondary"] = bh["reject"]
    return out


def analyze_qrel(q: pd.DataFrame):
    rel = q[q.relevance_group == "relevant"].groupby(["qid"] + KEY)["similarity"].agg(["mean", "size"])
    non = q[q.relevance_group == "nonrelevant"].groupby(["qid"] + KEY)["similarity"].agg(["mean", "size"])
    pq = rel.join(non, lsuffix="_rel", rsuffix="_non", how="inner").reset_index()
    if len(pq) != len(rel) or len(pq) != len(non):
        raise RuntimeError("a query lacks one of the two relevance groups")
    pq = pq.rename(columns={"mean_rel": "sim_rel", "mean_non": "sim_nonrel", "size_rel": "n_rel", "size_non": "n_nonrel"})
    if (pq.n_rel != pq.n_nonrel).any():
        raise RuntimeError("qrel sample is not balanced within query")
    pq["delta_sim_real"] = pq.sim_rel - pq.sim_nonrel
    mono = q.groupby(["qid", "relevance_group"])["monot5_score"].mean().unstack()
    pq = add_steps(pq, ["sim_rel", "sim_nonrel", "delta_sim_real"], ["qid"])
    _, glob = two_level_mean(pq.rename(columns={}), "qid",
                             ["sim_rel", "sim_nonrel", "delta_sim_real", "step_sim_rel", "step_sim_nonrel",
                              "step_delta_sim_real"])
    glob = glob.rename(columns={"step_delta_sim_real": "delta_step_real"})
    glob["frac_queries_delta_positive"] = pq.groupby("checkpoint_index")["delta_sim_real"].apply(lambda s: (s > 0).mean()).values
    return pq, glob, mono


def analyze_attacks(a: pd.DataFrame, cfg: dict):
    per = a.groupby(["attack_id", "attack_name", "attack_token", "attack_position", "repetitions"] + KEY).agg(
        mean_sim_attack=("sim_attack", "mean"), mean_sim_control=("sim_control", "mean"),
        mean_delta_sim=("delta_sim", "mean"), mean_delta_score=("delta_score", "mean"),
        n_examples=("delta_sim", "size"), frac_delta_score_positive=("delta_score", lambda s: (s > 0).mean()),
    ).reset_index()
    per = add_steps(per, ["mean_sim_attack", "mean_sim_control", "mean_delta_sim"], ["attack_id"])
    vals = ["mean_sim_attack", "mean_sim_control", "mean_delta_sim", "mean_delta_score",
            "step_mean_sim_attack", "step_mean_sim_control", "step_mean_delta_sim"]
    _, glob = two_level_mean(per, "attack_id", vals)   # per-attack rows are already attack means
    glob = glob.rename(columns={c: c.replace("mean_", "global_", 1) for c in glob.columns if c.startswith("mean_")})
    glob["frac_attacks_delta_sim_positive"] = per.groupby("checkpoint_index")["mean_delta_sim"].apply(lambda s: (s > 0).mean()).values
    acc = glob[KEY + ["global_sim_attack", "global_sim_control", "global_delta_sim", "step_mean_sim_attack",
                      "step_mean_sim_control", "step_mean_delta_sim", "step_mean_delta_sim_se_units"]].rename(columns={
        "step_mean_sim_attack": "step_attack", "step_mean_sim_control": "step_control",
        "step_mean_delta_sim": "delta_step", "step_mean_delta_sim_se_units": "delta_step_se_attacks"})
    acc["transition"] = ["" if i == 0 else f"{CHECKPOINTS[i - 1]} -> {CHECKPOINTS[i]}" for i in acc.checkpoint_index]
    acc["frac_attacks_delta_step_positive"] = per.groupby("checkpoint_index")["step_mean_delta_sim"].apply(
        lambda s: (s > 0).mean() if s.notna().any() else np.nan).values

    st = cfg["statistics"]
    tests = []
    for c, g in per.groupby("checkpoint_index", sort=True):
        r = sign_flip_test(g.sort_values("attack_id")["mean_delta_sim"].values, int(st["n_sign_flips"]),
                           int(st["sign_flip_seed"]), stream=int(c))
        tests.append({**g.iloc[0][KEY].to_dict(), "n_attacks": len(g), "global_mean_delta_sim": r["observed"],
                      "n_extreme": r["n_extreme"], "n_sign_flips": r["n_flips"], "p_one_sided": r["p_one_sided"]})
    tests = pd.DataFrame(tests)
    bh = benjamini_hochberg(tests.p_one_sided, float(st["fdr_alpha"]))
    tests["q_bh"] = bh["q"]
    tests[f"reject_fdr_{st['fdr_alpha']}"] = bh["reject"]

    assoc = []
    ex = a
    for c, g in per.groupby("checkpoint_index", sort=True):
        s = spearman(g["mean_delta_sim"], g["mean_delta_score"])
        e = ex[ex.checkpoint_index == c]
        s2 = spearman(e["delta_sim"], e["delta_score"])
        assoc.append({**g.iloc[0][KEY].to_dict(), "n_attacks": s["n"], "rho_attack_level": s["rho"],
                      "p_attack_level": s["p"], "n_examples_secondary": s2["n"],
                      "rho_pooled_examples_secondary": s2["rho"], "p_pooled_examples_secondary": s2["p"]})
    assoc = pd.DataFrame(assoc)

    brk = []
    for factor in ("attack_token", "attack_position", "repetitions"):
        g = per.groupby([factor] + KEY)["mean_delta_sim"].agg(["mean", "size"]).reset_index()
        g = g.rename(columns={factor: "level", "mean": "mean_delta_sim_equal_attack", "size": "n_attacks"})
        g.insert(0, "factor", factor)
        g["level"] = g["level"].astype(str)
        brk.append(g)
    return per, glob, acc, tests, assoc, pd.concat(brk, ignore_index=True)


def compare(gen: pd.DataFrame, glob: pd.DataFrame):
    cmp_ = gen[KEY + ["sim_rel", "sim_nonrel", "delta_sim_real", "delta_sim_real_se_units", "delta_step_real"]].merge(
        glob[KEY + ["global_sim_attack", "global_sim_control", "global_delta_sim", "global_delta_sim_se_units",
                    "step_mean_delta_sim"]], on=KEY)
    cmp_ = cmp_.rename(columns={"global_delta_sim": "delta_sim_attack", "global_delta_sim_se_units": "delta_sim_attack_se_attacks",
                                "delta_sim_real_se_units": "delta_sim_real_se_queries", "step_mean_delta_sim": "delta_step_attack",
                                "global_sim_attack": "sim_attack", "global_sim_control": "sim_control"})
    cmp_["both_positive"] = (cmp_.delta_sim_real > 0) & (cmp_.delta_sim_attack > 0)
    cmp_["ratio_attack_to_real_secondary"] = cmp_.delta_sim_attack / cmp_.delta_sim_real
    return cmp_


def accumulation_split(steps: pd.Series, sublayer: pd.Series) -> dict:
    s = steps.fillna(0.0)
    return {"sum_attn_steps": float(s[sublayer == "attn"].sum()), "sum_mlp_steps": float(s[sublayer == "mlp"].sum()),
            "sum_positive_attn_steps": float(s[(sublayer == "attn") & (s > 0)].sum()),
            "sum_positive_mlp_steps": float(s[(sublayer == "mlp") & (s > 0)].sum())}


def main():
    args = stage_argparser("Exp 16 stage 04: analysis").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    stage = out / "04_analysis"
    ups = [(out / "01_clean", "clean_similarity.csv"), (out / "02_qrel", "qrel_similarity.csv.gz"),
           (out / "03_attack", "attack_similarity.csv.gz")]
    fp = {str(d): load_status(d).get("finished") for d, f in ups if (d / "status.json").exists()}
    if not args.force and is_already_successful(stage, ["summary.json"]) and load_status(stage).get("upstream") == fp:
        print(f"[04] already current ({stage})")
        return
    write_status(stage, {"status": "running", "started": now()})
    alpha = float(cfg["statistics"]["fdr_alpha"])

    clean = pd.read_csv(_need(*ups[0]), dtype=DT)
    qrel = pd.read_csv(_need(*ups[1]), dtype=DT)
    att = pd.read_csv(_need(*ups[2]), dtype=DT)
    _check_checkpoints(clean, "clean", "pair_id")
    _check_checkpoints(qrel, "qrel", "pair_id")
    _check_checkpoints(att, "attack", "attack_name")
    for name, df in (("clean", clean["similarity"]), ("qrel", qrel["similarity"]),
                     ("attack", att[["sim_attack", "sim_control"]].values.ravel())):
        if not np.isfinite(np.asarray(df, float)).all():
            raise RuntimeError(f"{name}: non-finite similarities")

    cc = analyze_clean(clean, alpha)
    pq, gen, mono = analyze_qrel(qrel)
    per, glob, acc, tests, assoc, brk = analyze_attacks(att, cfg)
    cmp_ = compare(gen, glob)

    stage.mkdir(parents=True, exist_ok=True)
    cc.to_csv(stage / "clean_correlation.csv", index=False)
    pq.to_csv(stage / "qrel_per_query_trajectory.csv", index=False)
    gen.to_csv(stage / "genuine_trajectory.csv", index=False)
    per.to_csv(stage / "attack_per_attack_means.csv", index=False)
    glob.to_csv(stage / "attack_global_trajectory.csv", index=False)
    acc.to_csv(stage / "attack_accumulation.csv", index=False)
    tests.to_csv(stage / "attack_signflip_fdr.csv", index=False)
    assoc.to_csv(stage / "delta_sim_vs_delta_score.csv", index=False)
    cmp_.to_csv(stage / "genuine_vs_attack.csv", index=False)
    brk.to_csv(stage / "attack_breakdowns.csv", index=False)

    rej_col = f"reject_fdr_{cfg['statistics']['fdr_alpha']}"
    first_sig = lambda df, col, thr=0.05: next((r.checkpoint_name for r in df.itertuples() if getattr(r, col) < thr), None)  # noqa: E731
    last = CHECKPOINTS[-1]
    summ = {
        "n_clean": int(clean.pair_id.nunique()),
        "n_qrel_queries": int(qrel.qid.nunique()), "n_qrel_docs": int(qrel.pair_id.nunique()),
        "n_attacks": int(att.attack_name.nunique()), "n_attack_examples": int(len(att) // 25),
        "attack_delta_score_sign_counts": {
            "positive": int((att[att.checkpoint_index == 0].delta_score > 0).sum()),
            "zero": int((att[att.checkpoint_index == 0].delta_score == 0).sum()),
            "negative": int((att[att.checkpoint_index == 0].delta_score < 0).sum())},
        "clean": {"rho_by_checkpoint": dict(zip(cc.checkpoint_name, cc.spearman_rho.round(4))),
                  "first_checkpoint_p_lt_0.05": first_sig(cc, "spearman_p"),
                  "max_rho_checkpoint": cc.loc[cc.spearman_rho.idxmax(), "checkpoint_name"],
                  "max_rho": float(cc.spearman_rho.max())},
        "genuine": {"delta_sim_real_by_checkpoint": dict(zip(gen.checkpoint_name, gen.delta_sim_real.round(5))),
                    "delta_embedding": float(gen.delta_sim_real.iloc[0]), "delta_final": float(gen.delta_sim_real.iloc[-1]),
                    "accumulation": accumulation_split(gen.delta_step_real, gen.sublayer),
                    "mean_monot5_score_by_group": mono.mean().to_dict()},
        "attack": {"delta_sim_by_checkpoint": dict(zip(glob.checkpoint_name, glob.global_delta_sim.round(5))),
                   "delta_embedding": float(glob.global_delta_sim.iloc[0]),
                   "delta_final": float(glob.global_delta_sim.iloc[-1]),
                   "mean_delta_score_equal_attack": float(per[per.checkpoint_index == 0].mean_delta_score.mean()),
                   "accumulation": accumulation_split(acc.delta_step, acc.sublayer),
                   "fdr_rejected_checkpoints": tests.loc[tests[rej_col], "checkpoint_name"].tolist(),
                   "fdr_not_rejected_checkpoints": tests.loc[~tests[rej_col], "checkpoint_name"].tolist()},
        "delta_sim_vs_delta_score": {"rho_attack_level_by_checkpoint": dict(zip(assoc.checkpoint_name, assoc.rho_attack_level.round(4))),
                                     "rho_attack_level_final": float(assoc.rho_attack_level.iloc[-1]),
                                     "max_rho_checkpoint": assoc.loc[assoc.rho_attack_level.idxmax(), "checkpoint_name"],
                                     "max_rho": float(assoc.rho_attack_level.max())},
        "comparison": {
            "n_checkpoints_both_positive": int(cmp_.both_positive.sum()),
            "trajectory_pearson_delta_sim_descriptive": float(np.corrcoef(cmp_.delta_sim_real, cmp_.delta_sim_attack)[0, 1]),
            "trajectory_pearson_delta_step_descriptive": float(np.corrcoef(cmp_.delta_step_real.iloc[1:], cmp_.delta_step_attack.iloc[1:])[0, 1]),
            "final_ratio_attack_to_real_secondary": float(cmp_.ratio_attack_to_real_secondary.iloc[-1]),
            "final_checkpoint": last},
    }
    (stage / "summary.json").write_text(json.dumps(summ, indent=2, default=float))
    write_status(stage, {"status": "success", "finished": now(), "upstream": fp})
    print(json.dumps({k: summ[k] for k in ("n_clean", "n_qrel_queries", "n_attacks", "n_attack_examples",
                                           "attack_delta_score_sign_counts")}, indent=1))
    print(f"[04] analysis -> {stage}")


if __name__ == "__main__":
    main()
