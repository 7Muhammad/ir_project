"""
scripts/05_plot.py
===================
Stage 05 — figures from 04_analysis/ into plots/ (PNG, 200 dpi).

  fig1_clean_rho.png                 Spearman(sim_clean, score_clean) per checkpoint
  fig2_genuine_trajectory.png        equal-query mean sim, qrel 2/3 vs qrel 0
  fig3_genuine_gap.png               delta_sim_real
  fig4_attack_control_trajectory.png equal-attack mean sim, attacked vs padded control
  fig5_attack_gap.png                delta_sim_attack, BH-FDR rejections filled
  fig6_attack_accumulation.png       delta_step_attack, attention vs MLP transitions
  fig7_genuine_vs_attack_gap.png     delta_sim_real and delta_sim_attack on one axis
  fig8_genuine_vs_attack_steps.png   delta_step_real / delta_step_attack (two panels, shared y)
  fig9_delta_sim_vs_delta_score.png  Spearman across attacks (primary) + pooled examples (secondary)
  figS1_breakdown_token.png          descriptive: delta_sim by attack token (heatmap, diverging)
  figS2_breakdown_reps_position.png  descriptive: delta_sim by repetitions / insertion position

Bands are +-1 SE across the top-level unit (queries or attacks); they are
descriptive, not the test. Palette: validated reference categorical slots 1-3.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib.checkpoints import CHECKPOINTS  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.run_utils import is_already_successful, load_status, now, stage_argparser, write_status  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
X = np.arange(len(CHECKPOINTS))
LABELS = ["emb"] + [f"L{int(n[1:3])} {'attn' if n.endswith('attn') else 'mlp'}" for n in CHECKPOINTS[1:]]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
})


def _axis(ax, ylabel, zero=True):
    ax.set_xticks(X)
    ax.set_xticklabels(LABELS, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(X) - 0.4)
    ax.set_ylabel(ylabel)
    if zero:
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)


def _line(ax, y, color, label, se=None, marker="o", filled=None, ls="-"):
    if se is not None:
        ax.fill_between(X, y - se, y + se, color=color, alpha=0.15, lw=0)
    ax.plot(X, y, color=color, ls=ls, lw=2, label=label, zorder=3)
    filled = np.ones(len(X), bool) if filled is None else np.asarray(filled, bool)
    ax.scatter(X[filled], y[filled], s=22, color=color, edgecolor="white", linewidth=1, zorder=4, marker=marker)
    if (~filled).any():
        ax.scatter(X[~filled], y[~filled], s=22, facecolor="white", edgecolor=color, linewidth=1.2, zorder=4, marker=marker)


def _label_end(ax, y, text, color_ink=INK):
    ax.annotate(text, (X[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=color_ink)


def _save(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _step_bars(ax, steps, sub, title):
    s = np.asarray(steps, float)
    for kind, color, hatch, lab in (("attn", BLUE, None, "attention transition"), ("mlp", AQUA, "////", "MLP transition")):
        m = (np.asarray(sub) == kind)
        ax.bar(X[m], np.nan_to_num(s[m]), width=0.72, color=color, hatch=hatch, edgecolor="white", linewidth=0.5, label=lab)
    _axis(ax, "Δ step (cosine)")
    ax.set_title(title, loc="left", fontsize=9, color=INK)


def main():
    args = stage_argparser("Exp 16 stage 05: plots").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    an = out / "04_analysis"
    if not is_already_successful(an, ["summary.json"]):
        raise FileNotFoundError(f"run stage 04 first ({an})")
    pdir = out / "plots"
    if not args.force and is_already_successful(pdir, ["fig9_delta_sim_vs_delta_score.png"]) and \
            load_status(pdir).get("analysis_finished") == load_status(an).get("finished"):
        print(f"[05] plots already current ({pdir})")
        return
    pdir.mkdir(parents=True, exist_ok=True)
    cc = pd.read_csv(an / "clean_correlation.csv")
    gen = pd.read_csv(an / "genuine_trajectory.csv")
    glob = pd.read_csv(an / "attack_global_trajectory.csv")
    acc = pd.read_csv(an / "attack_accumulation.csv")
    tests = pd.read_csv(an / "attack_signflip_fdr.csv")
    assoc = pd.read_csv(an / "delta_sim_vs_delta_score.csv")
    cmp_ = pd.read_csv(an / "genuine_vs_attack.csv")
    brk = pd.read_csv(an / "attack_breakdowns.csv", dtype={"level": str})
    nq, na, nc = int(gen.n_units.iloc[0]), int(glob.n_units.iloc[0]), int(cc.n.iloc[0])
    rej = tests[[c for c in tests.columns if c.startswith("reject_fdr")][0]].values.astype(bool)

    # 1
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _line(ax, cc.spearman_rho.values, BLUE, "Spearman ρ", filled=cc.spearman_p.values < 0.05)
    _axis(ax, "Spearman ρ(sim, monoT5 score)")
    ax.set_title(f"Clean Type-A pairs (n = {nc}): query–document similarity vs monoT5 score\n"
                 "filled = nominal p < 0.05, hollow = not", loc="left", fontsize=9)
    _save(fig, pdir / "fig1_clean_rho.png")

    # 2
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _line(ax, gen.sim_rel.values, BLUE, "qrel 2/3 (relevant)", se=gen.sim_rel_se_units.values)
    _line(ax, gen.sim_nonrel.values, ORANGE, "qrel 0 (non-relevant)", se=gen.sim_nonrel_se_units.values, marker="s")
    _axis(ax, "cos(Q̄, D̄)", zero=False)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Genuine relevance: equal-query mean similarity ({nq} queries, ±1 SE across queries)", loc="left", fontsize=9)
    _save(fig, pdir / "fig2_genuine_trajectory.png")

    # 3
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _line(ax, gen.delta_sim_real.values, BLUE, "Δsim real", se=gen.delta_sim_real_se_units.values)
    _axis(ax, "Δsim real = rel − non-rel")
    ax.set_title(f"Genuine-relevance gap (within query, {nq} queries, ±1 SE)", loc="left", fontsize=9)
    _save(fig, pdir / "fig3_genuine_gap.png")

    # 4
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _line(ax, glob.global_sim_attack.values, ORANGE, "attacked", se=glob.global_sim_attack_se_units.values)
    _line(ax, glob.global_sim_control.values, INK2, "padded control", se=glob.global_sim_control_se_units.values, marker="s")
    _axis(ax, "cos(Q̄, D̄)", zero=False)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Attacked vs padded control: equal-attack mean similarity ({na} attacks, ±1 SE across attacks)",
                 loc="left", fontsize=9)
    _save(fig, pdir / "fig4_attack_control_trajectory.png")

    # 5
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _line(ax, glob.global_delta_sim.values, ORANGE, "Δsim attack", se=glob.global_delta_sim_se_units.values, filled=rej)
    _axis(ax, "Δsim attack = attack − control")
    ax.set_title(f"Attack-induced similarity gap ({na} attacks, ±1 SE); filled = one-sided sign-flip "
                 f"BH-FDR q ≤ {cfg['statistics']['fdr_alpha']}", loc="left", fontsize=9)
    _save(fig, pdir / "fig5_attack_gap.png")

    # 6
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _step_bars(ax, acc.delta_step.values, acc.sublayer.values,
               "Attack gap accumulation: Δstep(c) = Δsim(c) − Δsim(c−1), equal attack weight")
    ax.legend(loc="best", fontsize=8)
    _save(fig, pdir / "fig6_attack_accumulation.png")

    # 7
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _line(ax, cmp_.delta_sim_real.values, BLUE, f"genuine: qrel 2/3 − qrel 0 ({nq} queries)",
          se=cmp_.delta_sim_real_se_queries.values)
    _line(ax, cmp_.delta_sim_attack.values, ORANGE, f"adversarial: attacked − padded control ({na} attacks)",
          se=cmp_.delta_sim_attack_se_attacks.values, marker="s")
    _axis(ax, "Δ cos(Q̄, D̄)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Genuine vs adversarial similarity gap (different populations and units; ±1 SE)", loc="left", fontsize=9)
    _save(fig, pdir / "fig7_genuine_vs_attack_gap.png")

    # 8
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, sharey=True)
    _step_bars(axes[0], cmp_.delta_step_real.values, cmp_.sublayer.values, "Genuine relevance: Δstep real")
    _step_bars(axes[1], cmp_.delta_step_attack.values, cmp_.sublayer.values, "Adversarial: Δstep attack")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].tick_params(labelbottom=False)
    _save(fig, pdir / "fig8_genuine_vs_attack_steps.png")

    # 9
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _line(ax, assoc.rho_attack_level.values, ORANGE, f"attack level (primary, n = {na})",
          filled=assoc.p_attack_level.values < 0.05)
    _line(ax, assoc.rho_pooled_examples_secondary.values, INK2, "pooled examples (secondary)", marker="s", ls="--")
    _axis(ax, "Spearman ρ(Δsim, Δscore)")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Does a larger similarity increase go with a larger score increase? (filled = nominal p < 0.05)",
                 loc="left", fontsize=9)
    _save(fig, pdir / "fig9_delta_sim_vs_delta_score.png")

    # S1 — token heatmap (diverging, symmetric around 0)
    tok = brk[brk.factor == "attack_token"].pivot(index="level", columns="checkpoint_index", values="mean_delta_sim_equal_attack")
    v = float(np.nanmax(np.abs(tok.values))) or 1.0
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(tok) + 1.4))
    im = ax.imshow(tok.values, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_yticks(range(len(tok)))
    ax.set_yticklabels(tok.index)
    ax.set_xticks(X)
    ax.set_xticklabels(LABELS, rotation=90, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mean Δsim (equal attack weight)", fraction=0.03)
    ax.set_title("Descriptive: attack-induced gap by attack token", loc="left", fontsize=9)
    _save(fig, pdir / "figS1_breakdown_token.png")

    # S2 — repetitions (sequential blue ramp) and position (categorical)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    reps = brk[brk.factor == "repetitions"]
    levels = sorted(reps.level.unique(), key=lambda s: int(s))
    ramp = ["#a9c9f0", "#78a9e6", "#4a8ad9", "#2a6fc2", "#1b4f8f"]
    for i, lv in enumerate(levels):
        y = reps[reps.level == lv].sort_values("checkpoint_index").mean_delta_sim_equal_attack.values
        axes[0].plot(X, y, color=ramp[i % len(ramp)], lw=2, label=f"{lv} rep")
    _axis(axes[0], "mean Δsim")
    axes[0].legend(fontsize=7)
    axes[0].set_title("by repetition count", loc="left", fontsize=9)
    pos = brk[brk.factor == "attack_position"]
    for lv, color, mk in zip(["start", "end", "random"], [BLUE, ORANGE, AQUA], ["o", "s", "^"]):
        y = pos[pos.level == lv].sort_values("checkpoint_index").mean_delta_sim_equal_attack.values
        if len(y):
            _line(axes[1], y, color, lv, marker=mk)
    _axis(axes[1], "")
    axes[1].legend(fontsize=7)
    axes[1].set_title("by insertion position", loc="left", fontsize=9)
    _save(fig, pdir / "figS2_breakdown_reps_position.png")

    write_status(pdir, {"status": "success", "finished": now(), "analysis_finished": load_status(an).get("finished")})
    print(f"[05] plots -> {pdir}")


if __name__ == "__main__":
    main()
