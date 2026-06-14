"""
src/plotting.py
===============
Plotting utilities for the monoT5 activation-patching experiment.

Plots produced
--------------
1. attack_delta_hist.png
    Histogram of (attack_score - control_score) over selected examples.
    Shows how large the adversarial effect is and whether the selected
    population is concentrated or spread.

2. forward_layer_component_heatmap.png
    Rows = layers, Columns = components.
    Cell value = mean forward patching effect.
    Interpretation: a bright cell means that INJECTING the attack activation
    at (layer, component) into the control run recovers most of the attack's
    score increase → this component is SUFFICIENT to carry the attack signal.

3. reverse_layer_component_heatmap.png
    Same layout, but reverse patching effects.
    Interpretation: a bright cell means that REMOVING the attack activation
    at (layer, component) from the attack run cancels most of the score
    increase → this component is NECESSARY for the attack.

4. combined_layer_component_heatmap.png
    Cell value = min(forward_effect, reverse_effect).
    Highlights components that are BOTH sufficient and necessary.
    These are the most mechanistically important layers/components.

How to interpret the heatmaps
-------------------------------
- X axis: component type (enc_self_attn, enc_mlp, dec_self_attn, dec_cross_attn, dec_mlp)
- Y axis: layer index (0 = earliest, 11 = latest for T5-base)
- Colour: patching effect (normalised, 0–1 range typical, >1 possible)
  * Near 0  → component irrelevant to attack
  * Near 1  → component fully explains the attack
  * >1      → over-recovery / amplification
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Human-readable names for x-axis labels
COMPONENT_LABELS = {
    "encoder_self_attn": "Enc Self-Attn",
    "encoder_mlp":       "Enc MLP",
    "decoder_self_attn": "Dec Self-Attn",
    "decoder_cross_attn":"Dec Cross-Attn",
    "decoder_mlp":       "Dec MLP",
}

# Display order for components on x-axis
COMPONENT_ORDER = [
    "encoder_self_attn",
    "encoder_mlp",
    "decoder_self_attn",
    "decoder_cross_attn",
    "decoder_mlp",
]


# ---------------------------------------------------------------------------
# Plot 1: Attack delta histogram
# ---------------------------------------------------------------------------

def plot_attack_delta_histogram(
    scored_pairs: List[Dict],
    output_path: pathlib.Path,
    min_attack_delta: float = 0.0,
) -> None:
    """
    Plot a histogram of attack_score - control_score for selected examples.

    The dashed vertical line marks the selection threshold (min_attack_delta).

    Parameters
    ----------
    scored_pairs : list of dicts with "attack_delta_vs_control"
    output_path : where to save the PNG
    min_attack_delta : threshold used for selection (drawn as vertical line)
    """
    deltas = [p["attack_delta_vs_control"] for p in scored_pairs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(deltas, bins=30, color="#4878CF", edgecolor="white", linewidth=0.5)
    ax.axvline(
        min_attack_delta, color="red", linestyle="--", linewidth=1.5,
        label=f"selection threshold ({min_attack_delta})"
    )
    ax.set_xlabel("attack_score − control_score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Distribution of adversarial score increase\n"
        "(attack_score − padded_control_score, selected examples)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Shared heatmap builder
# ---------------------------------------------------------------------------

def _build_heatmap_matrix(
    agg_df: pd.DataFrame,
    direction: str,
) -> tuple[np.ndarray, List[int], List[str]]:
    """
    Build a 2-D numpy matrix (layers × components) from the aggregated results.

    Parameters
    ----------
    agg_df : DataFrame with columns: layer, component, direction, mean_effect
    direction : "forward" or "reverse"

    Returns
    -------
    (matrix, sorted_layers, ordered_components)
    """
    df = agg_df[agg_df["direction"] == direction].copy()

    layers = sorted(df["layer"].unique())
    components = [c for c in COMPONENT_ORDER if c in df["component"].unique()]

    matrix = np.full((len(layers), len(components)), np.nan)
    for i, layer in enumerate(layers):
        for j, comp in enumerate(components):
            row = df[(df["layer"] == layer) & (df["component"] == comp)]
            if not row.empty:
                matrix[i, j] = row["mean_effect"].values[0]

    return matrix, layers, components


def _plot_heatmap(
    matrix: np.ndarray,
    layers: List[int],
    components: List[str],
    title: str,
    output_path: pathlib.Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Render and save a single heatmap.

    Parameters
    ----------
    matrix : 2-D float array, shape (n_layers, n_components)
    layers : list of layer indices (y-axis labels)
    components : list of component names (x-axis labels)
    title : plot title
    output_path : PNG save path
    vmin, vmax : colour scale limits (auto-computed if None)
    """
    x_labels = [COMPONENT_LABELS.get(c, c) for c in components]
    y_labels = [f"Layer {l}" for l in layers]

    fig, ax = plt.subplots(figsize=(max(6, len(components) * 1.4), max(5, len(layers) * 0.4)))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean patching effect", fontsize=10)

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_title(title, fontsize=12, pad=10)

    # Annotate each cell with its value
    for i in range(len(layers)):
        for j in range(len(components)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="black" if 0.3 < val < 0.7 else "white",
                )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plotting] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2 & 3: Forward and reverse patching heatmaps
# ---------------------------------------------------------------------------

def plot_patching_heatmap(
    agg_df: pd.DataFrame,
    direction: str,
    output_path: pathlib.Path,
) -> None:
    """
    Plot either the forward or reverse patching heatmap.

    Parameters
    ----------
    agg_df : DataFrame with aggregated patching results
    direction : "forward" or "reverse"
    output_path : PNG save path
    """
    matrix, layers, components = _build_heatmap_matrix(agg_df, direction)

    if direction == "forward":
        title = (
            "Forward patching effect\n"
            "(inject attack activation into control run)\n"
            "≈1 → this component is sufficient to raise the score"
        )
    else:
        title = (
            "Reverse patching effect\n"
            "(inject control activation into attack run)\n"
            "≈1 → this component is necessary for the score increase"
        )

    _plot_heatmap(matrix, layers, components, title, output_path, vmin=0.0, vmax=1.2)


# ---------------------------------------------------------------------------
# Plot 4: Combined importance heatmap
# ---------------------------------------------------------------------------

def plot_combined_heatmap(
    agg_df: pd.DataFrame,
    output_path: pathlib.Path,
) -> None:
    """
    Plot the combined importance heatmap:
        combined = min(mean_forward_effect, mean_reverse_effect)

    Highlights components that are BOTH sufficient AND necessary.

    Parameters
    ----------
    agg_df : DataFrame with aggregated patching results
    output_path : PNG save path
    """
    fwd_matrix, layers, components = _build_heatmap_matrix(agg_df, "forward")
    rev_matrix, _, _               = _build_heatmap_matrix(agg_df, "reverse")

    # Element-wise minimum — handles NaN gracefully
    combined = np.fmin(fwd_matrix, rev_matrix)  # fmin ignores NaN

    title = (
        "Combined importance = min(forward, reverse)\n"
        "High value → component is both sufficient AND necessary for the attack\n"
        "(i.e. attack signal passes through this component)"
    )

    _plot_heatmap(combined, layers, components, title, output_path, vmin=0.0, vmax=1.2)
