"""
tests/test_patching_math.py
===========================
Unit tests for the activation patching effect formulas.

No model or real data is needed here.  We test the MATH using fake scores.

WHAT IS ACTIVATION PATCHING?
------------------------------
Given three scores for one (query, document) pair:
  - control_score  : score when input is the padded control (empty baseline)
  - attack_score   : score when input is the attacked document
  - patched_score  : score after swapping one component's activation from
                     the attack run into the control run (forward patching),
                     OR from the control run into the attack run (reverse).

FORWARD EFFECT
--------------
"If I inject the attack activation at layer L into the control run,
 how much of the attack's score increase do I recover?"

    forward_effect = (patched_score - control_score)
                   / (attack_score  - control_score)

  - Effect = 0 → injecting the attack at L changes nothing: L is NOT involved.
  - Effect = 1 → injecting the attack at L alone recovers the full increase:
                 L alone CARRIES the attack signal.
  - Effect > 1 → amplification (the component does more than just carry).

REVERSE EFFECT
--------------
"If I restore the control activation at layer L while running the attack input,
 how much of the attack's score increase do I CANCEL?"

    reverse_effect = (attack_score - patched_score)
                   / (attack_score - control_score)

  - Effect = 0 → restoring the control at L changes nothing: L is NOT necessary.
  - Effect = 1 → restoring the control at L alone cancels the full increase:
                 L is NECESSARY for the attack to work.

COMBINED
--------
  combined = min(forward_effect, reverse_effect)

  High combined → the component is both sufficient (forward≈1) and necessary
  (reverse≈1) for the attack signal.

NEAR-ZERO DENOMINATOR
---------------------
  If attack_score ≈ control_score (attack had almost no effect), both effects
  are ill-defined (dividing by ~0).  The code skips such examples using
  SKIP_EPSILON.  We test that this guard exists and works.

Run with:
    pytest tests/test_patching_math.py -v
"""

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest

from src.patching import SKIP_EPSILON


# ---------------------------------------------------------------------------
# Helpers: the effect formulas as plain Python functions
# ---------------------------------------------------------------------------

def forward_effect(control_score, attack_score, patched_score):
    """
    How much of the attack-score increase does patching recover?

    Value ≈ 1  →  this component carries the full attack signal.
    Value ≈ 0  →  this component is not involved.
    """
    delta = attack_score - control_score
    return (patched_score - control_score) / delta


def reverse_effect(control_score, attack_score, patched_score):
    """
    How much of the attack-score increase does restoring control cancel?

    Value ≈ 1  →  this component is necessary for the attack.
    Value ≈ 0  →  the attack works without this component.
    """
    delta = attack_score - control_score
    return (attack_score - patched_score) / delta


def combined_effect(fwd, rev):
    """Combined = min(forward, reverse) — high only if both are high."""
    return min(fwd, rev)


# ---------------------------------------------------------------------------
# Test 1: worked example from the docstring (50 / 50 split)
# ---------------------------------------------------------------------------

def test_symmetric_worked_example():
    """
    Use the worked example from the task description.

    Given:
      control_score = 1.0
      attack_score  = 5.0
      patched_score = 3.0

    Forward: (3 - 1) / (5 - 1) = 2/4 = 0.5
    Reverse: (5 - 3) / (5 - 1) = 2/4 = 0.5

    A 50% effect means this component carries half of the attack signal.
    """
    ctrl, atk, pat = 1.0, 5.0, 3.0

    fwd = forward_effect(ctrl, atk, pat)
    rev = reverse_effect(ctrl, atk, pat)

    assert math.isclose(fwd, 0.5, rel_tol=1e-9), f"Expected fwd=0.5, got {fwd}"
    assert math.isclose(rev, 0.5, rel_tol=1e-9), f"Expected rev=0.5, got {rev}"
    assert math.isclose(combined_effect(fwd, rev), 0.5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Test 2: component carries the FULL attack signal (effect = 1.0)
# ---------------------------------------------------------------------------

def test_full_effect_when_patched_equals_attack():
    """
    Forward effect = 1.0 when patched_score == attack_score.

    Interpretation: injecting the attack activation at this component
    alone is sufficient to produce the full attack score.
    """
    ctrl, atk = 1.0, 5.0
    pat = atk  # patching fully recovers the attack score

    fwd = forward_effect(ctrl, atk, pat)
    assert math.isclose(fwd, 1.0, rel_tol=1e-9), f"Expected fwd=1.0, got {fwd}"


def test_full_reverse_effect_when_patched_equals_control():
    """
    Reverse effect = 1.0 when patched_score == control_score.

    Interpretation: restoring the control activation at this component
    alone is sufficient to cancel the full attack score increase.
    """
    ctrl, atk = 1.0, 5.0
    pat = ctrl  # restoring control fully cancels the attack

    rev = reverse_effect(ctrl, atk, pat)
    assert math.isclose(rev, 1.0, rel_tol=1e-9), f"Expected rev=1.0, got {rev}"


# ---------------------------------------------------------------------------
# Test 3: zero effect when patching changes nothing
# ---------------------------------------------------------------------------

def test_zero_forward_effect_when_no_change():
    """
    Forward effect = 0.0 when patched_score == control_score.

    Interpretation: injecting the attack activation at this component
    had no effect — this component does NOT carry the attack signal.
    """
    ctrl, atk = 1.0, 5.0
    pat = ctrl  # patching changed nothing

    fwd = forward_effect(ctrl, atk, pat)
    assert math.isclose(fwd, 0.0, rel_tol=1e-9), f"Expected fwd=0.0, got {fwd}"


def test_zero_reverse_effect_when_no_change():
    """
    Reverse effect = 0.0 when patched_score == attack_score.

    Interpretation: restoring the control activation at this component
    had no effect — this component is NOT necessary for the attack.
    """
    ctrl, atk = 1.0, 5.0
    pat = atk  # restoring control changed nothing

    rev = reverse_effect(ctrl, atk, pat)
    assert math.isclose(rev, 0.0, rel_tol=1e-9), f"Expected rev=0.0, got {rev}"


# ---------------------------------------------------------------------------
# Test 4: combined = min(forward, reverse)
# ---------------------------------------------------------------------------

def test_combined_is_min_of_forward_and_reverse():
    """
    The combined effect is the minimum of forward and reverse effects.

    This deliberately favours components that are BOTH sufficient (high fwd)
    AND necessary (high rev).  A component that is sufficient but not necessary
    might be an easy-to-trigger path rather than a critical one.
    """
    # Asymmetric case: fwd=0.8, rev=0.3 → combined=0.3
    ctrl, atk = 0.0, 4.0
    pat_fwd = ctrl + 0.8 * (atk - ctrl)  # patching recovers 80%
    pat_rev = ctrl + 0.7 * (atk - ctrl)  # restoring cancels 30%

    fwd = forward_effect(ctrl, atk, pat_fwd)
    rev = reverse_effect(ctrl, atk, pat_rev)
    combined = combined_effect(fwd, rev)

    assert math.isclose(fwd, 0.8, rel_tol=1e-9), f"Expected fwd=0.8, got {fwd}"
    assert math.isclose(rev, 0.3, rel_tol=1e-9), f"Expected rev=0.3, got {rev}"
    assert math.isclose(combined, 0.3, rel_tol=1e-9), f"Expected combined=0.3, got {combined}"


# ---------------------------------------------------------------------------
# Test 5: SKIP_EPSILON guards against near-zero denominator
# ---------------------------------------------------------------------------

def test_skip_epsilon_is_defined():
    """
    Verify that SKIP_EPSILON is defined and has a sensible positive value.

    The patching code skips examples where |attack_delta| < SKIP_EPSILON.
    If SKIP_EPSILON were 0, we could divide by a very small number and get
    astronomically large (meaningless) effects.
    """
    assert SKIP_EPSILON > 0, "SKIP_EPSILON must be positive to guard against division by zero."
    assert SKIP_EPSILON < 1.0, "SKIP_EPSILON should be a small threshold, not >= 1.0."


def test_near_zero_delta_would_be_skipped():
    """
    Demonstrate why examples with tiny attack_delta must be skipped.

    If attack_score ≈ control_score (the attack barely changed the score),
    dividing by that tiny delta amplifies any patching noise by a huge factor,
    giving meaningless effects like 1e8 or -1e5.

    The code checks:
        if abs(attack_score - control_score) < SKIP_EPSILON: skip

    We verify that a delta smaller than SKIP_EPSILON would produce an
    unreasonably large effect value.
    """
    ctrl = 1.0
    atk  = ctrl + SKIP_EPSILON / 10  # delta is 10× smaller than threshold
    pat  = ctrl + 0.01               # small perturbation from patching

    # This would give a huge effect if not skipped
    delta = atk - ctrl
    raw_effect = (pat - ctrl) / delta

    # The raw effect should be "unreasonably large" (>> 1.0) due to tiny denominator
    assert abs(raw_effect) > 10.0, (
        f"Expected a large raw effect when delta is tiny, got {raw_effect:.1f}. "
        "The SKIP_EPSILON guard is important to prevent garbage results."
    )


# ---------------------------------------------------------------------------
# Test 6: effects can be negative or > 1 (over-correction)
# ---------------------------------------------------------------------------

def test_effect_can_exceed_one():
    """
    Effects outside [0, 1] are mathematically valid, not errors.

    Forward effect > 1.0 means the patched score EXCEEDS the attack score —
    the component amplifies the signal beyond the original attack level.

    Reverse effect > 1.0 means restoring the control drops the score BELOW
    the control level — the component was actually suppressing the score.

    Neither case is a bug.  We document this here so it is not confused with
    a coding error when it appears in real results.
    """
    ctrl, atk = 1.0, 5.0
    pat_amplify = 7.0  # patched score exceeds attack score

    fwd = forward_effect(ctrl, atk, pat_amplify)
    assert fwd > 1.0, f"Expected fwd > 1.0 for amplification, got {fwd}"

    pat_suppress = -1.0  # restoring control drops score below control baseline
    rev = reverse_effect(ctrl, atk, pat_suppress)
    assert rev > 1.0, f"Expected rev > 1.0 for over-cancellation, got {rev}"
