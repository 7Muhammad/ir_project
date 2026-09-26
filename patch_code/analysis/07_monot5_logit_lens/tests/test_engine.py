"""End-to-end row-shape/mechanics tests for exp7lib/engine.py, on the tiny model."""

from __future__ import annotations

import json

from conftest import DEVICE

from exp7lib.engine import run_example
from exp7lib.run_utils import ExampleInputs


class _FakeTokenizer:
    """Deterministic id<->token mapping, avoids needing real SentencePiece vocab for shape tests."""
    def convert_ids_to_tokens(self, idx):
        return f"▁tok{idx}"


def test_run_example_row_shape_no_flagged_heads(tiny_model, control_enc, attack_enc):
    tokenizer = _FakeTokenizer()
    inputs = ExampleInputs(
        status="ok", clean_enc=control_enc, control_enc=control_enc, attack_enc=attack_enc,
        attack_token_id=3,
    )
    rows = run_example(
        tiny_model, tokenizer, inputs,
        query="q", passage="d", attack_token_text="tok3",
        flagged_heads=[], k_values=[5, 10, 20], composition_k=10, device=DEVICE,
    )
    n_layers = tiny_model.config.num_decoder_layers
    assert len(rows) == 3 * n_layers  # 3 run_types x whole-block layers, no per-head rows
    for row in rows:
        assert row["head"] is None
        assert row["run_type"] in ("clean", "control", "attack")
        for k in (5, 10, 20):
            toks = json.loads(row[f"top_k{k}_tokens"])
            vals = json.loads(row[f"top_k{k}_logits"])
            assert len(toks) == k == len(vals)
        comp = json.loads(row["bucket_composition"])
        assert abs(sum(comp.values()) - 100.0) < 1e-6
        assert isinstance(row["attack_token_rank"], int) and row["attack_token_rank"] >= 1


def test_run_example_includes_flagged_head_rows(tiny_model, control_enc, attack_enc):
    tokenizer = _FakeTokenizer()
    inputs = ExampleInputs(
        status="ok", clean_enc=control_enc, control_enc=control_enc, attack_enc=attack_enc,
        attack_token_id=3,
    )
    flagged = [(0, 0), (1, 2)]
    rows = run_example(
        tiny_model, tokenizer, inputs,
        query="q", passage="d", attack_token_text="tok3",
        flagged_heads=flagged, k_values=[5], composition_k=5, device=DEVICE,
    )
    n_layers = tiny_model.config.num_decoder_layers
    expected = 3 * (n_layers + len(flagged))
    assert len(rows) == expected

    head_rows = [r for r in rows if r["head"] is not None]
    assert len(head_rows) == 3 * len(flagged)
    seen_pairs = {(r["layer"], r["head"]) for r in head_rows}
    assert seen_pairs == set(flagged)


def test_attack_token_rank_1_when_contribution_favors_it(tiny_model, control_enc):
    """Sanity: if we hand-craft a contribution that strongly favors the tracked
    token, its rank must come out as 1 — exercises the full projection path."""
    import torch
    from exp7lib.lens import project_to_logits, token_rank_and_logit

    contribution = torch.zeros(1, 1, tiny_model.config.d_model)
    # Push the unembedding output for token 7 up by biasing the input toward
    # the corresponding lm_head row (works because lm_head is linear).
    w_row = tiny_model.lm_head.weight[7]
    contribution[0, 0] = w_row * 1000.0
    logits = project_to_logits(tiny_model, contribution)
    rank, _ = token_rank_and_logit(logits, token_id=7)
    assert rank == 1
