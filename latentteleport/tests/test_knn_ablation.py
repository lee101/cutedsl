from __future__ import annotations

import torch

from scripts.spec_knn_ablation import (
    _apply_plan,
    _fit_two_input_gate,
    _motion_descriptor,
    _neighbor_plan,
    _normalize,
)


def test_normalize_returns_unit_rows() -> None:
    values = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    normalized = _normalize(values)
    torch.testing.assert_close(normalized.norm(dim=1), torch.ones(2))


def test_motion_descriptor_is_compact_and_normalized() -> None:
    delta = torch.randn(3, 16, 64, 64)
    descriptor = _motion_descriptor(delta)
    assert descriptor.shape == (3, 16 * 8 * 8)
    torch.testing.assert_close(descriptor.norm(dim=1), torch.ones(3))


def test_neighbor_plan_excludes_query_itself() -> None:
    embeddings = _normalize(torch.eye(4))
    motion = embeddings.clone()
    indices, weights = _neighbor_plan(
        embeddings,
        motion,
        embeddings,
        motion,
        top_k=1,
        pool_k=3,
        temperature=0.1,
        motion_weight=0.5,
        exclude_diagonal=True,
    )
    assert not torch.any(indices[:, 0] == torch.arange(4))
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_motion_pruning_selects_matching_candidate_from_text_pool() -> None:
    query_embedding = _normalize(torch.tensor([[1.0, 0.0]]))
    bank_embeddings = _normalize(torch.tensor([[1.0, 0.01], [1.0, -0.01]]))
    query_motion = _normalize(torch.tensor([[0.0, 1.0]]))
    bank_motion = _normalize(torch.tensor([[0.0, -1.0], [0.0, 1.0]]))
    indices, _ = _neighbor_plan(
        query_embedding,
        query_motion,
        bank_embeddings,
        bank_motion,
        top_k=1,
        pool_k=2,
        temperature=0.1,
        motion_weight=0.75,
    )
    assert indices.item() == 1


def test_apply_plan_computes_weighted_neighbor_move() -> None:
    bank = torch.tensor([[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]])
    indices = torch.tensor([[0, 2]])
    weights = torch.tensor([[0.25, 0.75]])
    output = _apply_plan(bank, indices, weights)
    torch.testing.assert_close(output, torch.tensor([[7.0, 8.0]]))


def test_two_input_gate_recovers_exact_linear_transport() -> None:
    local = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    neighbor = torch.tensor([[2.0, -1.0], [0.5, 4.0]])
    target = 0.8 * local + 0.3 * neighbor
    local_weight, neighbor_weight = _fit_two_input_gate(local, neighbor, target)
    assert abs(local_weight - 0.8) < 1e-6
    assert abs(neighbor_weight - 0.3) < 1e-6
