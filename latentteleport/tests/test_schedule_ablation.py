import json

import pytest
import torch

from latentteleport.speculative import build_active_lane_plan, scaled_momentum
from scripts.spec_e2e import load_momentum_scales
from scripts.spec_schedule_ablation import (
    _cell_scores,
    _compositions,
    _schedule_cells,
    _uniform_schedule,
)


def test_schedule_compositions_cover_span_with_bounded_horizons():
    schedules = _compositions(total=6, parts=3, max_horizon=4)

    assert (2, 2, 2) in schedules
    assert all(len(schedule) == 3 for schedule in schedules)
    assert all(sum(schedule) == 6 for schedule in schedules)
    assert all(max(schedule) <= 4 for schedule in schedules)


def test_uniform_schedule_and_cells_are_deterministic():
    schedule = _uniform_schedule(total=8, parts=3)

    assert schedule == (3, 3, 2)
    assert _schedule_cells(2, schedule) == [(2, 3), (5, 3), (8, 2)]


def test_scaled_momentum_is_exact_on_linear_trajectory():
    base = torch.arange(12, dtype=torch.float32).view(1, 12, 1, 1, 1)
    offsets = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1, 1)
    trajectories = base + offsets

    scores = _cell_scores(
        train=trajectories[:4],
        test=trajectories[4:],
        step=3,
        horizon=4,
    )

    assert abs(float(scores["alpha"]) - 1.0) < 1e-6
    assert float(scores["test"]["taylor1"]) < 1e-6
    assert float(scores["test"]["scaled_momentum"]) < 1e-6


def test_deployment_scale_loader_is_schedule_scoped(tmp_path):
    path = tmp_path / "coefficients.json"
    path.write_text(
        json.dumps(
            {
                "protocol": {"n_steps": 16},
                "deployment_coefficients": [
                    {"step": 3, "draft_k": 4, "momentum_scale": 1.25}
                ],
            }
        )
    )

    assert load_momentum_scales(path, 16) == {(3, 4): 1.25}
    with pytest.raises(ValueError, match="fit for 16 steps"):
        load_momentum_scales(path, 30)


def test_scaled_momentum_applies_fitted_step_length():
    previous = torch.tensor([1.0])
    current = torch.tensor([3.0])

    assert torch.equal(
        scaled_momentum(current, previous, k=4, scale=1.25),
        torch.tensor([13.0]),
    )


def test_active_lane_plan_compacts_staggered_schedules_and_reconverges():
    plan = build_active_lane_plan(
        schedules=[
            [8, 1, 1, 2],
            [7, 1, 1, 3],
            [7, 2, 1, 2],
            [7, 1, 2, 2],
        ],
        start_step=2,
    )

    assert plan[0] == (9, (1, 2, 3))
    assert plan[-1] == (14, (0, 1, 2, 3))
    assert all(len(lanes) < 4 for _, lanes in plan[:-1])


def test_active_lane_plan_rejects_mismatched_spans():
    with pytest.raises(ValueError, match="same total span"):
        build_active_lane_plan([[2, 2], [2, 3]])
