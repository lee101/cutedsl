#!/usr/bin/env python3
"""Cross-validated anchor-schedule search for latent teleportation.

The experiment treats a stored trajectory as a sequence of true refinement
anchors.  For a fixed number of forecast intervals, it asks where those anchors
should be placed: uniformly, or at a schedule selected on the training folds.
Each interval is evaluated from its true starting anchor, so this is an offline
schedule-quality proxy rather than a branched end-to-end sampler benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import product
from pathlib import Path

import torch


def _dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dtype=torch.float64)


def _rel_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    anchor: torch.Tensor,
) -> float:
    movement = (target - anchor).flatten(1).norm(dim=1).clamp_min(1e-8)
    error = (prediction - target).flatten(1).norm(dim=1)
    return float((error / movement).mean())


def _cell_scores(
    train: torch.Tensor,
    test: torch.Tensor,
    step: int,
    horizon: int,
) -> dict[str, dict[str, float] | float]:
    anchor_train = train[:, step]
    move_train = train[:, step + horizon] - anchor_train
    delta_train = anchor_train - train[:, step - 1]
    velocity_train = horizon * delta_train
    alpha = _dot(velocity_train, move_train) / _dot(
        velocity_train, velocity_train
    ).clamp_min(1e-12)

    outputs: dict[str, dict[str, float] | float] = {"alpha": float(alpha)}
    for split, trajectories in (("train", train), ("test", test)):
        anchor = trajectories[:, step]
        target = trajectories[:, step + horizon]
        delta = anchor - trajectories[:, step - 1]
        outputs[split] = {
            "taylor1": _rel_l2(anchor + horizon * delta, target, anchor),
            "scaled_momentum": _rel_l2(
                anchor + float(alpha) * horizon * delta,
                target,
                anchor,
            ),
        }
    return outputs


def _compositions(total: int, parts: int, max_horizon: int) -> list[tuple[int, ...]]:
    return [
        values
        for values in product(range(1, max_horizon + 1), repeat=parts)
        if sum(values) == total
    ]


def _uniform_schedule(total: int, parts: int) -> tuple[int, ...]:
    short, extra = divmod(total, parts)
    return tuple(short + (index < extra) for index in range(parts))


def _schedule_cells(start_step: int, schedule: tuple[int, ...]) -> list[tuple[int, int]]:
    cells = []
    step = start_step
    for horizon in schedule:
        cells.append((step, horizon))
        step += horizon
    return cells


def _schedule_score(
    cells: dict[tuple[int, int], dict[str, dict[str, float] | float]],
    start_step: int,
    schedule: tuple[int, ...],
    split: str,
    method: str,
) -> float:
    values = [
        float(cells[cell][split][method])  # type: ignore[index]
        for cell in _schedule_cells(start_step, schedule)
    ]
    return statistics.mean(values)


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(values), 6),
        "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
    }


def run(
    trajectory_dir: Path,
    folds: int,
    budgets: tuple[int, ...],
    max_horizon: int,
) -> dict:
    files = sorted(trajectory_dir.glob("*.pt"))
    if len(files) < folds:
        raise ValueError(f"need at least {folds} trajectories, found {len(files)}")
    trajectories = torch.stack(
        [
            torch.load(path, map_location="cpu", weights_only=False)["latents"]
            for path in files
        ]
    ).float()

    n_trajectories, n_steps = trajectories.shape[:2]
    start_step = 2
    end_step = n_steps - 2  # exclude the recorded duplicate/no-op final latent
    span = end_step - start_step
    all_indices = torch.arange(n_trajectories)
    fold_rows = []

    for fold in range(folds):
        test_mask = all_indices.remainder(folds) == fold
        train = trajectories[~test_mask]
        test = trajectories[test_mask]
        cells = {}
        for step in range(start_step, end_step):
            for horizon in range(1, min(max_horizon, end_step - step) + 1):
                cells[(step, horizon)] = _cell_scores(
                    train, test, step, horizon
                )

        for budget in budgets:
            schedules = _compositions(span, budget, max_horizon)
            if not schedules:
                raise ValueError(
                    f"no schedules cover span={span} with budget={budget}, "
                    f"max_horizon={max_horizon}"
                )
            ranked = sorted(
                schedules,
                key=lambda schedule: _schedule_score(
                    cells,
                    start_step,
                    schedule,
                    "train",
                    "scaled_momentum",
                ),
            )
            selected = ranked[0]
            uniform = _uniform_schedule(span, budget)
            staggered = ranked[: min(4, len(ranked))]
            fold_rows.append(
                {
                    "fold": fold,
                    "forecast_intervals": budget,
                    "uniform_schedule": list(uniform),
                    "selected_schedule": list(selected),
                    "top4_schedules": [list(schedule) for schedule in staggered],
                    "test_rel_l2": {
                        "uniform_taylor1": round(
                            _schedule_score(
                                cells, start_step, uniform, "test", "taylor1"
                            ),
                            6,
                        ),
                        "uniform_scaled": round(
                            _schedule_score(
                                cells,
                                start_step,
                                uniform,
                                "test",
                                "scaled_momentum",
                            ),
                            6,
                        ),
                        "aligned_scaled": round(
                            _schedule_score(
                                cells,
                                start_step,
                                selected,
                                "test",
                                "scaled_momentum",
                            ),
                            6,
                        ),
                        "top4_staggered_scaled": round(
                            statistics.mean(
                                _schedule_score(
                                    cells,
                                    start_step,
                                    schedule,
                                    "test",
                                    "scaled_momentum",
                                )
                                for schedule in staggered
                            ),
                            6,
                        ),
                    },
                }
            )

    summary = {}
    for budget in budgets:
        rows = [row for row in fold_rows if row["forecast_intervals"] == budget]
        methods = {
            method: _mean_std([row["test_rel_l2"][method] for row in rows])
            for method in (
                "uniform_taylor1",
                "uniform_scaled",
                "aligned_scaled",
                "top4_staggered_scaled",
            )
        }
        baseline = methods["uniform_taylor1"]["mean"]
        for method in (
            "uniform_scaled",
            "aligned_scaled",
            "top4_staggered_scaled",
        ):
            methods[method]["improvement_vs_uniform_taylor_pct"] = round(
                100.0 * (1.0 - methods[method]["mean"] / baseline), 2
            )
        summary[str(budget)] = {
            "methods": methods,
            "selected_schedules": [row["selected_schedule"] for row in rows],
        }

    return {
        "protocol": {
            "trajectory_dir": str(trajectory_dir),
            "n_trajectories": n_trajectories,
            "n_steps": n_steps,
            "folds": folds,
            "fold_assignment": "sorted trajectory index modulo fold count",
            "start_step": start_step,
            "end_step_inclusive": end_step,
            "covered_intervals": span,
            "max_horizon": max_horizon,
            "interval_budgets": list(budgets),
            "metric": "mean local relL2 across forecast intervals",
            "scope": (
                "offline true-anchor schedule proxy; does not simulate accumulated "
                "branched sampler error or end-to-end latency"
            ),
        },
        "summary": summary,
        "fold_rows": fold_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validated latent-teleport anchor schedule ablation"
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("/sdb-disk/latentteleport-spec/trajs-16step-512"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=(3, 4, 6))
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/speculative/schedule-ablation.json"),
    )
    args = parser.parse_args()

    result = run(
        args.trajectory_dir,
        args.folds,
        tuple(args.budgets),
        args.max_horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    for budget, row in result["summary"].items():
        methods = row["methods"]
        print(
            f"intervals={budget}: Taylor {methods['uniform_taylor1']['mean']:.3f}; "
            f"uniform scaled {methods['uniform_scaled']['mean']:.3f}; "
            f"aligned scaled {methods['aligned_scaled']['mean']:.3f}; "
            f"top4 staggered {methods['top4_staggered_scaled']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
