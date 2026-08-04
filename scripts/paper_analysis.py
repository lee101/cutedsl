#!/usr/bin/env python3
"""Regenerate every number in the Latent Teleportation paper from raw results.

No figure in the paper is hand-typed. Run this and the .tex picks up whatever
the measurements currently say:

    python scripts/paper_analysis.py --out paper/generated

Two corrections are applied to the raw logs, and both are worth stating rather
than quietly fixing, because both changed a headline number:

1. Warm-up contamination. scripts/spec_e2e.py times the first baseline call
   without a warm-up iteration, so prompt 0's baseline absorbs CUDA context
   creation, kernel autotuning and lazy module init. Measured: 140.6 s against a
   28.3 s median for the same work at draft_k=3. That single row carries the
   reported 1.50x mean; the steady-state figure is 1.04x. Rows whose baseline
   exceeds WARMUP_FACTOR times the median baseline are reported separately
   instead of averaged in.

2. A degenerate final scheduler step. The last step of the 16-step schedule
   moves the latent by ~0, so relL2 -- error divided by actual movement --
   divides by nothing and reports 5.5e9 at t=14. That cell is excluded from
   aggregates and reported as the no-op it is.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC_BASE = Path("/sdb-disk/latentteleport-spec")
RESULTS = REPO / "results" / "speculative"

# A baseline this many times the median is warm-up, not a measurement.
WARMUP_FACTOR = 2.0
# relL2 above this is a division by ~zero movement, not a prediction error.
DEGENERATE_REL_L2 = 10.0
MODES = ("spec", "taylor", "skip")


def load_json(*candidates: Path):
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    return None


def split_warmup(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separates warm-up-contaminated rows from steady-state ones."""
    if len(rows) < 3:
        return rows, []
    median = st.median(r["t_baseline"] for r in rows)
    steady = [r for r in rows if r["t_baseline"] <= WARMUP_FACTOR * median]
    warm = [r for r in rows if r["t_baseline"] > WARMUP_FACTOR * median]
    return steady, warm


def e2e_table() -> dict:
    """Draft-length ablation: what each k costs and buys, warm-up excluded."""
    out = {}
    for k in (1, 2, 3):
        raw = load_json(
            RESULTS / f"e2e-16step-k{k}.json",
            SPEC_BASE / f"e2e-16step-k{k}" / "summary.json",
        )
        if raw is None:
            continue
        rows = raw["rows"]
        steady, warm = split_warmup(rows)
        entry = {
            "draft_k": k,
            "big_steps": rows[0].get("big_steps"),
            "total_steps": raw.get("steps", 16),
            "n_prompts": len(rows),
            "n_steady": len(steady),
            "n_warmup": len(warm),
            "warmup_baseline_s": [round(r["t_baseline"], 1) for r in warm],
            "median_baseline_s": round(st.median(r["t_baseline"] for r in steady), 1),
            "reported_mean_speedup_spec": raw.get("mean_speedup_spec"),
        }
        for mode in MODES:
            entry[f"speedup_{mode}"] = round(
                st.mean(r[f"speedup_{mode}"] for r in steady), 3)
            entry[f"psnr_{mode}"] = round(st.mean(r[f"psnr_{mode}"] for r in steady), 2)
        # Model calls saved is exact and hardware-independent; the wall clock is
        # neither. Reporting both is the whole point of the paper.
        entry["call_reduction"] = round(entry["total_steps"] / entry["big_steps"], 2)
        out[k] = entry
    return out


def gap_table() -> dict:
    """Predictability of the trajectory: relL2 of each forecaster by (t, k).

    relL2 is error over actual movement, so 1.0 means the predictor is no better
    than not moving at all. taylor1 is the training-free floor the learned
    walker has to beat.
    """
    raw = load_json(RESULTS / "gap-16step.json", SPEC_BASE / "gap-16step-512.json")
    if raw is None:
        return {}
    cells, degenerate = {}, []
    for key, value in raw.items():
        t = int(key.split("+k")[0][1:])
        k = int(key.split("+k")[1])
        if value.get("taylor1", 0) > DEGENERATE_REL_L2:
            degenerate.append({"t": t, "k": k, "taylor1": value["taylor1"]})
            continue
        cells[(t, k)] = value
    by_k = {}
    for k in sorted({k for _, k in cells}):
        vals = [v["taylor1"] for (_, kk), v in cells.items() if kk == k]
        aff = [v["affine"] for (_, kk), v in cells.items() if kk == k]
        by_k[k] = {
            "taylor1_mean": round(st.mean(vals), 3),
            "taylor1_min": round(min(vals), 3),
            "affine_mean": round(st.mean(aff), 3),
            "n": len(vals),
        }
    ts = sorted({t for t, _ in cells})
    best_t = min(ts, key=lambda t: cells.get((t, 1), {"taylor1": 9e9})["taylor1"])
    return {
        "cells": {f"t{t}+k{k}": v for (t, k), v in sorted(cells.items())},
        "by_k": by_k,
        "degenerate": degenerate,
        "t_range": [min(ts), max(ts)],
        "best_t_k1": best_t,
        "best_taylor1_k1": round(cells[(best_t, 1)]["taylor1"], 3),
    }


def train_table() -> dict:
    """Walker/interpolator training: did the learned model beat the free one?"""
    raw = load_json(RESULTS / "train-hist.json", SPEC_BASE / "ckpt-16step-512" / "hist.json")
    if raw is None:
        return {}
    walker = [e["rel_walker"] for e in raw]
    interp = [e["rel_interp"] for e in raw]
    return {
        "epochs": len(raw),
        "walker_best": round(min(walker), 3),
        "walker_best_epoch": int(min(range(len(walker)), key=lambda i: walker[i])),
        "walker_final": round(walker[-1], 3),
        "interp_best": round(min(interp), 3),
        "interp_final": round(interp[-1], 3),
        # The final epoch is not the best one for either net, so a run that
        # saves the last checkpoint ships a worse model than it trained.
        "final_is_best": min(walker) == walker[-1],
    }


def forecaster_ablation() -> dict:
    """Five-fold CPU ablation of cheap trajectory predictors."""
    raw = load_json(RESULTS / "forecaster-ablation.json")
    if not raw:
        return {}
    # Keep the paper-facing analysis compact.  Per-cell scores and fitted
    # coefficients remain available in the separately published raw result.
    return {
        "protocol": raw.get("protocol", {}),
        "summary": raw.get("summary", {}),
    }


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tables(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    e2e = data["e2e"]
    lines = [
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r"$k$ & big steps & call red. & \multicolumn{3}{c}{wall-clock speedup} & \multicolumn{3}{c}{PSNR vs baseline (dB)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r" & & & spec & taylor & skip & spec & taylor & skip \\",
        r"\midrule",
    ]
    for k in sorted(e2e):
        e = e2e[k]
        lines.append(
            f"{k} & {e['big_steps']} & {e['call_reduction']}$\\times$ & "
            f"{e['speedup_spec']:.2f} & {e['speedup_taylor']:.2f} & {e['speedup_skip']:.2f} & "
            f"{e['psnr_spec']:.1f} & {e['psnr_taylor']:.1f} & {e['psnr_skip']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_e2e.tex").write_text("\n".join(lines) + "\n")

    gap = data["gap"]
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$k$ & mean relL2 (taylor1) & best cell & mean relL2 (affine) & cells \\",
        r"\midrule",
    ]
    for k in sorted(gap.get("by_k", {})):
        b = gap["by_k"][k]
        lines.append(
            f"{k} & {b['taylor1_mean']:.3f} & {b['taylor1_min']:.3f} & "
            f"{b['affine_mean']:.3f} & {b['n']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_gap.tex").write_text("\n".join(lines) + "\n")

    forecaster = data.get("forecaster", {}).get("summary", {})
    lines = [
        r"\begin{tabular}{@{}crrrrr@{}}",
        r"\toprule",
        r"$k$ & Taylor-1 & avg. velocity & Taylor-2 & scaled momentum & two-delta fit \\",
        r"\midrule",
    ]
    for draft_k in sorted(forecaster, key=int):
        methods = forecaster[draft_k]["methods"]
        cells = []
        for method in (
            "taylor1",
            "average_velocity",
            "taylor2",
            "scaled_momentum",
            "two_delta_fit",
        ):
            value = methods[method]
            cells.append(f"{value['mean']:.3f}$\\pm${value['std']:.3f}")
        lines.append(f"{draft_k} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_forecaster_ablation.tex").write_text("\n".join(lines) + "\n")

    # Macros so prose can cite a number without anyone retyping it.
    macros = []
    k3 = e2e.get(3, {})
    macros.append(r"\newcommand{\ReportedSpeedupKThree}{%s}" % k3.get("reported_mean_speedup_spec", "?"))
    macros.append(r"\newcommand{\CorrectedSpeedupKThree}{%.2f}" % k3.get("speedup_spec", 0))
    macros.append(r"\newcommand{\CorrectedSkipKThree}{%.2f}" % k3.get("speedup_skip", 0))
    macros.append(r"\newcommand{\WarmupBaseline}{%s}" % (k3.get("warmup_baseline_s") or ["?"])[0])
    macros.append(r"\newcommand{\MedianBaseline}{%s}" % k3.get("median_baseline_s", "?"))
    macros.append(r"\newcommand{\CallReductionKThree}{%s}" % k3.get("call_reduction", "?"))
    k1 = e2e.get(1, {})
    macros.append(r"\newcommand{\PsnrTaylorKOne}{%.1f}" % k1.get("psnr_taylor", 0))
    macros.append(r"\newcommand{\PsnrSpecKOne}{%.1f}" % k1.get("psnr_spec", 0))
    macros.append(r"\newcommand{\SpeedupTaylorKOne}{%.2f}" % k1.get("speedup_taylor", 0))
    macros.append(r"\newcommand{\SpeedupSpecKOne}{%.2f}" % k1.get("speedup_spec", 0))
    macros.append(r"\newcommand{\BestTaylorRelLTwo}{%s}" % gap.get("best_taylor1_k1", "?"))
    macros.append(r"\newcommand{\BestTaylorStep}{%s}" % gap.get("best_t_k1", "?"))
    tr = data["train"]
    macros.append(r"\newcommand{\WalkerBest}{%s}" % tr.get("walker_best", "?"))
    macros.append(r"\newcommand{\WalkerFinal}{%s}" % tr.get("walker_final", "?"))
    macros.append(r"\newcommand{\InterpFinal}{%s}" % tr.get("interp_final", "?"))
    macros.append(r"\newcommand{\TrainEpochs}{%s}" % tr.get("epochs", "?"))
    deg = gap.get("degenerate") or [{}]
    # Rendered as LaTeX scientific notation rather than 1e+09, which math mode
    # would set as "1e + 09".
    value = deg[0].get("taylor1", 0)
    mantissa, exponent = f"{value:.2e}".split("e")
    macros.append(r"\newcommand{\DegenerateRelLTwo}{%s \times 10^{%d}}"
                  % (mantissa, int(exponent)))
    macros.append(r"\newcommand{\DegenerateStep}{%s}" % deg[0].get("t", "?"))
    (out_dir / "macros.tex").write_text("\n".join(macros) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "paper" / "generated"))
    args = ap.parse_args()

    data = {
        "e2e": e2e_table(),
        "gap": gap_table(),
        "train": train_table(),
        "forecaster": forecaster_ablation(),
    }
    out_dir = Path(args.out)
    write_tables(data, out_dir)
    (out_dir / "analysis.json").write_text(json.dumps(data, indent=2, default=str) + "\n")

    print(
        f"wrote {out_dir}/table_e2e.tex, table_gap.tex, "
        "table_forecaster_ablation.tex, macros.tex, analysis.json"
    )
    for k, e in sorted(data["e2e"].items()):
        print(f"  k={k}: reported {e['reported_mean_speedup_spec']}x -> corrected "
              f"{e['speedup_spec']}x (dropped {e['n_warmup']} warm-up row"
              f"{'s' if e['n_warmup'] != 1 else ''}: {e['warmup_baseline_s']} s "
              f"vs {e['median_baseline_s']} s median)")
    if data["gap"].get("degenerate"):
        for d in data["gap"]["degenerate"]:
            print(f"  excluded degenerate cell t={d['t']} k={d['k']}: relL2 {d['taylor1']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
