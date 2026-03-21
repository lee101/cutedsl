"""Head-to-head: CuteChronos2 vs upstream Chronos-2.
MAE equivalence + latency on short, medium, and long (8192) sequences."""

import time
import gc
import torch
import numpy as np

MODEL_ID = "amazon/chronos-2"
DEVICE = "cuda"
DTYPE = torch.bfloat16
N_WARMUP = 3
N_RUNS = 10

# --- test sequences ---
def make_linear(n, start=2, step=2):
    return torch.arange(start, start + step * n, step, dtype=torch.float32)

def make_sinusoidal(n, freq=0.05, amp=50, offset=100):
    t = torch.arange(n, dtype=torch.float32)
    return offset + amp * torch.sin(2 * np.pi * freq * t) + t * 0.1

def make_random_walk(n, seed=42):
    torch.manual_seed(seed)
    return 100.0 * torch.exp((torch.randn(n) * 0.02).cumsum(0))

TESTS = {
    "linear_short": (make_linear(7), make_linear(4, start=16, step=2)),     # [2..14] -> [16..22]
    "linear_64": (make_linear(64), make_linear(4, start=130, step=2)),
    "sine_512": (make_sinusoidal(512), None),
    "walk_512": (make_random_walk(512), None),
    "walk_2048": (make_random_walk(2048), None),
    "walk_8192": (make_random_walk(8192), None),
}
PRED_LEN = 4


def time_predict(pipe, ctx, pred_len, is_upstream, n_warmup, n_runs):
    def run():
        inp = [ctx] if is_upstream else ctx
        return pipe.predict(inp, prediction_length=pred_len, limit_prediction_length=False)

    for _ in range(n_warmup):
        run()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    preds = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        preds = run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return preds, times


def extract_median(preds, quantiles):
    mid = quantiles.index(0.5)
    return preds[0][0, mid, :]


def main():
    # --- load upstream ---
    print("Loading upstream Chronos2Pipeline...")
    from chronos import BaseChronosPipeline
    upstream = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=DTYPE)
    up_q = upstream.quantiles
    print(f"  done ({len(up_q)} quantiles)")

    # --- load cute ---
    print("Loading CuteChronos2Pipeline...")
    from cutechronos.pipeline import CuteChronos2Pipeline
    cute = CuteChronos2Pipeline.from_pretrained(MODEL_ID, device=DEVICE, dtype=DTYPE)
    cute_q = cute.quantiles
    print(f"  done ({len(cute_q)} quantiles)")

    # --- load cute compiled ---
    print("Loading CuteChronos2Pipeline (compiled, reduce-overhead)...")
    cute_compiled = CuteChronos2Pipeline.from_pretrained(
        MODEL_ID, device=DEVICE, dtype=DTYPE, compile_mode="reduce-overhead",
    )
    cute_c_q = cute_compiled.quantiles
    print(f"  done ({len(cute_c_q)} quantiles)")

    header = (f"{'Test':<18} {'Len':>5} | {'Up ms':>8} {'Cute ms':>8} {'Compiled':>8} | "
              f"{'SpeedE':>7} {'SpeedC':>7} | {'MAE-up':>10} {'MAE-cute':>10} {'MaxDiff':>8} {'MaxDiffC':>9}")
    print(f"\n{header}")
    print("-" * 130)

    for name, (ctx, actual) in TESTS.items():
        up_preds, up_times = time_predict(upstream, ctx, PRED_LEN, True, N_WARMUP, N_RUNS)
        cute_preds, cute_times = time_predict(cute, ctx, PRED_LEN, False, N_WARMUP, N_RUNS)
        comp_preds, comp_times = time_predict(cute_compiled, ctx, PRED_LEN, False, N_WARMUP + 2, N_RUNS)

        up_med = extract_median(up_preds, up_q)
        cute_med = extract_median(cute_preds, cute_q)
        comp_med = extract_median(comp_preds, cute_c_q)

        max_abs_diff = (up_med - cute_med).abs().max().item()
        max_abs_diff_c = (up_med - comp_med).abs().max().item()

        mae_up = (up_med - actual).abs().mean().item() if actual is not None else float("nan")
        mae_cute = (cute_med - actual).abs().mean().item() if actual is not None else float("nan")

        up_ms = np.mean(up_times) * 1000
        cute_ms = np.mean(cute_times) * 1000
        comp_ms = np.mean(comp_times) * 1000
        speedup_e = up_ms / max(cute_ms, 0.001)
        speedup_c = up_ms / max(comp_ms, 0.001)

        print(f"{name:<18} {len(ctx):>5} | {up_ms:>8.2f} {cute_ms:>8.2f} {comp_ms:>8.2f} | "
              f"{speedup_e:>6.1f}x {speedup_c:>6.1f}x | {mae_up:>10.4f} {mae_cute:>10.4f} {max_abs_diff:>8.4f} {max_abs_diff_c:>9.4f}")

    del upstream, cute, cute_compiled
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
