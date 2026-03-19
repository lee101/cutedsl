import torch
from cutechronos.pipeline import CuteChronos2Pipeline

pipe = CuteChronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device="cuda",
    dtype=torch.bfloat16,
)

context = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
preds = pipe.predict(context, prediction_length=4)

# preds[0] shape: (1, 21, 4) -- 21 quantiles, 4 steps
p = preds[0].squeeze(0)  # (21, 4)
quantiles = pipe.quantiles
median_idx = quantiles.index(0.5)

print(f"Input: [2, 4, 6, 8, 10, 12, 14]")
print(f"Expected next: 16, 18, 20, 22")
print(f"\nMedian predictions (q0.5): {p[median_idx].tolist()}")
print(f"All quantiles for step 1:")
for i, q in enumerate(quantiles):
    print(f"  q{q:.2f}: {p[i, 0]:.2f}")
