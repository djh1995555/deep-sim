# B4 Component Ablation Scaffold Report

This report aggregates M6 scaffold runs R015-R033. Ratios below are 100-step RMSE relative to physics-only; lower is better. These are scaffold-level comparisons, not final PyTorch module results.

| Run | Family | Variant | Passed | Val / Phys | Held-out Road / Phys | Held-out Vehicle / Phys | Residual / Physics | Constraint Rate | Teacher Oracle |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| R015 | Tire | T0 | 1 | 0.702755 | 0.783772 | 0.751968 | 1.009066 | 0.000000 | 0 |
| R016 | Tire | T1 | 1 | 0.700753 | 0.782969 | 0.742754 | 1.013495 | 0.000000 | 0 |
| R017 | Tire | T1-no-proj | 1 | 0.700753 | 0.782969 | 0.742754 | 1.013495 | 0.000000 | 0 |
| R018 | Tire | T2 | 1 | 0.830921 | 0.971058 | 0.784076 | 1.515302 | 0.000000 | 0 |
| R019 | Fz | F0 | 1 | 0.707250 | 0.782784 | 0.744143 | 1.008835 | 0.000000 | 0 |
| R020 | Fz | F1 | 1 | 0.700753 | 0.782969 | 0.742754 | 1.013495 | 0.000000 | 0 |
| R021 | Fz | F2 | 1 | 0.741209 | 0.864735 | 0.782331 | 1.142525 | 0.000000 | 1 |
| R022 | Steering | S0 | 1 | 0.701288 | 0.781565 | 0.741098 | 1.013950 | 0.000000 | 0 |
| R023 | Steering | S1 | 1 | 0.700753 | 0.782969 | 0.742754 | 1.013495 | 0.000000 | 0 |
| R024 | Mu | M0-fixed | 1 | 0.697565 | 0.780638 | 0.744277 | 1.004400 | 0.000000 | 0 |
| R025 | Mu | M1a | 1 | 0.700753 | 0.782969 | 0.742754 | 1.013495 | 0.000000 | 0 |
| R026 | Mu | M1b | 1 | 0.713719 | 0.796865 | 0.741810 | 1.054676 | 0.000000 | 0 |
| R027 | Mu | M2-oracle | 1 | 0.702229 | 0.762220 | 0.730039 | 0.990491 | 0.000000 | 1 |
| R027a | Encoder | E1 | 1 | 0.700753 | 0.782969 | 0.742754 | 1.013495 | 0.000000 | 0 |
| R027b | Encoder | E2 | 1 | 0.629220 | 0.741948 | 0.679350 | 0.991442 | 0.000000 | 0 |
| R027c | Encoder | E3 | 1 | 0.789969 | 0.863526 | 0.790480 | 1.308923 | 0.000000 | 0 |
| R028 | Vehicle | V0 | 1 | 1.000093 | 1.000196 | 1.000139 | 0.000000 | 0.000000 | 0 |
| R029 | Vehicle | V1 | 1 | 0.700753 | 0.782969 | 0.742754 | 1.013495 | 0.000000 | 0 |
| R030 | Vehicle | V1-large | 1 | 0.732373 | 0.834397 | 0.761689 | 1.097228 | 0.000000 | 0 |
| R031 | Vehicle | V2-small | 1 | 0.685365 | 0.767014 | 0.739265 | 1.017194 | 0.000000 | 0 |
| R032 | Uncertainty | U0 | 1 | 0.700753 | 0.782969 | 0.742754 | 1.013495 | 0.000000 | 0 |
| R033 | Uncertainty | U1 | 1 | 0.665915 | 0.793733 | 0.760209 | 0.967944 | 0.000000 | 0 |

## Family Winners

- Tire: `T1` has the lowest validation ratio among deployable scaffold variants (`0.700753`).
- Fz: `F1` has the lowest validation ratio among deployable scaffold variants (`0.700753`).
- Steering: `S1` has the lowest validation ratio among deployable scaffold variants (`0.700753`).
- Mu: `M0-fixed` has the lowest validation ratio among deployable scaffold variants (`0.697565`).
- Encoder: `E2` has the lowest validation ratio among deployable scaffold variants (`0.629220`).
- Vehicle: `V2-small` has the lowest validation ratio among deployable scaffold variants (`0.685365`).
- Uncertainty: `U1` has the lowest validation ratio among deployable scaffold variants (`0.665915`).

## Notes

- `F2` and `M2-oracle` use teacher-only labels/features as simulator-only upper-bound or auxiliary scaffolds.
- A passed ablation run means the comparison completed under the shared DS1 split and emitted metrics; it does not mean the variant is selected for the final model.
- Final component selection still needs the full PyTorch implementation and multi-seed confirmation on training-scale DS1.
