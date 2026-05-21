# B5 Cross-Vehicle / Cross-Config Scaffold Report

This report aggregates M7 scaffold runs R034-R036. Ratios are 100-step RMSE relative to physics-only unless stated otherwise; lower is better.

| Run | System | Variant | Passed | Claim Supported | Seen / Phys | Held-out Road / Phys | Held-out Vehicle / Phys | Held-out Vehicle / Black-box | Vehicle Gap | Constraint Rate | Uncertainty OOD Lift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R034 | Base | E1+T1+F1+S1+M1a+V1+U0 | 1 | 0 | 0.703792 | 0.782969 | 0.742754 | 1.397572 | 1.055361 | 0.000000 | 1.141074 |
| R035 | Final single | E2+T1+F1+S1+M0-fixed+V2-small+U0 | 1 | 0 | 0.603213 | 0.721453 | 0.676170 | 1.283654 | 1.120946 | 0.000000 | 1.236491 |
| R036 | Final + U1 | E2+T1+F1+S1+M0-fixed+V2-small+U1 | 1 | 0 | 0.596453 | 0.728261 | 0.674862 | 1.246397 | 1.131459 | 0.000000 | 1.296895 |

## Selected Deployable Scaffold

```json
{
  "claim_supported": 0,
  "constraint_rate": 0.0,
  "gap_vehicle_over_seen": 1.120946240857456,
  "held_out_road_ratio": 0.7214529127772072,
  "held_out_vehicle_ratio": 0.6761696524004814,
  "held_out_vehicle_vs_black_box": 1.2836539615490097,
  "label": "Final single",
  "passed": 1,
  "run_id": "R035",
  "seen_ratio": 0.6032132744236268,
  "status": "success",
  "uncertainty_ood_lift": 1.2364911161267285,
  "variant": "E2+T1+F1+S1+M0-fixed+V2-small+U0"
}
```

## Notes

- `Final single` keeps `U0` because it is the single-model checkpoint candidate.
- `Final + U1` is the optional K=3 uncertainty wrapper and is not counted as the single-model checkpoint.
- `Claim Supported = 0` means this scaffold run completed but does not yet prove superiority over the black-box held-out-vehicle baseline.
- These are scaffold comparisons over generated DS1 data, not final PyTorch training results.
