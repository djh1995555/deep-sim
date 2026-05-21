# B6 Fine-Tune Data Efficiency Scaffold Report

This report aggregates M9 scaffold runs R038-R045. Rows are target-window 100-step RMSE values; lower is better.

## Summary

| Metric | Value |
|---|---:|
| fine_tune_best_full_relative_to_ft0 | 2.458700190937411 |
| fine_tune_best_small_relative_to_ft0 | 0.8669265831938702 |
| fine_tune_best_small_vs_full_gap | 0.3525954837394563 |
| fine_tune_completed_cell_count | 31 |
| fine_tune_completed_variant_count | 7 |
| fine_tune_ft0_test_rmse | 0.598497878462135 |
| fine_tune_summary_passed | 1 |

## Best Small Module

```json
{
  "adapter_to_base_ratio": 0.27775341718452334,
  "bucket": "FTD1",
  "ft_variant": "FT5",
  "test_constraint_violation_rate": 0.0,
  "test_relative_to_ft0": 0.8669265831938702,
  "test_rmse": 0.5188537208239589,
  "train_episode_count": 9,
  "train_transition_count": 900,
  "validation_rmse": 0.5521717229520635
}
```

## Cells

| FT | Bucket | Train Episodes | Test RMSE | Relative to FT0 | Constraint Rate | Adapter/Base |
|---|---|---:|---:|---:|---:|---:|
| FT0 | FTD0 | 0 | 0.598498 | 1.000000 | 0.000000 | 0.000000 |
| FT1 | FTD1 | 9 | 0.628631 | 1.050348 | 0.000000 | 0.222708 |
| FT1 | FTD2 | 18 | 0.605589 | 1.011849 | 0.000000 | 0.286920 |
| FT1 | FTD3 | 27 | 0.593542 | 0.991719 | 0.000000 | 0.244527 |
| FT1 | FTD4 | 36 | 0.572011 | 0.955744 | 0.000000 | 0.207816 |
| FT1 | FTD5 | 45 | 0.575695 | 0.961899 | 0.000000 | 0.196214 |
| FT2 | FTD1 | 9 | 1.508655 | 2.520736 | 0.000000 | 0.545958 |
| FT2 | FTD2 | 18 | 0.865408 | 1.445967 | 0.000000 | 0.450442 |
| FT2 | FTD3 | 27 | 0.869699 | 1.453136 | 0.000000 | 0.429042 |
| FT2 | FTD4 | 36 | 0.793610 | 1.326004 | 0.000000 | 0.300546 |
| FT2 | FTD5 | 45 | 0.788015 | 1.316655 | 0.000000 | 0.261505 |
| FT3 | FTD1 | 9 | 1.367282 | 2.284523 | 0.000000 | 0.475864 |
| FT3 | FTD2 | 18 | 0.554240 | 0.926053 | 0.000000 | 0.445459 |
| FT3 | FTD3 | 27 | 0.563514 | 0.941547 | 0.000000 | 0.366436 |
| FT3 | FTD4 | 36 | 0.685376 | 1.145161 | 0.000000 | 0.287624 |
| FT3 | FTD5 | 45 | 0.630075 | 1.052760 | 0.000000 | 0.245578 |
| FT4 | FTD1 | 9 | 1.897028 | 3.169649 | 0.000000 | 0.800698 |
| FT4 | FTD2 | 18 | 1.019751 | 1.703850 | 0.000000 | 0.667553 |
| FT4 | FTD3 | 27 | 1.189430 | 1.987360 | 0.000000 | 0.585950 |
| FT4 | FTD4 | 36 | 0.891195 | 1.489053 | 0.000000 | 0.468699 |
| FT4 | FTD5 | 45 | 1.054151 | 1.761328 | 0.000000 | 0.465778 |
| FT5 | FTD1 | 9 | 0.518854 | 0.866927 | 0.000000 | 0.277753 |
| FT5 | FTD2 | 18 | 0.552613 | 0.923333 | 0.000000 | 0.255164 |
| FT5 | FTD3 | 27 | 0.567748 | 0.948622 | 0.000000 | 0.229964 |
| FT5 | FTD4 | 36 | 0.537639 | 0.898314 | 0.000000 | 0.196524 |
| FT5 | FTD5 | 45 | 0.578836 | 0.967147 | 0.000000 | 0.181839 |
| FT6 | FTD1 | 9 | 4.154439 | 6.941444 | 0.000000 | 0.591481 |
| FT6 | FTD2 | 18 | 1.471527 | 2.458700 | 0.000000 | 0.681957 |
| FT6 | FTD3 | 27 | 1.671418 | 2.792689 | 0.000000 | 0.612917 |
| FT6 | FTD4 | 36 | 2.809543 | 4.694324 | 0.000000 | 0.526353 |
| FT6 | FTD5 | 45 | 2.741715 | 4.580994 | 0.000000 | 0.351314 |

## Notes

- `FT0` is no fine-tune and only has `FTD0`.
- `FT1-FT6` use cumulative target train episodes from `FTD1` to the reported bucket.
- This is not a PyTorch fine-tuning result; it validates the experiment matrix, target-window split, metrics, and reporting path.
