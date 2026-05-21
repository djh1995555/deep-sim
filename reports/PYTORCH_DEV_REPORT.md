# PyTorch Development Report

## Run Status

| Run | Status | Primary Metric | Value |
|---|---|---:|---:|
| R112 | success | torch_fair_compare_passed | 1 |
| R113 | success | torch_model_variant_smoke_passed | 1 |
| R114 | success | torch_fine_tune_smoke_passed | 1 |
| R115 | success | torch_ensemble_training_passed | 1 |

## R112 Fair Small Comparison

| Variant | Type | Val MSE | Val Normalized MSE | Rollout RMSE |
|---|---|---:|---:|---:|
| hybrid | hybrid | 6.027180 | 663.606339 | 11.448463 |
| direct_tcn | direct_tcn | 1.355315 | 1248.362701 | 7.612251 |
| direct_gru | direct_gru | 1.511166 | 1383.388489 | 7.430042 |
| direct_nbeats | direct_nbeats | 1.438999 | 1214.542908 | 6.871233 |

Interpretation:

- `hybrid_vs_best_black_box_mse_ratio = 4.447071`.
- `hybrid_vs_best_black_box_rollout_ratio = 1.666144`.
- This is a development-scale run; it validates fair comparison plumbing and highlights that the current hybrid still needs model/loss tuning before any superiority claim.

## R113 Variant Coverage

- Variant forward checks passed for `14` model/component variants.

## R114 Fine-Tune Adapter Smoke

| Mode | Buckets | Trainable Params | Val MSE |
|---|---|---:|---:|
| FT0 | FTD1 | 0 | 0.796404 |
| FT0 | FTD1+FTD2+FTD3 | 0 | 0.797560 |
| FT1 | FTD1 | 6092 | 1.312392 |
| FT1 | FTD1+FTD2+FTD3 | 6092 | 0.697363 |
| FT2 | FTD1 | 11204 | 0.792034 |
| FT2 | FTD1+FTD2+FTD3 | 11204 | 0.796707 |
| FT3 | FTD1 | 11204 | 0.797632 |
| FT3 | FTD1+FTD2+FTD3 | 11204 | 0.797508 |
| FT4 | FTD1 | 23956 | 0.794321 |
| FT4 | FTD1+FTD2+FTD3 | 23956 | 0.796689 |
| FT5 | FTD1 | 11009 | 0.797559 |
| FT5 | FTD1+FTD2+FTD3 | 11009 | 0.797633 |
| FT6 | FTD1 | 126785 | 0.697050 |
| FT6 | FTD1+FTD2+FTD3 | 126785 | 0.697986 |

## R115 Deep Ensemble Smoke

- Members: `3`.
- Ensemble MSE: `6.010266`.
- Predictive variance: `0.085524`.
