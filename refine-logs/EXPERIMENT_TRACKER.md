# Experiment Tracker

当前 tracker 以本文档为准。`refine-logs/EXPERIMENT_TRACKER_20260518_212918.md` 是历史快照，包含旧版命名，不再作为当前执行依据。

## First Runs

| Run ID | Milestone | Purpose                                     | System / Variant                                                           | Priority | Status |
| ------ | --------- | ------------------------------------------- | -------------------------------------------------------------------------- | -------- | ------ |
| R000   | M0        | B0.1 simulator minimal implementation       | high-fidelity teacher simulator v0                                         | MUST     | DONE   |
| R000a  | M0        | B0.2 tire/load validation                   | teacher simulator tire + suspension                                        | MUST     | DONE   |
| R000b  | M0        | B0.3 road scenario generation               | teacher simulator road μ map                                               | MUST     | DONE   |
| R000c  | M0        | B0.4 sensor/actuator realism                | actuator + sensor model                                                    | MUST     | DONE   |
| R000d  | M0        | B0.5 dataset export split                   | dataset builder                                                            | MUST     | DONE   |
| R000e  | M1        | B0.6 scenario matrix v1                     | dataset generator                                                          | MUST     | DONE   |
| R000f  | M1        | B0.7 single vehicle parameter randomization | teacher simulator vehicle_A                                                | MUST     | DONE   |
| R000g  | M1        | B0.8 dataset split generation               | dataset builder                                                            | MUST     | DONE   |
| R000h  | M1        | B0.9 dataset QA                             | generated dataset v1                                                       | MUST     | DONE   |
| R001   | M2        | B1.1 schema / 字段角色检查                        | data loader                                                                | MUST     | DONE   |
| R002   | M2        | B1.2 teacher simulator 物理一致性检查              | teacher outputs                                                            | MUST     | DONE   |
| R003   | M2        | B1.3 时间对齐 / dt 检查                           | sequence timestamps                                                        | MUST     | DONE   |
| R004   | M2        | B1.4 派生物理量检查                                | slip / steering / wheel dynamics                                           | MUST     | DONE   |
| R004a  | M2        | B1.5 最小可学习性检查                               | tiny black-box + tiny base hybrid                                          | MUST     | DONE   |
| R004b  | M2        | B1.6 physics-only rollout smoke test        | physics-only                                                               | MUST     | DONE   |
| R004c  | M3        | B2.1 proxy perturbation profiles            | teacher simulator perturbation generator                                   | MUST     | DONE   |
| R004d  | M3        | B2.2 proxy target windows                   | held-out vehicle/config target windows                                     | MUST     | DONE   |
| R004e  | M3        | B2.3 proxy distribution sanity              | perturbed DS1 target windows                                               | MUST     | DONE   |
| R005   | M4        | B3.1 physics-only baseline                  | student physics backbone only                                              | MUST     | DONE   |
| R006   | M4        | B3.2 black-box baseline                     | TCN / GRU / MLP direct predictor + N-BEATSx direct multi-horizon predictor | MUST     | DONE   |
| R007   | M4        | B3.3 baseline fairness audit                | matched inputs / budget / horizons                                         | MUST     | DONE   |
| R008   | M4        | B3.4 baseline rollout report                | physics-only vs black-box                                                  | MUST     | DONE   |
| R009   | M5        | B3.5 base hybrid training                   | E1 + T1 + F1 + S1 + M1a + V1 + U0                                          | MUST     | DONE   |
| R010   | M5        | B3.6 base seen-config evaluation            | base hybrid on seen vehicle/config                                         | MUST     | DONE   |
| R011   | M5        | B3.7 base held-out road evaluation          | base hybrid on held-out road μ                                             | MUST     | DONE   |
| R012   | M5        | B3.8 base held-out vehicle evaluation       | base hybrid on held-out vehicle/config                                     | MUST     | DONE   |
| R013   | M5        | B3.9 base residual/constraint audit         | residual magnitude + physical constraints                                  | MUST     | DONE   |
| R014   | M5        | B3.10 base seed replication                 | B3 key table 3 seeds                                                       | MUST     | DONE   |

## Later Explicit Runs

| Run ID | Milestone | Purpose | System / Variant | Priority | Status |
|--------|-----------|---------|------------------|----------|--------|
| R015-R018 | M6 | tire residual ablation | T0 / T1 / T1-no-proj / T2 | MUST | DONE |
| R019-R021 | M6 | Fz residual ablation | F0 / F1 / F2 | MUST | DONE |
| R022-R023 | M6 | steering ablation | S0 / S1 | MUST | DONE |
| R024-R027 | M6 | MuHead ablation | M0-fixed / M1a / M1b / M2-oracle | MUST | DONE |
| R027a-R027c | M6 | shared encoder ablation | E1 / E2 / E3 | MUST | DONE |
| R028-R031 | M6 | vehicle residual ablation | V0 / V1 / V1-large / V2-small | MUST | DONE |
| R032-R033 | M6 | uncertainty ablation | U0 / U1 K=3 ensemble | MUST | DONE |
| R034-R036 | M7 | cross-vehicle/config generalization | final single model vs baselines | MUST | DONE |
| R037 | M8 | final single model freeze | selected E/T/F/S/M/V/U configuration | MUST | DONE |
| R038-R045 | M9 | target fine-tune data efficiency | FT0-FT6 × FTD0-FTD5 | MUST | DONE |
| R100 | P3 | PyTorch data loader smoke | canonical DS1 loader through runner | MUST | DONE |
| R101 | P3 | PyTorch forward/loss smoke | E2+T1+F1+S1+M0+V2+U0 one-step loss | MUST | DONE |
| R102 | P3 | PyTorch tiny overfit smoke | small DS1 train subset with AdamW | MUST | DONE |
| R103 | P3 | PyTorch rollout smoke | short autoregressive rollout | MUST | DONE |
| R104 | P3 | PyTorch checkpoint smoke | save/load trained-weight format | MUST | DONE |
| R105 | P4 | PyTorch GPU forward smoke | CUDA-required E2+T1+F1+S1+M0+V2+U0 one-step loss | MUST | DONE |
| R106 | P4 | PyTorch GPU tiny overfit | CUDA-required backward + AdamW optimizer loss descent | MUST | DONE |
| R107 | P5 | PyTorch one-step training | train/validation/checkpoint on CUDA | MUST | DONE |
| R108 | P5 | PyTorch rollout eval | rollout from R107 checkpoint | MUST | DONE |
| R109 | P5 | PyTorch resume/eval-only smoke | resume R107 checkpoint and evaluate | MUST | DONE |
| R110 | P5 | PyTorch black-box baseline | direct TCN train/validation/checkpoint on CUDA | MUST | DONE |
| R111 | P6 | PyTorch base model small training | final single hybrid small training on CUDA | MUST | DONE |
| R112 | P6 | PyTorch fair small comparison | matched-budget hybrid vs direct TCN/GRU/N-BEATS with one-step + rollout metrics | MUST | DONE |
| R113 | P6 | PyTorch component variant smoke | E/T/F/S/M/V switches plus direct baselines forward on CUDA | MUST | DONE |
| R114 | P7 | PyTorch fine-tune adapter smoke | FT0-FT6 trainability over two target data buckets | MUST | DONE |
| R115 | P7 | PyTorch deep ensemble smoke | U1 K=3 ensemble checkpoint + predictive variance | MUST | DONE |
| R046 | M10 | DS2 extreme dataset smoke | emergency/fishhook/lane-change scaffold data generation | NICE | DONE |
| R047 | M10 | DS2 MoE tire residual smoke | T1/T2/T3-MoE forward coverage on DS2 | NICE | DONE |
| R200-R216 | P8 | Generated PyTorch ablation matrix | real trainable configs for E/T/F/S/M/V/U single-factor comparisons | MUST | READY |
| R300-R334 | P9 | Generated PyTorch fine-tune matrix | FT0-FT6 × five fine-tune data buckets | MUST | READY |
| R048+ | M10 | DS2 MoE tire residual training/eval | full large-slip MoE vs T1/T2 comparison | NICE | READY |

## Development Tooling

| Item | Milestone | Purpose | Artifact | Status |
|------|-----------|---------|----------|--------|
| Q1 | P10 | Local experiment queue | `experiments/experiment_queue.py` with retry, dry-run, skip-success, state JSON, per-attempt logs, and optional post-rollout eval | DONE |
| Q2 | P10 | Queue smoke on existing successful run | `runs/queue_state_smoke.json` and R111 `artifacts/post_rollout_eval/summary.json` | DONE |
| Q3 | P10 | Matrix status/result report | `experiments/matrix_report.py`, `reports/PYTORCH_MATRIX_REPORT.md/json` | DONE |
| Q4 | P10 | CSV real-data adapter | `experiments/real_data_adapter.py` with canonical dataset validation | DONE |
| Q5 | P10 | Training governance | best checkpoint, early stopping, LR scheduler, nonfinite loss guard, filtered rollout eval in `experiments/torch_training.py` | DONE |
