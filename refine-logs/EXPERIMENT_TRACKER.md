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
| R100 | P3 | PyTorch data loader smoke | canonical DS1 loader through runner | MUST | BLOCKED: PyTorch missing |
| R101 | P3 | PyTorch forward/loss smoke | E2+T1+F1+S1+M0+V2+U0 one-step loss | MUST | READY after PyTorch install |
| R102 | P3 | PyTorch tiny overfit smoke | small DS1 train subset with AdamW | MUST | READY after PyTorch install |
| R103 | P3 | PyTorch rollout smoke | short autoregressive rollout | MUST | READY after PyTorch install |
| R104 | P3 | PyTorch checkpoint smoke | save/load trained-weight format | MUST | READY after PyTorch install |
| R046+ | M10 | MoE tire residual after extreme data | DS2 MoE tire residual | NICE | TODO |
