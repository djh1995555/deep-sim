# Experiment Tracker

> Historical snapshot from 2026-05-18. This file contains earlier `G* / S2 / S3` naming and is not the current execution tracker. Current tracker: `refine-logs/EXPERIMENT_TRACKER.md`.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R000 | M-1 | simulator minimal implementation | high-fidelity teacher simulator v0 | dry steady | stable simulation, exported fields | MUST | TODO | Build minimal high-DOF generator |
| R000a | M-1 | tire/load validation | teacher simulator tire + suspension | synthetic braking/turning | `Fz/Fx/Fy`, load transfer signs | MUST | TODO | Braking front load, turning outside load |
| R000b | M-1 | road scenario generation | teacher simulator road μ map | dry/wet/snow/ice/Split-μ/transition | scenario labels, μ_true_i | MUST | TODO | Wheel-level μ export |
| R000c | M-1 | sensor/actuator realism | actuator + sensor model | representative scenarios | delay/noise/filter stats | MUST | TODO | Match observable-only deployment assumption |
| R000d | M-1 | dataset export split | dataset builder | all initial scenarios | observable schema vs teacher-only schema | MUST | TODO | Prevent teacher leakage |
| R000e | M-0.5 | scenario matrix v1 | dataset generator | S001-S006, S101-S106, S201-S206, S301-S303, S401-S403 | scenario coverage | MUST | TODO | First-version single-vehicle matrix |
| R000f | M-0.5 | single vehicle parameter randomization | teacher simulator vehicle_A | internal parameter sweeps | metadata/hash coverage | MUST | TODO | mass/inertia/cg/suspension not exposed to student |
| R000g | M-0.5 | dataset split generation | dataset builder | train/val/test by episode/scenario | leakage checks | MUST | TODO | No random time-slice leakage |
| R000h | M-0.5 | dataset QA | generated dataset v1 | all scenarios | missing/dt/units/Fz/ellipse checks | MUST | TODO | Gate before M0 |
| R001 | M0 | B0.1 schema / 字段角色检查 | data loader | 5-10 short sequences | missing fields, teacher leakage | MUST | TODO | 输入张量不得包含 teacher-only 字段 |
| R002 | M0 | B0.2 teacher simulator 物理一致性检查 | teacher outputs | synthetic braking/turning/Split-μ | Fz/Fx/Fy/μ consistency | MUST | TODO | 检查 Fz>=0、载荷转移方向、摩擦约束 |
| R003 | M0 | B0.3 时间对齐 / dt 检查 | sequence timestamps | all initial scenarios | dt, dropped frames, x/u alignment | MUST | TODO | 执行器 delay 不能被错误对齐消掉 |
| R004 | M0 | B0.4 派生物理量检查 | slip / steering / wheel dynamics | dry/wet short sequences | κ/α/delta/wheel consistency | MUST | TODO | 检查 slip 符号和低速处理 |
| R004a | M0 | B0.5 最小可学习性检查 | tiny black-box + tiny G1 | tiny split | train loss decreases | MUST | TODO | 检查 normalization、target alignment |
| R004b | M0 | B0.6 physics-only rollout smoke test | physics-only | dry steady/gentle turning | 1s/5s RMSE, violations | MUST | TODO | 不应立即发散 |
| R005 | M1 | baseline | physics-only + S2 | dry/wet/low-μ | rollout RMSE, violations | MUST | TODO | Baseline lower complexity |
| R006 | M1 | baseline | black-box TCN/GRU | dry/wet/low-μ | rollout RMSE | MUST | TODO | No physics inputs beyond observables |
| R007 | M1 | baseline | physics-only on Split-μ | Split-μ | `vy/r/omega_i` RMSE | MUST | TODO | Finds physics-only failure modes |
| R008 | M1 | baseline | black-box on Split-μ | Split-μ | `vy/r/omega_i` RMSE | MUST | TODO | Black-box reference |
| R009 | M2 | G1 no residual | G1-V0 | dry/wet/low-μ | rollout RMSE | MUST | TODO | TirePhysics only |
| R010 | M2 | G1 derivative residual | G1-V1 | dry/wet/low-μ | rollout RMSE, residual mag | MUST | TODO | Default if stable |
| R011 | M2 | G1 state residual | G1-V2 | dry/wet/low-μ | rollout RMSE, residual mag | MUST | TODO | Compare fixed-dt correction |
| R012 | M2 | G1 Split-μ eval | best G1 | Split-μ | yaw/r/omega error | MUST | TODO | Decision for tire residual need |
| R013 | M2 | G1 transition eval | best G1 | dry->wet/snow | rollout RMSE, μ uncertainty | MUST | TODO | Road transition |
| R014 | M2 | G1 residual audit | best G1 | all val | residual norms | MUST | TODO | Check residual swallowing |
| R015 | M3 | G2 force residual | G2-V1 | dry/wet/low-μ | rollout RMSE, violations | MUST | TODO | ΔFx/ΔFy + projection |
| R016 | M3 | G2 Split-μ | G2-V1 | Split-μ | `vy/r/omega_i` RMSE | MUST | TODO | Main low-μ evidence |
| R017 | M3 | G2 transition | G2-V1 | transition | rollout RMSE | MUST | TODO | Compare to G1 |
| R018 | M3 | residual allocation | G1 vs G2 | all val | vehicle residual mag | MUST | TODO | Does tire residual reduce global residual? |
| R019 | M3 | projection audit | G2-V1 | all val | ellipse violation | MUST | TODO | Projection/penalty check |
| R020 | M4 | G3 parameter residual | G3-V1 | dry/wet/low-μ | rollout RMSE | MUST | TODO | ΔC/Δμ_scale |
| R021 | M4 | G3 Split-μ | G3-V1 | Split-μ | `vy/r/omega_i` RMSE | MUST | TODO | Compare to G2 |
| R022 | M4 | parameter plausibility | G3-V1 | all val | bounds, smoothness | MUST | TODO | Interpretability check |
| R023 | M4 | G2 vs G3 decision | best G2/G3 | all test | main metrics | MUST | TODO | Select tire residual form |
| R024 | M5 | Fz residual | best main + F1 | low-μ braking/turning | roll/pitch/yaw RMSE | MUST | TODO | Needs Fz module |
| R025 | M5 | Fz teacher | best main + F2 | low-μ braking/turning | Fz teacher RMSE, rollout | MUST | TODO | Only if simulator exports Fz_true; if unavailable, mark blocked |
| R026 | M5 | steering ablation | S2 vs S3 | steering/low-μ turning | `vy/r/yaw`, Δdelta | MUST | TODO | S3 must stay bounded |
| R027 | M5 | uncertainty | U0 single | validation | NLL, coverage | MUST | TODO | Single model logvar |
| R028 | M5 | uncertainty | U1 K=3 ensemble | validation | NLL, coverage, corr | MUST | TODO | After best single model |
| R029 | M5 | black-box OOD | black-box | held-out road/vehicle-like | rollout RMSE | NICE | TODO | Compare generalization |
| R030 | M6 | geometry shift | best main | held-out geometry sim | rollout RMSE | NICE | TODO | New vehicle proxy |
| R031 | M6 | adapter fine-tune | residual/adapters only | small held-out geometry train | data efficiency | NICE | TODO | Future real-data path |
| R032 | M6 | full fine-tune reference | all trainable | small held-out geometry train | data efficiency | NICE | TODO | Check adapter value |
| R033 | M7 | extreme scenario generation | teacher simulator extreme handling | large slip / emergency / fishhook | scenario coverage, stability | NICE | TODO | Only after first-version data passes |
| R034 | M7 | single residual on extreme | best G1/G2/G3 non-MoE | extreme handling | large-slip rollout RMSE | NICE | TODO | Establish non-MoE baseline |
| R035 | M7 | MoE tire residual | MoE tire residual | extreme handling | large-slip rollout RMSE, normal-regime regression | NICE | TODO | Keep only if clear benefit |
| R036 | M7 | MoE regression check | best MoE vs selected main | normal dry/wet/low-μ | no degradation on normal regimes | NICE | TODO | Cut MoE if it hurts main scenarios |
