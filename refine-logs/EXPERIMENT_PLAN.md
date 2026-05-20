# EXPERIMENT_PLAN

**日期**：2026-05-20  
**版本**：experiment-only v3  
**目标**：构建一个用于高保真车辆动力学仿真的 hybrid dynamics model。模型先通过大批量多车、多工况数据训练出通用 base model；当需要针对某一台车、某个目标时间段仿真时，只提供固定几何/layout 与最小 nominal prior，再用该时间段少量实车数据 fine-tune 小规模 adapter。

## 1. 文档边界

本文档只承担实验设计职责：定义研究问题、阶段路线、实验系统命名、证据链、实验块、训练协议、指标、运行顺序和风险控制。

具体设计细节放在对应 DESIGN 文档中：

| 内容 | 权威文档 | EXPERIMENT_PLAN 中只保留 |
|---|---|---|
| Teacher simulator 动力学、自由度、内部真值、sanity gate | `refine-logs/TEACHER_SIMULATOR_DESIGN.md` | B0 / B1 / B2 的实验目标和验收标准 |
| 数据集、工况、字段 schema、`fixed_vehicle_context`、`nominal_physics_prior`、teacher-only 字段、split | `refine-logs/DATA_DESIGN.md` | DS0/DS1/DS2 的实验用途和必须支持的划分 |
| Student model、physics backbone、Steering、VehicleParamAdapter、FzResidualNN、MuHead、TireResidualNN、VehicleResidualNN、Uncertainty、模块边界与训练 schedule | `refine-logs/MODULE_DESIGN.md` | 实验中使用的系统编号和 ablation 关系 |
| 具体 run id、状态、优先级 | `refine-logs/EXPERIMENT_TRACKER.md` | 里程碑和运行顺序摘要 |

维护规则：

```text
EXPERIMENT_PLAN.md:
  可以定义“比较哪些系统、用什么数据、看什么指标、成功/失败如何解释”
  不定义网络 hidden size、head 结构、状态向量细节、字段级 schema、积分器实现或 teacher 内部动力学

*_DESIGN.md:
  定义组件接口、输入输出、内部结构、参数化、约束、diagnostics 和实现默认值
```

如果后续修改组件结构，先更新对应 DESIGN 文档；只有当实验配置、ablation 组、指标或运行顺序改变时，才同步修改本文档。

## 2. 研究问题

目标不是做实时控制器，而是做可长时 rollout 的车辆动力学仿真模型。模型应覆盖：

- 干燥、湿滑、雨雪、低附着路面；
- Split-μ；
- 路面 μ transition；
- 制动、加速、转向、制动加转向等组合工况；
- 同车型不同车辆；
- 多车型；
- 目标车辆某个时间段的状态适配，例如载荷、轮胎状态、悬架状态、执行器状态发生变化。

核心假设：

```text
固定几何/layout + 最小 nominal mass/inertia/cg prior
+ 少量目标时间段实车数据 fine-tune
足以让小规模 adapter 适配某台车某段时间的 effective dynamics。
```

这个假设必须通过 FT0-FT6 fine-tune 对照和 fine-tune data efficiency 曲线验证，不能直接当作前提。

## 3. 总体路线

整体分三阶段。

### 3.1 Stage A：Teacher Simulator 阶段

先构建高保真 teacher simulator，用于第一版方案验证。

teacher simulator 的职责：

- 生成多车、多工况数据；
- 导出真实车可观测信号；
- 导出 teacher-only 内部量，用于 auxiliary loss 和 diagnostics；
- 提供可控的参数随机化和 held-out configuration；
- 支持 sim-to-real proxy：train on config A，test/fine-tune on config B。

teacher simulator 的保真度应高于 student model。

### 3.2 Stage B：通用 Base Model 阶段

使用 DS1 V1 Research Dataset 训练通用 hybrid base model，并与 physics-only、black-box baseline 对照。

Stage B 需要完成三件事：

```text
1. 证明 base hybrid 在多车、多配置、多工况上优于 physics-only 和 black-box
2. 通过单因素 ablation 判断 T/F/S/M/V/U 各组件是否值得保留
3. 冻结一个 M8 final single model checkpoint，作为 B6 fine-tune 的统一初始化点
```

固定 base 配置：

```text
Base = T1 + F1 + S1 + M1a + V1 + U0
```

配置编号的含义见第 4 章；组件内部设计见 `MODULE_DESIGN.md`；具体实验块见 B3、B4、B5。

### 3.3 Stage C：目标车 / 目标时间段 Fine-Tune 阶段

针对某台车某个目标时间段，验证少量目标数据是否足以适配 effective dynamics。

Stage C 需要完成：

```text
1. 比较 no fine-tune、小模块 fine-tune 和 full model fine-tune
2. 比较不同目标数据量下的 data efficiency
3. 验证 fine-tune 后在 held-out target windows 上是否稳定，而不是只记住目标训练片段
```

fine-tune 配置使用 `FT0-FT6`，数据量轴使用 `FTD0-FTD5`。编号定义见第 4 章，数据划分和 episode 定义见 `DATA_DESIGN.md`，具体实验块见 B6。

## 4. 实验系统与配置命名

本节只定义实验中需要比较的系统名称。每个系统的具体模块设计见 `MODULE_DESIGN.md`。

### 4.1 Baseline 系统

```text
physics-only:
  只使用 student 物理 backbone，不启用神经 residual / adapter

black-box baseline:
  使用相同 observable 输入训练 TCN/GRU/MLP 直接预测状态或状态增量
  不使用显式车辆/轮胎物理结构
```

black-box baseline 的作用是判断 hybrid route 是否真的优于纯数据驱动模型；它不是最终部署候选。

### 4.2 Base Hybrid 系统

Stage B 固定 base 方案：

```text
Base = T1 + F1 + S1 + M1a + V1 + U0
```

含义：

```text
T1: force-level tire residual with friction ellipse projection
F1: bounded FzResidualNN without mandatory teacher Fz label
S1: S0 steering model + bounded steering residual
M1a: MuHead with μ_mean / μ_logvar output
V1: small bounded VehicleResidualNN on Δx_dot before integration
U0: single-model heteroscedastic uncertainty
```

该 base 是 B3、B4、B5、B6 的共同参考点。B4 中所有组件级 ablation 都以该 base 为基线，每次只替换一个维度。

### 4.3 Stage B Ablation Families

```text
Tire residual family:
  T0, T1, T1-no-proj, T2

Fz residual family:
  F0, F1, F2

Steering family:
  S0, S1

MuHead family:
  M0-fixed, M1a, M1b, M2-oracle

Vehicle residual family:
  V0, V1, V1-large, V2-small

Uncertainty family:
  U0, U1
```

这些编号只表达实验配置差异。每个编号对应的模块接口、约束、head 设计和默认超参见 `MODULE_DESIGN.md`。

### 4.4 Stage C Fine-Tune Families

fine-tune 对照定义如下：

```text
FT0: no fine-tune
FT1: only VehicleParamAdapter trainable
FT2: only MuHead calibration trainable
FT3: only FzResidualNN adapter trainable
FT4: only TireResidualNN adapter trainable
FT5: only steerResidualNN / SteeringHead adapter trainable
FT6: full model fine-tune
```

fine-tune 数据量轴：

```text
FTD0: 0 episode / 0 min, only for FT0
FTD1: 1 episode or about 1-2 min target data
FTD2: 5 episodes or about 5-10 min target data
FTD3: 10 episodes or about 10-20 min target data
FTD4: 30 episodes or about 30-60 min target data
FTD5: 100 episodes or about 2-3 h target data
```

B6 主实验使用：

```text
rows:    FT0, FT1, FT2, FT3, FT4, FT5, FT6
columns: FTD0, FTD1, FTD2, FTD3, FTD4, FTD5
```

其中 `FT0` 只在 `FTD0` 下有意义；`FT1-FT6` 在 `FTD1-FTD5` 下分别训练和评估。

## 5. 实验归因原则

### 5.1 单因素替换

B4 的基本规则：

```text
固定 Base = T1 + F1 + S1 + M1a + V1 + U0
每次只替换一个 family 中的一项
其他模块、训练数据、split、budget、optimizer、rollout horizon 保持一致
每个 ablation 配置都从头训练到 validation loss 最低
```

这样才能把收益归因到具体组件选择，而不是归因到训练预算、数据划分或 checkpoint 差异。

### 5.2 Teacher-Only 使用边界

teacher-only 字段只能用于：

```text
auxiliary loss
oracle upper bound
diagnostics / plotting / sanity check
```

禁止用于：

```text
student input
真实车可部署模型输入
目标车 fine-tune 阶段的必需字段
```

涉及 `F2`、`M2-oracle` 或 teacher label diagnostics 的实验，必须报告这些收益在真实数据阶段不可用时的解释限制。

### 5.3 Ablation 与 Fine-Tune 的 checkpoint 规则

```text
B4 ablation:
  每个配置从头训练

B6 fine-tune:
  所有 FT0-FT6 从同一个 M8 final single model checkpoint 初始化
  FT1-FT5 每次只开放指定模块
  FT6 开放 full model，作为上界和过拟合风险对照
```

### 5.4 组件复杂度升级规则

新增或保留复杂组件必须同时满足：

```text
主指标改善
物理约束不变差
residual / adapter magnitude 可解释
held-out road 或 held-out vehicle/config 不明显退化
相比更简单配置有清晰收益
```

如果复杂组件只在训练集或单一场景上改善，不能进入第一版主线，只能作为后续候选。

## 6. 证据链

| 编号 | 需要证明的结论 | 为什么重要 | 最小可信证据 | 对应实验 |
|---|---|---|---|---|
| C0 | teacher simulator 与数据 schema 可信 | 数据生成器不可信时后续训练无意义 | `DATA_DESIGN.md` 中 DS0 Debug Dataset 和 DS1 V1 Research Dataset 可稳定生成；observable 与 teacher-only 严格隔离；物理 sanity 通过 | B0, B1 |
| C1 | Base hybrid model：`T1 + F1 + S1 + M1a + V1 + U0` 优于 physics-only 和 black-box | 证明 hybrid 路线值得做 | 在 DS1 多车/多配置/多工况数据上，base 在 1s/5s/10s rollout 上优于 physics-only；长 rollout/约束表现优于 black-box | B3 |
| C2 | 关键组件设计选择有必要且可归因 | 防止无依据堆模块，并决定最终模型复杂度 | T0/T1/T1-no-proj/T2、F0/F1/F2、S0/S1、M0-fixed/M1a/M1b/M2-oracle、V0/V1/V1-large/V2-small、U0/U1 均完成单项 ablation；收益、失败、uncertainty calibration 和 residual/constraint 变化可解释 | B4 |
| C3 | 目标车/目标时间段可通过指定模块 fine-tune 适配 | 支撑真实车落地 | `FT1-FT5 × fine-tune data buckets FTD1-FTD5` 中至少有小模块 fine-tune 明显优于 `FT0@FTD0`，并在 held-out target windows 上接近或稳定优于 FT6 | B6 |
| C4 | base model 具备跨车辆/配置泛化能力 | 支撑“多车 base + 少量新车数据适配”的核心假设 | held-out vehicle/config 上，base 优于 physics-only 和 black-box；泛化退化可由 B6 作为 supplementary evidence 检查 fine-tune 修复能力 | B3, B5 |
| C5 | 极限操控下 MoE Tire Residual 是否值得进入后续版本 | 决定 DS2 阶段是否扩展模型复杂度 | 在 DS2 extreme handling 上，MoE tire residual 相比单专家 tire residual 改善 large-slip / emergency maneuver rollout，且不伤害常规工况 | B7 |

## 7. 实验块

### B0：Teacher Simulator 构建

- **目的**：得到可信的高保真数据生成器。
- **设计文档**：`refine-logs/TEACHER_SIMULATOR_DESIGN.md`。
- **必须完成**：
  - 车辆、轮胎、路面、执行器、传感器模块可运行；
  - observable 与 teacher-only 字段导出完整；
  - 每个字段有 role metadata；
  - 支持参数随机化和 held-out config。
  - 按 `DATA_DESIGN.md` 生成 DS0 Debug Dataset；
  - 按 `DATA_DESIGN.md` 生成 DS1 V1 Research Dataset 的最小版本，至少包含多车/多配置、多工况和 held-out split metadata。
- **成功标准**：
  - 所有 required scenario 可以 deterministic 复现，seed / teacher version / hidden parameter hash 可追踪；
  - observable 与 teacher-only role metadata 完整，导出 schema 与 `DATA_DESIGN.md` 一致；
  - 所有 DS0 episode 无 NaN / Inf，长 episode 不出现数值爆炸；
  - DS0 必做工况能稳定生成；
  - DS1 多车/多配置数据能稳定生成，metadata 完整；
  - 制动时前轴 `Fz` 增加；
  - 转弯时外侧轮 `Fz` 增加；
  - Split-μ 制动产生合理 yaw moment；
  - tire force 大体满足 friction ellipse。

### B1：数据与物理 Sanity

- **目的**：排除坐标系、单位、时间对齐、字段泄漏和物理符号错误。
- **数据**：
  - 先在 DS0 Debug Dataset 上完成全部 sanity；
  - DS0 通过后，再在 DS1 V1 Research Dataset 的抽样子集上复查 schema、metadata、held-out split 和物理 sanity。
- **检查项**：
  - schema / teacher-only leakage；
  - `dt` 与时间戳；
  - slip ratio / slip angle 符号；
  - wheel speed 与 `vx` 一致性；
  - torque 与 wheel angular acceleration 方向；
  - physics-only rollout smoke test；
  - zero-input static test；
  - left/right symmetry test；
  - braking monotonicity test；
  - `Fz_i >= 0` 与 `ΣFz_i` 检查；
  - friction ellipse sanity；
  - tiny black-box 与 tiny base overfit 5-10 条短序列。
- **成功标准**：
  - schema / role metadata / wheel_order / unit 检查全部通过；
  - teacher-only 字段不会进入 student input，dataloader leakage test 必须显式通过；
  - `dt`、时间戳、输入/状态对齐检查全部通过；
  - slip ratio / slip angle / steering / torque / wheel acceleration 符号检查全部通过；
  - `Fz_i >= 0`、`ΣFz_i` 合理范围、friction ellipse sanity 全部通过；
  - zero-input static、left/right symmetry、braking monotonicity 全部通过；
  - physics-only rollout smoke test 不立即发散；
  - tiny black-box 与 tiny base 能 overfit 5-10 条短序列；
  - train/val/test、held-out road μ、held-out vehicle/config、held-out target time windows 无 episode/window 重叠。

### B2：Sim-to-Real Proxy Stress Test

- **目的**：在没有实车数据前，先验证 teacher simulator 能生成可复现、可追踪的分布扰动 target windows，为 B6 fine-tune 和后续 sim-to-real 分析提供 proxy 数据。
- **数据**：
  - 基于 DS1 生成扰动版 target windows；
  - 对真实 mass/inertia/cg、suspension/tire 参数、sensor bias、actuator delay 加入 5%-15% 的系统偏移；
  - 扰动参数仍作为 teacher-only，不进入 student input。
- **检查项**：
  - perturbation profile 可复现；
  - target train / validation / test windows 互不重叠；
  - perturbation metadata 完整，但不进入 student input；
  - 扰动前后 observable distribution 有可测差异；
  - physics sanity 不因扰动直接崩溃。
- **成功标准**：
  - 能为每个 held-out vehicle/config 生成完整 perturbation target windows；
  - 扰动 metadata、teacher-only 字段和 student input 隔离全部通过；
  - 5%-15% 系统偏移能造成可测但不致命的 dynamics distribution shift；
  - 扰动数据通过 B1 的 schema、时间对齐、符号和基础物理 sanity；
  - B6 将在该 proxy 数据上评估 `FT0@FTD0`、小模块 fine-tune 和 `FT6` 的恢复能力。

### B3：Base Hybrid 闭环

- **目的**：验证固定 base 方案是否优于 physics-only 和 black-box。
- **数据**：
  - 使用 DS1 V1 Research Dataset；
  - 训练集包含多车/多配置/多工况；
  - 测试集必须单独报告 seen vehicle/config、held-out road μ、held-out vehicle/config。
- **系统**：
  - physics-only baseline；
  - black-box TCN/GRU/MLP baseline；
  - `Base = T1 + F1 + S1 + M1a + V1 + U0`。
- **base 组成**：
  - `T1`：`ΔFx/ΔFy` tire residual + friction ellipse projection；
  - `F1`：bounded `FzResidualNN`；
  - `S1`：`S0 + steerResidualNN`；
  - `M1a`：`MuHead` 输出 `μ_mean/μ_logvar`；
  - `V1`：小容量 bounded `Δx_dot` vehicle residual；
  - `U0`：single model heteroscedastic uncertainty。
- **设置**：
  - B3 不做多项 ablation，只验证 base 闭环可训练、可 rollout、优于 baseline；
  - 所有单项替换放到 B4。
- **成功标准**：
  - 在 seen vehicle/config、held-out road μ、held-out vehicle/config 上分别报告 `1s/5s/10s` rollout RMSE；
  - base 在主要状态量 `vx/vy/r/roll/pitch/omega_i` 的 `1s/5s/10s` rollout 上整体优于 physics-only；
  - base 在 `10s` rollout stability、constraint violation 和 yaw drift 上优于或不差于 black-box；
  - base 的 physical constraint violation 不高于 physics-only 的合理范围；
  - residual magnitude、residual / physics ratio 和 smoothness 受控；
  - held-out vehicle/config 上允许退化，但必须单独报告，不能用 seen performance 掩盖；
  - 关键结果至少 3 seeds；预算不足时先 1 seed 筛选，再对 B3 主表补齐 3 seeds。

### B4：组件级 Ablation Suite

- **目的**：验证关键组件设计选择是否必要、收益是否可归因，并决定第一版最终单模型结构。
- **统一协议**：
  - 使用 DS1 V1 Research Dataset 的同一 train/val/test split；
  - 每个配置从头训练到 validation loss 最低状态；
  - 每次只替换 base 中的一项，其余模块保持 `T1 + F1 + S1 + M1a + V1 + U0`；
  - 报告 rollout、物理约束、residual magnitude、smoothness、held-out road μ 和 held-out vehicle/config 指标。
  - “优于 / 明显改善 / 不退化”必须在同一 test split、同一训练预算、同一 rollout horizon 下判断；关键对照至少补 3 seeds，预算不足时先 1 seed 筛选。
  - 若主指标改善但物理约束、residual magnitude、held-out 泛化显著变差，则不能作为第一版保留组件。

#### B4.1：`TireResidualNN` Ablation

- **目的**：以 base 的 `T1` 为中心，验证 tire residual 是否应从 global residual 中拆出、force-level residual 是否优于 parameter-level residual，以及 friction ellipse projection 是否必要。
- **数据**：
  - 使用 DS1 V1 Research Dataset 的同一 train/val/test split；
  - 必须包含 low-μ、Split-μ、transition 和 combined slip 场景。
- **系统**：
  - `T0`：把 base 中的 `T1` 替换为无 tire residual；
  - `T1`：base 默认，`TirePhysics + ΔFx/ΔFy TireResidualNN + friction ellipse projection + V1 VehicleResidualNN`；
  - `T1-no-proj`：把 base 中的 `T1` 替换为不做 friction ellipse projection 的 force residual。
  - `T2`：把 base 中的 `T1` 替换为 parameter-level residual，输出 `ΔC_alpha/ΔC_kappa/Δμ_scale`。
- **成功标准**：
  - T1 在 `vy/r/omega_i` 上优于 T0；
  - `VehicleResidualNN` 幅值下降；
  - friction ellipse violation 接近 0；
  - T1 vs T1-no-proj 能分离 projection 的贡献。
  - 如果 T2 接近 T1 且更平滑、更稳定，则优先 T2；
  - 如果 T2 在 low-μ / large slip 明显弱于 T1，则保留为可解释对照。

#### B4.2：`FzResidualNN` Ablation

- **目的**：验证显式修正 `Fz_i` 是否必要，以及 teacher `Fz_true_i` 辅助监督是否有价值。
- **对照系统**：
  - `F0`：无 `FzResidualNN`，只用 physics `Fz`；
  - `F1`：bounded `FzResidualNN`，只由 rollout loss / constraint loss 端到端训练；
  - `F2`：`F1 + Fz_true_i` teacher auxiliary loss，仅仿真阶段可用。
- **固定条件**：
  - 其余模块保持 base：`T1 + S1 + M1a + V1 + U0`；
  - `TireResidualNN`、`MuHead`、`VehicleResidualNN` 容量不变；
  - 训练数据必须包含 braking、turning、combined brake+steer、Split-μ。
- **指标**：
  - `Fz_i` RMSE，如果有 `Fz_true_i`；
  - `ΣFz_i - m_eff*g` error；
  - `Fz_i < 0` rate；
  - 制动前轴载荷转移方向准确率；
  - 转弯外侧载荷转移方向准确率；
  - downstream rollout RMSE：`vy/r/roll/pitch/omega_i`；
  - tire friction ellipse violation；
  - `VehicleResidualNN` magnitude 是否下降。
- **成功标准**：
  - `F1` 相比 `F0` 在载荷转移工况和 downstream rollout 上稳定改善；
  - `F2` 可以作为仿真辅助训练保留，但必须报告真实数据阶段不可用 `Fz_true_i` 时的退化；
  - 如果 `F2 >> F1` 且 `F1` 无明显收益，说明该模块过度依赖 teacher-only label，不能作为真实数据主线。
- **淘汰标准**：
  - `F1` 不改善 downstream rollout；
  - 或 `F1` 造成更高 constraint violation；
  - 或 `FzResidualNN` 输出大幅高频抖动。

#### B4.3：Steering Ablation

- **目的**：验证 base 中的 `S1` 是否确实优于简单 `S0` 一阶转向滞后。
- **对照系统**：
  - `S0`：一阶 steering lag；
  - `S1`：`S0 + steerResidualNN`。
- **固定条件**：
  - 其余模块保持 base：`T1 + F1 + M1a + V1 + U0`；
  - 重点使用 step steer、sine steer、constant radius、low-μ turning、brake+steer 数据。
- **指标**：
  - `delta_eff` 误差，如果 teacher/export 可用；
  - yaw rate `r` rollout RMSE；
  - lateral velocity `vy` RMSE；
  - path/yaw drift；
  - `S1` residual magnitude；
  - `S1` residual smoothness；
  - low-μ turning 下是否引入不稳定。
- **成功标准**：
  - `S1` 在转向动态强相关场景显著改善 `r/vy/yaw drift`；
  - `S1` residual 保持 bounded、smooth；
  - 常规 dry/wet 场景无退化。
- **淘汰标准**：
  - `S1` 只在训练集改善，held-out steering 场景不改善；
  - 或 residual 大到替代 steering physics；
  - 或 low-μ 下引入 rollout instability。

#### B4.4：`MuHead` Ablation

- **目的**：验证 base 中的 `M1a-MuHead` 是否真正提供路面/轮胎有效附着信息，并比较两种可部署输出设计，而不是让其与 mass/Fz/tire residual 混淆。
- **对照系统**：
  - `M0-fixed`：固定 nominal μ 或按场景给 coarse prior；
  - `M1a`：使用 `MuHead` 输出 `μ_mean/μ_logvar`，base 默认；
  - `M1b`：使用 `MuHead` 输出 `μ_scale/confidence`；
  - `M2-oracle`：使用 teacher-only `μ_true_i`，只作为上界，不作为可部署模型。
- **固定条件**：
  - 其余模块保持 base：`T1 + F1 + S1 + V1 + U0`；
  - 必须分 slip level 统计：small slip、medium slip、large slip；
  - 必须包含 low-μ、Split-μ、transition 数据。
- **指标**：
  - rollout RMSE：`vy/r/omega_i`；
  - `μ` RMSE / NLL / calibration，如果有 `μ_true_i`；
  - small-slip 区 confidence 是否降低；
  - μ transition 响应延迟；
  - friction ellipse violation；
  - `TireResidualNN` magnitude 是否下降。
- **成功标准**：
  - `M1a` 或 `M1b` 在 low-μ / Split-μ / transition 上优于 `M0-fixed`；
  - `M1a`、`M1b` 与 `M2-oracle` 的差距可解释；
  - small-slip 区不会输出虚假的高置信绝对 μ；
  - `MuHead` 改善不是由 `VehicleResidualNN` 吞掉误差造成。
- **淘汰或降级标准**：
  - `M1a/M1b` 与 `M0-fixed` 差距很小；
  - 或 `M1a/M1b` 在 small-slip 区 calibration 很差；
  - 或 `M1a/M1b` 导致 tire residual / vehicle residual 更大。

#### B4.5：Uncertainty / Ensemble Ablation

- **目的**：验证单模型不确定性和 K=3 ensemble 是否提供可靠 uncertainty，而不是只提高平均性能。
- **对照系统**：
  - `U0`：single model heteroscedastic uncertainty，输出 prediction mean 与 log variance；
  - `U1`：K=3 deep ensemble，使用相同结构、不同 seed / bootstrap split；总不确定性 = aleatoric 平均值 + ensemble mean 方差。
- **固定条件**：
  - 其余模块保持 base：`T1 + F1 + S1 + M1a + V1`；
  - ensemble 成员训练预算和数据划分记录清楚；
  - 不把 ensemble 作为弥补模型结构错误的主线。
- **指标**：
  - NLL；
  - coverage；
  - sharpness；
  - calibration error；
  - error-uncertainty correlation；
  - OOD / held-out config detection AUC；
  - OOD road / held-out vehicle config 上的不确定性上升是否合理；
  - 推理成本。
- **成功标准**：
  - `U1` 在 NLL、coverage、calibration、OOD AUC、error-uncertainty correlation 中至少两类指标明显优于 `U0`；
  - `U1` 的 rollout RMSE 和物理约束不退化；
  - OOD / held-out config 中 uncertainty 有可解释上升；
  - `U1` 的推理/训练成本相对收益可接受；
  - 如果只改善 RMSE、不改善 uncertainty 指标，则不能作为 uncertainty 贡献。
- **淘汰标准**：
  - K=3 ensemble 只带来小幅 RMSE 提升但不改善 uncertainty；
  - 推理/训练成本不符合后续使用目标。

#### B4.6：`VehicleResidualNN` 容量与位置 Ablation

- **目的**：防止 base 中的 `V1` 退化成黑箱主模型，并验证 `V1` 是否优于 `V0/V2`。
- **对照系统**：
  - `V0`：无 `VehicleResidualNN`；
  - `V1`：base 默认，小容量 bounded `Δx_dot` before integration；
  - `V1-large`：较大容量 `Δx_dot` residual，用于检测 residual 是否吞掉 physics；
  - `V2-small`：小容量 `Δx` after integration。
- **固定条件**：
  - 其余模块保持 base：`T1 + F1 + S1 + M1a + U0`；
  - 不改变 physics backbone；
  - 相同 rollout horizon。
- **指标**：
  - rollout RMSE；
  - residual magnitude / total acceleration ratio；
  - residual spectrum；
  - physical constraint violation；
  - closed-loop stability；
  - held-out road / held-out vehicle config 泛化。
- **成功标准**：
  - `V1` 在改善误差的同时 residual magnitude 受控；
  - `V1-large` 如果明显更好但 residual 过大，说明模型有退化风险，不能作为主线；
  - `V2-small` 只在稳定性明显更好且物理约束不变差时作为备选。
- **淘汰标准**：
  - `VehicleResidualNN` 对总加速度贡献过大；
  - held-out config 泛化变差；
  - constraint violation 上升。

#### B4.7：Ablation Suite 执行顺序与退出条件

推荐顺序：

1. `F0/F1/F2`：先确认 `Fz` 工作点是否可靠；
2. `M0-fixed/M1a/M1b/M2-oracle`：再确认 `MuHead` 是否有效，并比较两种输出设计；
3. `T0/T1/T1-no-proj/T2`：验证 tire residual 的组织方式和 projection 贡献；
4. `S0/S1`：验证 steering residual；
5. `V0/V1/V1-large/V2-small`：确认 final residual 不退化；
6. `U0/U1`：最后做 ensemble，因为成本最高，且依赖单模型稳定。
- **退出条件**：
  - 每个增强模块都能证明收益，或被明确淘汰；
  - 所有保留模块在 rollout、约束、residual magnitude 和 held-out 泛化中没有明显副作用；
  - F2 的收益不能被解释为真实数据阶段不可用 teacher label 依赖；
  - ensemble 要提升 NLL、coverage、sharpness、OOD AUC 或 error-uncertainty correlation，而不仅是 RMSE；
  - B4 结束后必须给出一个明确的 final single model configuration，供 B5 / B6 使用。

### B5：Cross-Vehicle / Cross-Config 泛化验证

- **目的**：验证多车 base model 是否真的学到跨车辆/配置的通用 dynamics，而不是只在 seen vehicle/config 上拟合得好。
- **数据**：
  - 使用 DS1 V1 Research Dataset；
  - 训练集使用 seen vehicle/config families；
  - 测试集拆分报告 seen vehicle/config、held-out road μ、held-out vehicle/config、held-out vehicle + held-out road μ。
- **系统**：
  - physics-only baseline；
  - black-box baseline；
  - `Base = T1 + F1 + S1 + M1a + V1 + U0`；
  - B4 后冻结的 final single model。
- **指标**：
  - 1s / 5s / 10s rollout RMSE；
  - closed-loop stability；
  - physical constraint violation；
  - residual magnitude / total acceleration ratio；
  - uncertainty 是否在 held-out config 上合理上升。
- **成功标准**：
  - final single model 在 held-out vehicle/config 上优于 physics-only 和 black-box；
  - held-out vehicle/config 上 `1s/5s/10s` rollout RMSE、stability、constraint violation 分开报告；
  - held-out config 上 residual magnitude / total acceleration ratio 不明显失控；
  - uncertainty 在 held-out config 上有合理上升，且与误差有正相关；
  - seen config 与 held-out config 的性能 gap 必须量化。
- **失败解释**：
  - 如果 seen config 表现好但 held-out config 明显失败，说明多车 base 泛化不足，需扩大 DS1 vehicle/config 覆盖或调整 `VehicleParamAdapter`。
  - 如果 B6 能用少量 target data 修复该退化，说明 adaptation pipeline 有价值，但不能把它计为 B5 base generalization 成功。

### B6：Target Vehicle / Target Time-Window Fine-Tune

- **目的**：验证新车/目标时间段能否通过小 adapter 适配。
- **数据**：
  - 第一阶段无真实数据时，用 teacher simulator 构造 proxy；
  - 主实验使用 M8 冻结的 final single model checkpoint，即 B4 组件选择和 B5 泛化验证后的模型；
  - 可选补充实验使用原始 Stage B base checkpoint，用于判断 B4/B5 结构选择对 fine-tune 的传递影响；
  - test/fine-tune 使用 DS1 中 held-out vehicle/config/time-window；
  - 后续有实车数据时，用目标车目标时间段少量数据 fine-tune。
- **数据量对照**：
  - 使用 Stage C 定义的 `FTD0-FTD5` 数据量轴；
  - 每个目标车/目标时间段使用 `{0, 1, 5, 10, 30, 100}` 条 episode 或等价时长片段，episode 定义见 `DATA_DESIGN.md`；
  - fine-tune 数据和测试数据必须来自不同 target windows；
  - 所有 FT 方案从同一个 M8 final single model checkpoint 初始化。
- **fine-tune 对照**：
  - 使用 3.3 Stage C 中定义的 `FT0-FT6`；
  - `FT1-FT5` 每次只开放一个指定模块；
  - `FT6` 作为 full model fine-tune 上界和过拟合风险对照。
- **实验矩阵**：
  - `FT0@FTD0`：无目标数据，直接测试 base；
  - `FT1-FT6 @ fine-tune data buckets FTD1-FTD5`：每个 FT 方案在每个数据量下单独 fine-tune；
  - 每个格子至少覆盖多个 held-out target vehicles/configs/time windows；
  - 预算不足时先跑 `FTD1/FTD3/FTD5`，确认趋势后补齐 `FTD2/FTD4`；
  - 所有格子报告相同 test set 上的 rollout、物理约束和 adapter stability。
- **成功标准**：
  - 至少一个 `FT1-FT5` 小模块方案在 held-out target windows 上相对 `FT0@FTD0` 明显改善；
  - “明显改善”必须在主指标、物理约束和 adapter/residual stability 中同时成立，不能只改善 train window；
  - 小模块 FT 在少量数据下接近 FT6，或比 FT6 在 held-out windows 上更稳；
  - “接近 FT6”需要报告小模块 FT 与 FT6 的 gap，并说明 gap 是否低于预设容忍比例；
  - 数据效率曲线随 fine-tune 数据量合理改善，并能识别收益饱和点；
  - 报告达到固定误差阈值所需的最小 target data；
  - 若 B2 proxy perturbation 数据可用，必须报告 `FT0@FTD0`、最佳小模块 FT 和 `FT6` 对 perturbation 退化的恢复能力；
  - 如果只有 FT6 有效，说明当前 adapter 分工或 base 泛化能力不足。

### B7：极限操控与 MoE Tire Residual

- **状态**：后续阶段。
- **触发条件**：
  - T0/T1/T2 稳定；
  - low-μ / Split-μ 已验证；
  - DS2 Expanded Dataset 中加入 large-slip / fishhook / emergency maneuver 数据。
- **原则**：
  - MoE 只在极限操控明显提升且不伤害常规工况时保留；
  - 不作为第一版主线。
- **指标**：
  - large-slip / emergency maneuver rollout RMSE；
  - yaw/side-slip drift；
  - closed-loop stability；
  - friction ellipse violation；
  - 常规工况回归测试性能。
- **成功标准**：
  - MoE tire residual 在 DS2 extreme handling 上优于同等训练预算、相近容量的 single-expert T1/T2；
  - large-slip / emergency maneuver 的 rollout RMSE、yaw/side-slip drift 或 closed-loop stability 至少有明确改善；
  - 常规 DS1 test set 无明显退化；
  - friction ellipse violation 不上升；
  - gating / expert usage 行为可解释，不能只是更大黑箱；
  - 如果只在少数极端场景提升且常规场景退化，则不进入主线。

## 8. 数据设计

详细数据集设计、工况矩阵、字段 schema、metadata、划分规则和质量检查放在 `refine-logs/DATA_DESIGN.md`。本节只保留实验计划层面的数据目标。

### 8.1 数据集阶段

```text
DS0 Debug Dataset:
  单车型 / 少量工况
  用于 teacher simulator、schema、单位、符号和物理 sanity

DS1 V1 Research Dataset:
  第一版正式研究数据
  多车 / 多配置 / 多工况
  用于 base model 训练、held-out vehicle/config、sim-to-real proxy、FT0-FT6 和 FTD0-FTD5 fine-tune 数据量对照

DS2 Expanded Dataset:
  更多车型、更多载荷/轮胎状态、极限操控
  用于 MoE tire residual 和大范围泛化验证
```

### 8.2 核心设计要求

```text
第一版正式数据集必须包含多车/多配置，不再把单车型作为 full dataset
observable 与 teacher_only 字段必须 schema 级隔离
student input 只允许 observable + fixed_vehicle_context + nominal_physics_prior
teacher_only 字段只允许用于 loss / diagnostics
所有划分按完整 episode / scenario / target window 完成
禁止把同一条 rollout 的相邻片段同时放入 train/test
工况空间生成全覆盖，但 DS1 training、balanced test、deployment-weighted test 和 B6 fine-tune 必须使用不同采样比例
```

### 8.3 必须支持的划分

```text
train / validation / test episodes
held-out road μ episodes
held-out vehicle/config episodes
held-out target time windows
FT0-FT6 × FTD0-FTD5 fine-tune data amount splits
```

### 8.4 工况覆盖摘要

```text
基础稳态
低附着 dry/wet/snow/ice
Split-μ
路面 μ transition
patchy μ / wheel-level μ map
加速、制动、转向、制动加转向
执行器 delay / saturation
传感器 noise / bias / filtering
```

## 9. Loss 与训练策略

### 9.1 总 Loss

ResidualNN 默认是端到端 hybrid model 的子模块，不是独立完整模型。

```text
L_total =
  L_rollout
+ λ_one_step L_one_step
+ λ_constraint L_constraint
+ λ_residual L_residual_magnitude
+ λ_smooth L_smoothness
+ λ_teacher L_teacher_aux
```

其中：

```text
L_rollout:
  多步状态预测误差

L_constraint:
  Fz >= 0
  friction ellipse
  force balance

L_residual_magnitude:
  防止 residual 吞掉 physics

L_smoothness:
  防止 residual 高频抖动

L_teacher_aux:
  只在仿真阶段使用
```

### 9.2 Ablation 训练协议

本文档中的 ablation 指不同配置各自完整训练，而不是在同一个 checkpoint 上临时关闭或替换模块。

```text
每个 ablation 配置：
  从头初始化
  使用相同 train/val/test split
  使用相同 optimizer、batch、rollout horizon、训练预算和 early stopping 规则
  用 validation rollout loss 选择 best checkpoint
  在固定 test episodes 上报告结果
  至少 3 seeds，预算不足时先 1 seed 筛选再补关键对照
```

唯一例外是 B6 fine-tune：所有 FT0-FT6 都从同一个 M8 final single model checkpoint 初始化，只开放指定模块。

### 9.3 Fine-Tune 策略

目标车/目标时间段 fine-tune 遵循以下规则：

```text
所有 FT0-FT6 从同一个 M8 final single model checkpoint 初始化
FT1-FT5 每次只开放 Stage C 定义的一个指定模块
FT6 开放 full model，作为上界和过拟合风险对照
fine-tune train window / validation window / test window 必须互不重叠
用 validation target window early stopping
记录 adapter output drift、residual magnitude drift 和 held-out window 性能
```

具体 FT 对照以 3.3 Stage C 为准：FT1-FT5 每次只开放一个指定模块，FT6 才开放全模型。

## 10. 指标

### 10.1 主指标

```text
1s / 5s / 10s rollout RMSE
closed-loop rollout stability
vx, vy, r
roll, pitch, p, q
omega_fl/fr/rl/rr
yaw drift
```

### 10.2 物理约束指标

```text
Fz_i < 0 rate
ΣFz_i - m_eff*g error
friction ellipse violation rate
residual magnitude
residual smoothness
energy / acceleration sanity
```

### 10.3 不确定性指标

```text
NLL
coverage
sharpness
calibration error
error-uncertainty correlation
OOD / held-out config detection AUC
```

### 10.4 适配指标

```text
fine-tune data efficiency
held-out window RMSE
adapter parameter stability
FT1-FT5 vs FT0@FTD0 improvement
FT1-FT5 vs FT6 gap
不同 fine-tune 数据量 FTD0-FTD5 下的性能曲线
达到固定误差阈值所需的最小 target data
fine-tune 数据量增加后的收益饱和点
```

### 10.5 极限操控指标

```text
large-slip rollout RMSE
emergency maneuver yaw/side-slip drift
closed-loop stability under fishhook / lane change
friction ellipse violation under large slip
DS1 常规工况回归测试性能
MoE vs single-expert tire residual gap
```

## 11. 运行顺序

1. 执行 B0：构建 high-fidelity teacher simulator。
2. 生成第一版多车、多工况数据集。
3. 做 B1：schema、teacher-only leakage、时间对齐、物理符号 sanity。
4. 做 B2：sim-to-real proxy stress test。
5. 实现 physics-only baseline。
6. 实现 black-box baseline。
7. 实现 base：`T1 + F1 + S1 + M1a + V1 + U0`。
8. 执行 B4 组件级 ablation：`T0/T1/T1-no-proj/T2`、`F0/F1/F2`、`S0/S1`、`M0-fixed/M1a/M1b/M2-oracle`、`V0/V1/V1-large/V2-small`、`U0/U1`。
9. 执行 B5：cross-vehicle / cross-config 泛化验证。
10. 根据 ablation 和泛化结果冻结第一版单模型结构。
11. 执行 B6：FT0-FT6 × FTD0-FTD5 fine-tune data efficiency。
12. 如果 B4.5 证明 U1 有价值，再训练最终 K=3 ensemble；否则保留 U0。
13. 执行 B7：DS2 extreme handling + MoE tire residual。

## 12. 里程碑

| Milestone | 目标 | 核心 Runs | 决策门槛 |
|---|---|---|---|
| M0 | teacher simulator v0 | R000-R000d | 所需场景和导出字段可用 |
| M1 | 数据工况矩阵 | R000e-R000h | 多车、多工况数据集生成，metadata 干净 |
| M2 | 数据/物理 sanity | R001-R004b | schema、时间、符号、physics-only smoke test 通过 |
| M3 | Sim-to-real proxy | R004c-R004e | 分布扰动下 fine-tune 恢复能力可解释 |
| M4 | baseline | R005-R008 | physics-only 与 black-box 可训练可评估 |
| M5 | Base | R009-R014 | base 优于 physics-only 和 black-box，residual 受控 |
| M6 | Component ablation suite | R015-R032 | T/F/S/M/V/U 单项 ablation 完成，第一版组件选择明确 |
| M7 | Cross-vehicle/config generalization | R033-R036 | held-out vehicle/config 泛化结果明确 |
| M8 | Final single model | R037 | 第一版单模型结构冻结 |
| M9 | B6 adaptation | R038-R045 | FT0-FT6 × FTD0-FTD5 数据效率曲线完成 |
| M10 | B7 extreme + MoE | R046+ | large-slip 明显收益且不伤害常规工况 |
| M11 | Real vehicle validation | TBD | 有实车数据后验证 sim-to-real gap 和少量数据 fine-tune 假设 |

## 13. 风险与缓解

- **teacher simulator 不可信**
  - 先验证 tire force、Fz、load transfer、road μ map、actuator lag、sensor noise。
- **teacher-only 字段泄漏**
  - schema 级隔离，dataloader 单元测试显式断言。
- **Fz / μ / tire residual 相互混淆**
  - 先做 Fz accuracy budget，再做 tire residual；记录模块归因指标。
- **MuHead 在线性区不可辨识**
  - 使用 slip-dependent confidence；增加 μ oracle 和 fixed μ 对照。
- **VehicleParamAdapter 与 VehicleResidualNN 抢解释权**
  - adapter 输出慢变量；`VehicleResidualNN` 小容量 bounded；做 FT1-FT6。
- **只给 mass/I/cg nominal prior 不足**
  - 在 held-out vehicle/config 上观察 FT0-FT6 和数据效率；如果小模块 FT 均无效，再考虑增加可获得的 nominal prior。
- **单项 ablation 不可归因**
  - 固定 `Base = T1 + F1 + S1 + M1a + V1 + U0`，每次只替换一个模块；共享同一个 base tire model、训练数据和模型容量，并从头训练各配置。
- **black-box baseline 不公平**
  - 保证输入信息一致；报告参数量、训练预算和 rollout horizon。
- **sim-to-real gap 未验证**
  - 第一版用 B2 分布扰动作为 proxy；有实车数据后必须增加 M11 real vehicle validation，不能只用仿真结果声称真实车落地。

## 14. 最终 Checklist

- [x] 计划是独立自包含版本。
- [x] 目标是多车 base model + 目标车/目标时间段 fine-tune。
- [x] `fixed_vehicle_context` 已在 `DATA_DESIGN.md` 中定义。
- [x] `nominal_physics_prior` 已在 `DATA_DESIGN.md` 中限定为 `mass/I/cg`。
- [x] teacher-only 字段已在 `DATA_DESIGN.md` / `TEACHER_SIMULATOR_DESIGN.md` 中定义，且禁止作为 student input。
- [x] `FT0-FT6` 已在 Stage C 中定义。
- [x] `Base = T1 + F1 + S1 + M1a + V1 + U0` 已定义。
- [x] 单项 ablation 规则已定义。
- [x] B4 组件级 ablation suite 已定义。
- [x] B6 adaptation 对照已定义。
- [x] Dataset stage `DS0-DS2` 与 fine-tune data bucket `FTD0-FTD5` 已分离命名。
- [x] B2 sim-to-real proxy stress test 已定义。
- [x] `VehicleParamAdapter` 与 `VehicleResidualNN` 分工已在 `MODULE_DESIGN.md` 中定义。
- [ ] teacher simulator v0 已实现。
- [ ] 第一版数据集已生成。
- [ ] B1 sanity 已通过。
- [ ] physics-only baseline 已实现。
- [ ] black-box baseline 已实现。
