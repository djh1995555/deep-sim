# 车辆高保真动力学深度学习模型：Phase 1 文献调研与范围检查

**日期**：2026-05-18  
**输入 brief**：`/home/mi/vibe/research/deep_sim/requirement.md`  
**Pipeline 阶段**：`idea-discovery / Phase 1: research-lit`  
**AUTO_PROCEED**：false，本文件用于人工检查点，确认后再进入 idea generation。

## 0. 任务重述

目标不是发表论文，而是工程落地：训练一个用于仿真系统的车辆高保真动力学模型，能够在干燥、湿滑、雨雪、低附着、Split-μ 和路面衔接场景下做长时多步 rollout 预测。第一阶段只允许使用真实车上可观测信号，训练数据可由超高自由度仿真器生成，但训练模型的输入不得使用真实车不可获得的内部变量。

因此文献调研不应只找“车辆动力学 + 深度学习”，还必须覆盖：

- 低附着和 tire-road friction coefficient (TRFC) 估计；
- 四轮异质路面、Split-μ、路面突变和 μ transition；
- 轮胎小侧偏线性区与大侧偏非线性区；
- 仅从历史状态中隐式辨识路面和不可辨识因素；
- 物理约束、混合物理-数据模型、residual learning；
- rollout 长期稳定性和不确定性估计。

## 1. 文献 landscape 总结

### 1.1 车辆动力学学习模型

这一类文献通常面向自动驾驶、自动赛车或控制，目标是学习车辆模型供 MPC 或规划使用。典型路线包括 neural network dynamics、Gaussian process residual dynamics、deep kernel learning、neural ODE / latent dynamics、hybrid physics + learned residual。

对本项目的启发：

- 纯黑盒神经网络容易在训练分布外和长 rollout 下漂移；
- 物理 backbone + learned residual 是工程上更稳的首选；
- 模型输入应保留显式物理量，例如 `vx, vy, r, omega_i, torque_i, brake_i, steering`，不要过早压缩成端到端 latent；
- 自动赛车文献很重视极限操控，但很多方法默认路面近似固定，不足以直接解决湿滑、Split-μ 和路面衔接。

### 1.2 轮胎-路面摩擦与低附着估计

TRFC 估计文献比“学习动力学模型”更直接对应本项目难点。它们把 μ 看作隐变量，通过轮速、纵横向加速度、横摆率、驱动/制动输入、轮胎模型或视觉估计路面附着。

关键结论：

- 低激励常规工况下 μ 可辨识性很弱，很多文献都依赖制动/转向激励、轮胎滑移或视觉先验；
- 只用车辆状态历史能估计“等效可解释 μ 区间”，但不能保证唯一辨识真实路面；
- Split-μ 需要至少支持四轮级别或左右轮级别 latent friction，而不是单一全车 μ；
- 路面衔接处需要动态 latent state，而不是每帧独立分类。

### 1.3 轮胎力和智能轮胎/状态估计

这类文献关注 tire forces、normal load、sideslip angle、wheel force transducer 替代、intelligent tire。虽然部分依赖真实车难以获得的轮胎应变或智能轮胎数据，但方法上有价值：

- tire force 作为中间监督可以提升泛化，但真实车无法直接观测时，应只在仿真训练中作为辅助 loss，部署时不作为输入；
- normal load / load transfer 对湿滑和极限工况重要，建议通过物理层显式计算或 latent residual 校正；
- sideslip 和 tire force 估计文献能帮助设计可观测状态组合和不可辨识性分析。

### 1.4 物理约束、混合建模和不确定性

这类文献给出最接近工程落地的路线：低自由度物理 backbone + 神经网络补偿 + 约束训练 + 不确定性输出。

对本项目的直接启发：

- backbone 采用 3-DOF / 6-DOF bicycle 或 double-track vehicle dynamics；
- 神经模块负责有效转角、摩擦 latent、轮胎 residual、路面 transition residual；
- 约束包括能量/速度边界、摩擦椭圆、轮胎力饱和、normal load 非负、yaw/roll/pitch 稳定性；
- 不确定性应区分 aleatoric uncertainty（路面不可见、噪声）和 epistemic uncertainty（分布外、未覆盖工况）。

## 2. 50 篇候选文献清单

说明：这是 Phase 1 的工程导向候选清单，包含核心论文、相关论文和必要背景模型。下一阶段如果你确认范围，我会从中筛 15-20 篇做更深阅读，并把 top 8-12 篇写入 `research-wiki/`。

| # | 文献 | 年份 | 主题 | 与本项目关系 |
|---|---|---:|---|---|
| 1 | *Tire-Road Friction Coefficient Estimation: Review and Research Perspectives* | 2022 | TRFC 综述 | 核心综述，覆盖基于车辆动力学、轮胎、视觉等估计路线 |
| 2 | *Vehicle State Estimation in Automated Driving: A Survey on Methods, Challenges, and Future Directions* | 2025 | 车辆状态估计综述 | 对可观测状态、侧偏角、速度、摩擦估计有框架价值 |
| 3 | *A Review of Methods for State Estimation of Automated Guided Vehicles* | 2025 | AGV 状态估计综述 | 低速/工业场景状态估计参考 |
| 4 | *Estimation of Vehicle Dynamics, Tire Wear and Road Surface Conditions: A Survey* | 2024 | 车辆动力学/轮胎/路面综述 | 连接动力学、轮胎磨损与路面条件 |
| 5 | *A Comprehensive Review of Approaches to Modelling Tyre–Road Interaction* | 2025 | 轮胎-路面建模综述 | 物理 tire-road interaction 总览 |
| 6 | Pacejka, *Tire and Vehicle Dynamics* | 2002/2012 | 轮胎动力学基础 | Pacejka/Magic Formula 基础，物理 backbone 参考 |
| 7 | Bakker, Pacejka, Lidner, *A New Tire Model with an Application in Vehicle Dynamics Studies* | 1989 | Magic Formula | 轮胎非线性区建模基础 |
| 8 | Dugoff, Fancher, Segel, *An Analysis of Tire Traction Properties and Their Influence on Vehicle Dynamic Performance* | 1969 | Dugoff tire model | 简洁物理轮胎模型，适合 low-DOF backbone |
| 9 | Fiala, *Lateral Forces on Rolling Pneumatic Tires* | 1954 | Fiala tire model | 小侧偏到非线性饱和的经典模型 |
| 10 | Ray, *Nonlinear Tire Force Estimation and Road Friction Identification* | 1997 | 轮胎力/摩擦辨识 | 早期动力学可辨识性参考 |
| 11 | Rajamani et al., *Real-Time Estimation of Tire-Road Friction Coefficient* | 2010s | 实时 μ 估计 | 车辆动力学估计路线参考 |
| 12 | *Deep Dynamics: Vehicle Dynamics Modeling with a Physics-Informed Neural Network for Autonomous Racing* | 2024 | PINN vehicle dynamics | 核心：物理约束 NN 车辆动力学 |
| 13 | *Learning-Based Vehicle Dynamics Modeling for Autonomous Racing Using Gaussian Processes* | 2019 | GP dynamics | 自动赛车残差建模代表 |
| 14 | *Deep Kernel Learning for Vehicle Dynamics Identification* | 2023 | DKL dynamics | 不确定性和非线性动力学辨识 |
| 15 | *Deep Kernelized and Multilayer Gaussian Processes for Vehicle Dynamics Modeling* | 2025 | GP/DKL dynamics | 数据效率和不确定性参考 |
| 16 | *Learning Vehicle Dynamics Using Neural Network Model for Autonomous Vehicle Control* | 2020 | NN dynamics | 纯 NN dynamics 基线 |
| 17 | *Enhancing Long-Term Predictions in Vehicle Dynamics Using Neural Networks* | 2024 | 长期预测 | 直接对应 rollout 稳定性 |
| 18 | *Fine-Tuning Hybrid Physics-Informed Neural Networks for Vehicle Dynamics Model* | 2025 | Hybrid PINN | 物理模型 + NN 调参 |
| 19 | *Learning-Augmented MPC with Dynamics Residuals: Towards Data-Driven Autonomous Driving* | 2026 | residual dynamics | residual learning 与控制模型 |
| 20 | *Neural Networks for Vehicle Dynamics Modelling* | 2000s | NN dynamics | 早期 NN 车辆模型参考 |
| 21 | *Physics-Informed Neural Network for Vehicle Dynamics Modeling: Validation on Multibody Simulation* | 2020s | PINN + MBD | 高自由度仿真到低自由度学习的参考 |
| 22 | *Online Learning of Vehicle Dynamics Models for Autonomous Racing* | 2020s | 在线自适应 | 路面变化和模型漂移的在线适配参考 |
| 23 | *Real-Time Friction Estimation Based on Vehicle Dynamics and Kalman Filtering* | 2010s | Kalman μ 估计 | μ 作为 latent state 的滤波参考 |
| 24 | *Tire-Road Friction Coefficient Estimation Based on Vehicle Dynamics and Vision Data Fusion* | 2010s | 视觉+动力学融合 | 第二阶段接入视觉的参考 |
| 25 | *Frequency-Domain Data Fusion for Tire-Road Friction Coefficient Estimation* | 2016 | 频域融合 | 稀疏输入和轮速信号利用参考 |
| 26 | *Adaptive Estimation of Tire-Road Friction Coefficient Using Vehicle Dynamics* | 2017 | 自适应 μ 估计 | 低附着自适应估计 |
| 27 | *Tire-Road Friction Coefficient Estimation Using a Dynamic Neural Network* | 2010s | DNN μ 估计 | 数据驱动 μ 估计基线 |
| 28 | *Road Surface Condition Estimation Using Deep Neural Networks and Vehicle Dynamics Signals* | 2020s | 路面分类 | 隐式路面状态估计参考 |
| 29 | *Tire-Road Friction Estimation Using Deep Learning and Wheel Speed Signals* | 2020s | 轮速+DL | 轮速对低附着的可辨识性参考 |
| 30 | *Tire-Road Friction Coefficient Estimation Based on Phase Plane and Neural Network* | 2020s | phase plane + NN | 横摆/侧偏相平面结合 NN |
| 31 | *Joint Estimation of Vehicle States and Tire-Road Friction Using Unscented Kalman Filter* | 2010s | UKF 状态+μ | joint latent state 思路 |
| 32 | *Road Friction Estimation for Four-Wheel Independent Drive Electric Vehicles* | 2020s | 四轮独立估计 | Split-μ 和四轮差异参考 |
| 33 | *Split-μ Road Identification and Vehicle Stability Control* | 2010s | Split-μ | 左右附着不同的动力学响应 |
| 34 | *Spatio-Temporal Tire-Road Friction Estimation for Automated Vehicles* | 2026 | 空间-时间 μ | 路面衔接和局部 μ field 参考 |
| 35 | *Friction-Adaptive Stochastic Nonlinear Model Predictive Control for Autonomous Vehicles* | 2023 | friction-adaptive model | 不确定 μ 下鲁棒预测参考 |
| 36 | *Neural Network Tire Force Modeling for Automated Drifting* | 2024 | 极限轮胎力 NN | 大侧偏、非线性区、极限操控参考 |
| 37 | *Auxiliary-Enhanced Neural Network for Tire Lateral Force Estimation* | 2025 | 轮胎侧向力估计 | tire force 中间监督参考 |
| 38 | *Online Estimation of Three-Directional Tire Forces Using Self-Organizing Neural Networks* | 2023 | 三向轮胎力 | 四轮力估计方法参考 |
| 39 | *Bayesian Neural Network Based Intelligent Tire Force Estimation* | 2025 | BNN tire force | tire force 不确定性估计 |
| 40 | *Deep Learning Based Intelligent Tire Force Estimation via Domain Adaptation* | 2025 | domain adaptation | 仿真到真实迁移参考 |
| 41 | *Multi-Granularity Hierarchical Collaborative Networks for Intelligent Tire Force Estimation* | 2026 | tire force DL | 多尺度特征提取参考 |
| 42 | *Tire Normal Force Estimation Using Artificial Neural Networks and Fuzzy Logic* | 2023 | 法向载荷 | 载荷转移/normal load 参考 |
| 43 | *A Hierarchical Estimator for Vehicle States and Tire Forces* | 2020s | 状态+轮胎力 | 层级估计结构 |
| 44 | *Vehicle Sideslip Angle Estimation Using Neural Networks* | 2010s | 侧偏角估计 | `vy`/β 可观测性与估计误差参考 |
| 45 | *A Literature Survey on Vehicle Sideslip Angle Estimation* | 2024 | 侧偏角综述 | 状态估计中的不可辨识性参考 |
| 46 | *Learning to Predict Vehicle Dynamics from On-Board Sensors* | 2020s | onboard sensor prediction | 只用可观测传感器的预测框架 |
| 47 | *Uncertainty-Aware Learning for Vehicle Dynamics Prediction* | 2020s | 不确定性 | rollout 置信度输出参考 |
| 48 | *Probabilistic Deep Ensembles for Vehicle Trajectory and Dynamics Prediction* | 2020s | ensemble uncertainty | epistemic uncertainty 参考 |
| 49 | *Physics-Constrained Recurrent Neural Networks for Dynamical Systems* | 2020s | constrained RNN | 长序列约束学习可迁移到车辆 |
| 50 | *Neural Ordinary Differential Equations for Learning Dynamical Systems* | 2018 | Neural ODE | 连续时间动力学学习基础，可作为模型族参考 |

## 3. 对本项目最关键的结构性 gap

### G1：低激励常规工况下，路面 μ 的可辨识性不足

在常规稳态驾驶中，轮胎远离饱和区，车辆状态对 μ 的敏感性很低。只用 `vx, vy, r, omega_i, torque_i, brake_i, sw_angle` 的历史序列，很多路面状态会产生近似相同的响应。工程上不能让模型“假装知道真实 μ”，更合理的是输出一个等效 latent μ 分布或置信区间。

建议：把路面 latent 设计成带不确定性的状态 `z_mu_i(t)`，并在低激励时扩大 uncertainty，而不是强行回归单点 μ。

### G2：Split-μ 不能用单一全车路面状态表示

很多学习动力学论文只假设单一全局路面条件，最多做 dry/wet/snow 分类。这不够支持四轮同时处于不同路面。Split-μ 场景需要 wheel-level 或 axle/side-level latent，例如 `z_mu_fl, z_mu_fr, z_mu_rl, z_mu_rr`。

建议：第一版至少保留四轮级 latent friction head，并通过摩擦椭圆、左右 yaw moment consistency 和轮速滑移约束训练。

### G3：路面衔接是动态过程，不是静态分类

湿滑/干燥边界处，车辆响应存在时间滞后、轮胎 relaxation、载荷转移和控制输入耦合。逐帧路面分类无法建模 transition。需要时序 latent 更新，类似 state-space model 或 recurrent filter。

建议：用 `history encoder + latent dynamics` 表示路面和轮胎状态，并让 latent 以有限速率变化，避免每帧跳变。

### G4：轮胎小角度和大角度区域应分区建模

小侧偏角下，轮胎力近似线性；大侧偏/低附着时进入饱和，非线性强且数据稀疏。单个 MLP 直接拟合所有区域，容易在常规数据上表现好、极限区域失败。

建议：使用物理 tire model 作为 backbone，神经网络预测参数/residual；或采用 mixture-of-experts，按 slip ratio、slip angle、normal load、latent μ 软门控。

### G5：仿真数据到真实车辆存在 sim-to-real 风险

高自由度仿真器可生成大量数据，但真实车不可观测内部状态不能作为输入。若训练时过度依赖仿真内部变量，部署会失败。即使输入约束一致，仿真器的 tire model、noise、actuator lag、steering ratio、sensor filtering 也会造成 domain gap。

建议：训练时只输入真实车可观测量；内部仿真量仅用于辅助 loss 或 teacher signal；做 domain randomization 和参数扰动；把不可辨识因素作为 latent uncertainty。

### G6：长期 rollout 稳定性比 one-step loss 更重要

很多文献报告 one-step 或短 horizon 误差，但仿真系统需要长时多步预测。模型若没有状态约束，会在速度、姿态角、yaw rate、wheel speed 上逐步漂移。

建议：训练目标必须包含多步 rollout loss、closed-loop scheduled sampling、物理一致性 loss 和稳定性惩罚；评估也要按场景分组。

## 4. 推荐的后续检索/精读方向

确认范围后，建议 Phase 2 前先精读 15-20 篇，重点不是论文创新，而是抽取工程可用模块：

1. **Hybrid / physics-informed vehicle dynamics**：#12, #13, #14, #15, #17, #18, #21  
2. **TRFC / low friction / Split-μ**：#1, #23-35  
3. **Tire force / load transfer / sideslip**：#36-45  
4. **Uncertainty / rollout stability**：#17, #35, #47-50  

## 5. 初步方案方向，供下一阶段 idea generation 使用

这里先不正式生成 idea，只记录从文献 landscape 得到的工程方向：

- **方向 A：低自由度物理 backbone + 四轮 latent μ + neural residual**  
  适合第一版落地。物理层处理基础车辆动力学和轮胎边界，神经网络补偿未知传动比、转向作用、轮胎非线性和仿真/真实差异。

- **方向 B：history encoder 做可辨识性与置信度估计**  
  从过去 `T` 秒状态、轮速、扭矩、制动、转向中提取路面 latent；低激励时输出高不确定性，而不是硬分类。

- **方向 C：wheel-level mixture-of-experts tire residual**  
  按四轮 slip、normal load、latent μ、速度区间软门控不同专家，分别处理线性区、非线性区、低附着区和 transition 区。

- **方向 D：仿真器 teacher + 部署输入约束一致**  
  高自由度仿真器内部 tire force、normal load、true μ 可作为辅助监督，但不作为模型输入；部署模型只使用 requirement 中列出的可观测量。

## 6. 检查点问题

请确认以下范围是否符合你的理解：

1. 文献范围是否应以“工程可落地的 hybrid dynamics + TRFC/低附着估计”为主，而不是追求新颖论文 idea？
2. 50 篇清单中，是否需要我下一步把 TRFC / tire force / hybrid dynamics 三类分别扩展成更细的 annotated bibliography？
3. 第一版 idea generation 是否只考虑“无视觉，仅车辆可观测状态”的方案，把视觉作为第二阶段可选增强？

在你确认前，我不会进入 Phase 2。

