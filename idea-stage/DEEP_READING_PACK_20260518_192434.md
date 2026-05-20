# 当前方案精读包：Hybrid 车辆动力学、载荷转移、低附着与不确定性

**日期**：2026-05-18  
**阶段**：`idea-discovery / Phase 1.5 targeted deep reading`  
**目标**：围绕当前已定方案做精读，而不是泛读 50 篇。当前方案为：

```text
double-track 5-DOF body + 4-wheel rotational dynamics
+ roll/pitch dynamics
+ explicit Fz_fl/fr/rl/rr
+ wheel-level latent μ
+ tire physics + tire residual
+ small vehicle residual
+ uncertainty-aware rollout
```

## 1. 精读原则

每篇文献只抽取对工程方案有用的信息：

- 物理模型：状态、自由度、方程、参数、可观测量；
- 轮胎模型：`Fx, Fy, Fz, κ, α, μ` 的处理方式；
- 载荷转移：如何计算 `Fz_i`，是否包含 roll/pitch/suspension；
- 低附着：如何估计 μ，是否处理 Split-μ 和路面衔接；
- 学习模型：NN/GP/Residual 放在哪里，是否有物理约束；
- 不确定性：输出什么置信度，如何训练和校准；
- 评估：是否做 long rollout，是否按 dry/wet/snow/split-μ 分组。

## 2. 第一批必读：物理 backbone 和载荷转移

### P1. Pacejka, *Tire and Vehicle Dynamics*

- 作用：确定轮胎物理层、Magic Formula、载荷敏感性、摩擦椭圆。
- 重点读：
  - lateral/longitudinal tire force；
  - combined slip；
  - tire force saturation；
  - load sensitivity。
- 对方案的影响：
  - 决定 `TirePhysics(κ, α, Fz, μ, θ)` 的基础形式；
  - 决定 `TireResidualNN` 的边界和归一化方式。

### P2. Rajamani, *Vehicle Dynamics and Control*

- 作用：确定低自由度车辆动力学、状态估计、控制导向建模。
- 重点读：
  - bicycle model 与更高阶车辆模型；
  - yaw/sideslip dynamics；
  - tire-road friction/state estimation。
- 对方案的影响：
  - 确定 `vx, vy, r` 主方程；
  - 确定哪些状态适合作为 observer/history encoder 输入。

### P3. Gillespie, *Fundamentals of Vehicle Dynamics*

- 作用：确定载荷转移、侧倾/俯仰、悬架基础关系。
- 重点读：
  - longitudinal load transfer；
  - lateral load transfer；
  - roll center / CG height；
  - pitch/roll stiffness 和 damping。
- 对方案的影响：
  - 确定 `Fz_static + ΔFz_long + ΔFz_lat + ΔFz_roll/pitch` 的基础公式。

### P4. Milliken & Milliken, *Race Car Vehicle Dynamics*

- 作用：理解左右载荷转移、roll stiffness distribution、极限操控。
- 重点读：
  - lateral load transfer distribution；
  - roll stiffness front/rear split；
  - tire load sensitivity 对极限操控的影响。
- 对方案的影响：
  - 决定是否需要 `k_roll_front / k_roll_rear` 这类可学习参数。

## 3. 第二批必读：Hybrid / learned vehicle dynamics

### P5. *Deep Dynamics: Vehicle Dynamics Modeling with a Physics-Informed Neural Network for Autonomous Racing*

- 来源：arXiv / autonomous racing 方向。
- 作用：当前 hybrid 模型最接近的学习动力学参考。
- 重点读：
  - physics-informed neural network 如何嵌入 dynamics；
  - 使用哪些状态/控制输入；
  - loss 中如何加入物理项；
  - 是否报告 long-horizon prediction。
- 对方案的影响：
  - 决定 `VehiclePhysics + VehicleResidualNN` 的训练方式；
  - 判断 residual 应该作用于 acceleration、force 还是 next state。

### P6. Kabzan et al., *Learning-Based Vehicle Dynamics Modeling for Autonomous Racing*

- 来源：IEEE Robotics and Automation Letters / autonomous racing。
- 作用：GP residual dynamics 和不确定性参考。
- 重点读：
  - GP 修正模型误差的位置；
  - data-driven dynamics 在极限赛车中的数据需求；
  - uncertainty 如何用于控制。
- 对方案的影响：
  - 给 `VehicleResidualNN` 设置“不确定性/小残差”定位；
  - 对比 NN residual 与 GP residual 的工程取舍。

### P7. *Deep Kernel Learning for Vehicle Dynamics Identification*

- 作用：非线性动力学辨识和不确定性。
- 重点读：
  - DKL 输入输出定义；
  - 与 GP/NN 的比较；
  - 不确定性是否可靠。
- 对方案的影响：
  - 判断是否用 ensemble/BNN/DKL 估计 epistemic uncertainty。

### P8. *Enhancing Long-Term Predictions in Vehicle Dynamics Using Neural Networks*

- 作用：长 rollout 稳定性。
- 重点读：
  - one-step 与 multi-step loss 的差异；
  - scheduled sampling / rollout training；
  - 长期漂移抑制方式。
- 对方案的影响：
  - 决定训练目标必须以 multi-step rollout 为主，而不是只做 one-step。

## 4. 第三批必读：低附着、TRFC、Split-μ

### P9. *Tire-Road Friction Coefficient Estimation: Review and Research Perspectives*

- 作用：μ 估计综述，必须读。
- 重点读：
  - 基于车辆动力学的 μ 估计；
  - 基于轮胎/轮速的 μ 估计；
  - 基于视觉的 μ 估计；
  - 低激励下的可辨识性限制。
- 对方案的影响：
  - 支撑 `μ_i` 是 latent distribution，而不一定是真实单点 μ；
  - 决定什么时候输出高不确定性。

### P10. *Road Friction Estimation for Four-Wheel Independent Drive Electric Vehicles*

- 作用：四轮级摩擦估计参考。
- 重点读：
  - 是否独立估计四个轮的 μ；
  - 使用轮端扭矩和轮速的方式；
  - 低附着识别的激励条件。
- 对方案的影响：
  - 支持 `μ_fl, μ_fr, μ_rl, μ_rr` 的 wheel-level latent 设计。

### P11. *Split-μ Road Identification and Vehicle Stability Control*

- 作用：Split-μ 场景动力学响应。
- 重点读：
  - 左右附着差如何诱导 yaw moment；
  - 稳定性控制如何利用 Split-μ 识别；
  - 需要哪些传感器信号。
- 对方案的影响：
  - 决定 Split-μ 数据生成和评估场景；
  - 验证 `Fz_i * μ_i` 对左右 yaw moment 的影响。

### P12. *Spatio-Temporal Tire-Road Friction Estimation for Automated Vehicles*

- 作用：路面衔接和空间/时间 μ 表达。
- 重点读：
  - μ 是否作为时空场；
  - 当前车辆历史如何更新路面状态；
  - 是否考虑路面 transition。
- 对方案的影响：
  - 支持 `history encoder + latent μ dynamics`，而不是逐帧分类。

## 5. 第四批必读：轮胎力、法向载荷和 residual

### P13. *Neural Network Tire Force Modeling for Automated Drifting*

- 作用：大侧偏、非线性区轮胎力学习。
- 重点读：
  - drift/extreme slip 下 tire force 怎么学；
  - 是否用物理模型或纯 NN；
  - 训练数据覆盖和泛化。
- 对方案的影响：
  - 决定 `TireResidualNN` 在大侧偏区的结构；
  - 支持 mixture-of-experts 或 slip-region gating。

### P14. *Auxiliary-Enhanced Neural Network for Tire Lateral Force Estimation*

- 作用：用辅助物理量估计 tire lateral force。
- 重点读：
  - auxiliary variables 包括什么；
  - 是否使用 `Fz, α, κ, μ`；
  - 与传统估计方法比较。
- 对方案的影响：
  - 指导 tire residual 输入特征设计。

### P15. *Bayesian Neural Network Based Intelligent Tire Force Estimation*

- 作用：tire force 不确定性。
- 重点读：
  - BNN 输出的 uncertainty 类型；
  - 数据噪声与分布外不确定性如何区分；
  - 估计对象是 `Fx/Fy/Fz` 还是单一方向力。
- 对方案的影响：
  - 指导 `TireResidualNN` 或 `UncertaintyHead` 的 uncertainty 设计。

### P16. *Tire Normal Force Estimation Using Artificial Neural Networks and Fuzzy Logic*

- 作用：`Fz_i` 数据驱动补偿参考。
- 重点读：
  - 输入信号有哪些；
  - 是否显式考虑 suspension / roll / pitch；
  - ANN/fuzzy 是否直接输出 normal force。
- 对方案的影响：
  - 对比“直接输出 Fz”和“输出 ΔFz residual”的风险。

### P17. *Estimation of Wheel-Ground Contact Normal Forces of a Vehicle from Experimental Data Validation*

- 作用：四轮法向载荷估计与实验验证。
- 重点读：
  - 估计模型的输入和验证方式；
  - 是否需要不可部署传感器；
  - transient load transfer 处理方式。
- 对方案的影响：
  - 帮助设计 `Fz_i` 验证指标。

## 6. 第五批：不确定性和序列动力学

### P18. *Physics-Constrained Recurrent Neural Networks for Dynamical Systems*

- 作用：物理约束序列模型。
- 重点读：
  - 约束如何进入 RNN；
  - 长期 rollout 是否更稳定；
  - 是否可迁移到车辆动力学。
- 对方案的影响：
  - 指导 `history encoder` 和 rollout loss。

### P19. Chen et al., *Neural Ordinary Differential Equations*

- 作用：连续时间动力学基础。
- 重点读：
  - 可微 ODE solver；
  - irregular sampling；
  - 与离散 RNN/MLP dynamics 的取舍。
- 对方案的影响：
  - 若仿真步长可变，可考虑 Neural ODE；否则第一版不必采用。

### P20. Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*

- 作用：工程上最实用的不确定性基线。
- 重点读：
  - ensemble 如何估计 epistemic；
  - aleatoric head 如何训练；
  - calibration 指标。
- 对方案的影响：
  - 第一版 uncertainty 推荐用 deep ensemble + heteroscedastic head，而不是复杂 BNN。

## 7. 读完后要冻结的工程决策

精读完成后，需要冻结以下内容：

1. `Fz_i` 公式：是否采用 roll stiffness distribution；
2. tire model：Pacejka / Dugoff / Fiala / simplified brush model；
3. tire residual 输出：`ΔFx/ΔFy` 还是 tire parameter residual；
4. `μ_i` 表达：标量、分布、离散路面类别 + 连续附着；
5. residual 作用点：force-level、acceleration-level、state-level；
6. uncertainty：ensemble、heteroscedastic Gaussian、BNN 或 GP；
7. rollout loss horizon 和分场景权重。

## 8. 需要剔除或降级的方向

以下方向暂时不作为第一版主线：

- 纯端到端 `x_{t+1}=NN(history)`：不可解释，难加 Split-μ 和摩擦约束；
- 只用 bicycle model：无法充分表达左右轮 Split-μ 和四轮载荷；
- 直接用 NN 输出全部四轮 `Fz_i`：容易违反质量守恒和载荷转移方向；
- 第一版接视觉：会增加数据和同步复杂度，先保留为第二阶段增强；
- 把高自由度仿真器内部变量作为输入：与真实车部署不一致。

## 9. 下一步

建议接下来按以下顺序精读并产出笔记：

```text
P1-P4  -> 冻结物理 backbone 和 Fz_i 公式
P9-P12 -> 冻结 μ_i / Split-μ / 可辨识性假设
P13-P17 -> 冻结 TireResidualNN 和 FzResidualNN
P5-P8, P18-P20 -> 冻结学习框架、rollout loss 和 uncertainty
```

每篇笔记写入 `research-wiki/papers/`，同时更新 `query_pack.md`。

