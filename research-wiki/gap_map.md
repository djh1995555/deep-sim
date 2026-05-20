# Gap Map

## G1: 低激励常规工况下路面 μ 可辨识性不足

- Status: unresolved
- Source: idea-discovery Phase 1 literature landscape, 2026-05-18
- Summary: 常规稳态驾驶时轮胎远离饱和区，车辆状态对 tire-road friction coefficient 的敏感性弱。只用可观测车辆状态历史难以唯一辨识真实 μ。
- Engineering implication: 模型应输出带不确定性的等效路面 latent，而不是强行回归确定 μ。

## G2: Split-μ 需要四轮级或左右级 latent friction

- Status: unresolved
- Source: idea-discovery Phase 1 literature landscape, 2026-05-18
- Summary: 单一全车路面状态无法描述四个车轮同时处于不同附着条件的情况。
- Engineering implication: 第一版模型至少应保留 wheel-level latent friction head，并用轮速滑移、yaw moment consistency 和摩擦椭圆约束训练。

## G3: 路面衔接需要时序 latent transition

- Status: unresolved
- Source: idea-discovery Phase 1 literature landscape, 2026-05-18
- Summary: 干湿/冰雪边界不是逐帧静态分类问题，车辆响应受轮胎 relaxation、载荷转移、控制输入和历史状态影响。
- Engineering implication: 使用 history encoder + latent dynamics，避免路面状态每帧跳变。

## G4: 小侧偏线性区与大侧偏非线性区需要分区建模

- Status: unresolved
- Source: idea-discovery Phase 1 literature landscape, 2026-05-18
- Summary: 轮胎线性区和饱和区物理规律差异大，单个黑盒模型容易在常规区拟合好、极限区失败。
- Engineering implication: 优先采用物理 tire backbone + learned residual，或按 slip / load / latent μ 做 mixture-of-experts。

## G5: 高自由度仿真到真实车存在 sim-to-real gap

- Status: unresolved
- Source: idea-discovery Phase 1 literature landscape, 2026-05-18
- Summary: 高自由度仿真器可提供内部变量，但真实车不可观测。若训练时把内部变量作为输入，会破坏部署一致性。
- Engineering implication: 内部变量只能作为辅助监督或 teacher signal；部署输入必须限制在真实可观测信号。

## G6: 长时 rollout 稳定性比 one-step loss 更关键

- Status: unresolved
- Source: idea-discovery Phase 1 literature landscape, 2026-05-18
- Summary: 仿真系统需要长时多步预测，one-step 误差低不等于 rollout 稳定。
- Engineering implication: 训练和评估必须包含 multi-step rollout loss、scheduled sampling、物理一致性和场景分组误差。

## G7: 侧倾/俯仰动态与四轮法向载荷必须显式进入 hybrid dynamics

- Status: unresolved
- Source: load-transfer targeted survey, 2026-05-18
- Summary: `Fz_i` 决定每个轮胎的力上限，进而影响低附着、Split-μ、yaw moment 和长 rollout 稳定性。若不显式计算 `Fz_fl/fr/rl/rr`，神经网络会把载荷转移、路面附着和轮胎非线性混在 latent 中，泛化和约束都会变差。
- Engineering implication: 使用 double-track backbone；把 `roll, pitch, p, q` 放入物理状态更新；通过 roll/pitch dynamics 和载荷转移模型计算 `Fz_i`；神经网络只输出受约束的 `ΔFz_i` residual。
