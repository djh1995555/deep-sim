# 车辆高保真动力学深度学习模型：Phase 1 文献调研与范围检查

**日期**：2026-05-18  
**输入 brief**：`/home/mi/vibe/research/deep_sim/requirement.md`  
**Pipeline 阶段**：`idea-discovery / Phase 1: research-lit`  
**AUTO_PROCEED**：false，本文件用于人工检查点，确认后再进入 idea generation。

完整内容见 timestamped 版本：`idea-stage/LITERATURE_SURVEY_20260518_183048.md`。

## 检查点摘要

建议将后续范围聚焦在：工程可落地的 **hybrid vehicle dynamics + tire-road friction / low-friction estimation + wheel-level latent friction + uncertainty-aware rollout**。

关键 gap：

- 低激励常规工况下 μ 的可辨识性不足，不能强行输出确定单点；
- Split-μ 需要四轮级或左右级 latent friction；
- 路面衔接应作为时序 latent transition，而不是逐帧分类；
- 小侧偏线性区和大侧偏非线性区应通过物理 tire backbone 或 mixture-of-experts 区分；
- 高自由度仿真器内部变量只适合作辅助监督，不应作为部署输入；
- 评估必须以长时 rollout 和分场景误差为核心。

## 下一步需你确认

1. 文献范围是否应以“工程可落地的 hybrid dynamics + TRFC/低附着估计”为主？
2. 是否需要我下一步把 50 篇清单扩展成逐篇 annotated bibliography？
3. Phase 2 是否只考虑“无视觉，仅车辆可观测状态”的第一阶段方案？

