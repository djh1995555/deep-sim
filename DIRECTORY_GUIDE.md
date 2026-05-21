# 目录说明

本文件说明当前仓库各目录的内容和用途。当前状态截至 2026-05-21。

## 顶层目录总览

| 路径 | 类型 | 内容 | 用途 |
| --- | --- | --- | --- |
| `.agents/` | 内部工具 | 本仓库随附的 agent skills 定义 | 给 Codex/ARIS 工作流使用，例如 `experiment-bridge`、`research-wiki`、`experiment-plan`。不是车辆动力学模型源码。 |
| `.git/` | Git 元数据 | 提交历史、对象库、分支引用 | Git 内部目录，不需要手动修改。 |
| `configs/` | 实验配置 | Teacher 数据集配置和每个 run 的 YAML 配置 | 定义实验入口参数，是复现实验的主要配置来源。 |
| `data/` | 数据占位 | 当前为空 | 预留给后续持久化数据集或外部数据。当前 scaffold 数据主要写入 `runs/*/artifacts/`。 |
| `experiments/` | 实验执行代码 | runner、sanity、baseline、hybrid、ablation 报告代码 | `python -m experiments.run --config ...` 的执行入口和实验逻辑。 |
| `idea-stage/` | 早期研究资料 | 文献调研、精读、idea 报告、最终设计决策 | 记录从需求到方案形成的早期推理链，偏研究探索。 |
| `refine-logs/` | 方案与规格文档 | 实验计划、模块设计、数据设计、Teacher 设计、运行规格、结果总结 | 当前方案设计和实验执行状态的权威文档目录。 |
| `reports/` | 实验报告 | B0/B3/B4 阶段报告和 ablation 汇总 JSON | 面向阅读的实验阶段输出，通常由 run 或汇总脚本生成/更新。 |
| `research-wiki/` | 研究知识库 | 论文卡片、idea 卡片、claim/gap/query 记录 | 持久化研究知识库，用于追踪证据、文献和想法之间的关系。 |
| `runs/` | 实验运行产物 | R000-R033 的 artifacts、logs、checkpoints | 每次实验运行的输出目录。当前主要是 scaffold 数据、指标和报告中间产物。 |
| `teacher_simulator/` | Teacher simulator 源码 | 高保真车辆动力学 teacher 的当前 v0/scaffold 实现 | 生成 DS0/DS1/DS1 proxy 数据，支撑 sanity、baseline 和 hybrid scaffold 实验。 |
| `tests/` | 测试代码 | teacher simulator 单元测试 | 验证数据生成、采样和基础物理逻辑没有被破坏。 |

## 配置目录

### `configs/`

实验配置根目录。

### `configs/teacher/`

Teacher 数据集生成配置。

| 文件 | 用途 |
| --- | --- |
| `ds0_minimal.yaml` | 最小 teacher simulator 数据集配置，用于早期 sanity 和 R000-R000d。 |
| `ds1_v1.yaml` | DS1 多车、多工况 scaffold 数据集配置，用于当前主线实验。 |
| `ds1_proxy_v1.yaml` | sim-to-real proxy 数据配置，用于扰动 profile、target window 和分布 sanity。 |

### `configs/runs/`

每个实验 run 的入口配置。命名和 `runs/` 下的输出目录一一对应。

当前范围：

| Run 范围 | 作用 |
| --- | --- |
| `R000-R000d` | Teacher simulator DS0 最小构建和 sanity。 |
| `R000e-R000h` | DS1 scenario matrix、车辆参数随机化、split generation、dataset QA。 |
| `R001-R004b` | 数据 schema、物理一致性、dt 对齐、派生量、tiny learnability、physics rollout smoke。 |
| `R004c-R004e` | sim-to-real proxy 扰动、target time window、分布 sanity。 |
| `R005-R008` | physics-only 和 black-box baseline scaffold。 |
| `R009-R014` | base hybrid scaffold 训练、评估、残差审计和多 seed 复现。 |
| `R015-R033` | E/T/F/S/M/V/U 组件 ablation scaffold。 |
| `R034-R036` | B5 跨车 / 跨配置泛化 scaffold。 |
| `R037` | M8 final single model scaffold checkpoint descriptor。 |
| `R038-R045` | B6 target fine-tune 数据效率 scaffold。 |

运行方式统一使用 Miniforge/conda 环境：

```bash
conda run -n deep-sim python -m experiments.run --config configs/runs/R009.yaml
```

## 实验代码目录

### `experiments/`

实验执行和评估逻辑。

| 文件 | 用途 |
| --- | --- |
| `run.py` | 统一实验入口。读取 run config，调度 teacher、sanity、baseline、hybrid、ablation 等任务。 |
| `sanity.py` | 数据和物理 sanity 检查，包括 schema、dt、派生量、learnability 和 rollout smoke。 |
| `baselines.py` | physics-only baseline 和 black-box baseline scaffold 逻辑。 |
| `hybrid.py` | base hybrid、组件 ablation、评估指标和 residual 审计的 scaffold 实现。 |
| `ablation_report.py` | 汇总 R015-R033 组件 ablation 结果，生成 markdown/JSON 报告。 |
| `__init__.py` | Python package 标记。 |

`experiments/__pycache__/` 是 Python 自动生成的缓存目录，不属于源码。

## Teacher Simulator 目录

### `teacher_simulator/`

Teacher simulator 当前实现。它负责生成可控的车辆动力学数据，用于第一版模型训练与 sanity 验证。

| 文件 | 用途 |
| --- | --- |
| `config.py` | 数据集和仿真配置结构。 |
| `generate.py` | 数据生成主流程。 |
| `export.py` | episode、metadata、split 等导出逻辑。 |
| `scenario.py` | 工况组合、采样和 episode 定义。 |
| `simulator.py` | 单 episode 仿真推进主逻辑。 |
| `state.py` | 车辆状态和中间状态结构。 |
| `vehicle_params.py` | 车辆参数、参数随机化和 nominal prior。 |
| `validate.py` | 基础验证入口。 |
| `validators.py` | schema、泄漏检查、物理一致性等 validator。 |

### `teacher_simulator/modules/`

Teacher simulator 的物理/工程子模块。

| 文件 | 用途 |
| --- | --- |
| `aero.py` | 空气动力学相关项。 |
| `drive_brake.py` | 驱动/制动输入和轮端作用建模。 |
| `road.py` | 单一路况、split-μ、路面变化等 road profile。 |
| `sensors.py` | 传感器噪声、观测字段和可见/不可见字段处理。 |
| `steering.py` | 转向系统动态、延迟/滤波等 steering actuator 逻辑。 |
| `suspension.py` | 悬架、载荷转移和法向载荷相关近似。 |
| `tire.py` | 轮胎力、摩擦约束和 combined-slip 相关逻辑。 |

`teacher_simulator/**/__pycache__/` 是 Python 自动生成的缓存目录，不属于源码。

## 文档目录

### `refine-logs/`

当前方案设计、执行计划和阶段性结果的主文档目录。原则是：

- `EXPERIMENT_PLAN.md` 只放实验设计方案。
- 组件细节放对应 DESIGN/SPEC 文档。
- 阶段运行状态放 tracker/results/code review。

| 文件 | 用途 |
| --- | --- |
| `EXPERIMENT_PLAN.md` | 当前实验设计总方案，包含实验块、成功标准、数据/模型/训练策略的高层说明。 |
| `EXPERIMENT_TRACKER.md` | R000-R033 的执行跟踪表，记录 TODO/PASS 等状态。 |
| `EXPERIMENT_RESULTS.md` | 最新实验结果汇总。 |
| `EXPERIMENT_CODE_REVIEW.md` | 当前实现和实验产物的本地 code review 记录。 |
| `EXPERIMENT_RUN_SPEC.md` | 实验运行规范，包含 conda 环境要求和运行命令约定。 |
| `DATA_DESIGN.md` | 数据集设计细节，包括场景组合、字段、split、采样比例和 fine-tune 数据桶。 |
| `DATA_SCHEMA_SPEC.md` | 数据 schema 规格。 |
| `MODULE_DESIGN.md` | Student model 组件设计，包括 encoder、residual、MuHead、fine-tune adapter 等。 |
| `STUDENT_MODEL_SPEC.md` | Student model 的实现规格。 |
| `TEACHER_SIMULATOR_DESIGN.md` | Teacher simulator 设计文档。 |
| `TEACHER_SIMULATOR_SPEC.md` | Teacher simulator 实现规格。 |
| `*_2026*.md` | 历史快照或阶段性 checkpoint，用于追溯方案变更。 |

### `idea-stage/`

早期研究阶段文档。它回答“为什么选择现在这个方向”。

| 文件类型 | 用途 |
| --- | --- |
| `LITERATURE_SURVEY*.md` | 初始文献版图和候选论文。 |
| `LOAD_TRANSFER_SURVEY*.md` | 侧倾、俯仰、载荷转移相关专项调研。 |
| `DEEP_READING_PACK*.md` | 精读论文列表和阅读优先级。 |
| `PHYSICS_BACKBONE_READING*.md` | 车辆物理 backbone 相关精读。 |
| `DEEP_READING_SYNTHESIS*.md` | P5-P20 等论文的综合分析。 |
| `FINAL_DESIGN_DECISIONS*.md` | 早期方案冻结决策。 |
| `IDEA_REPORT*.md` | idea 排名和架构方案演化。 |

### `research-wiki/`

持久化研究知识库。

| 子目录/文件 | 用途 |
| --- | --- |
| `index.md` | wiki 首页和索引。 |
| `log.md` | wiki 更新日志。 |
| `gap_map.md` | 当前方案和文献/实验之间的 gap 映射。 |
| `query_pack.md` | 后续检索问题包。 |
| `papers/` | 论文卡片，每篇论文一个 markdown 文件。 |
| `ideas/` | idea 卡片，包括推荐方案、备选方案和淘汰方案。 |
| `claims/` | claim 相关记录目录，当前可能为空或待扩展。 |
| `experiments/` | wiki 层面的实验记录目录，当前可能为空或待扩展。 |
| `graph/edges.jsonl` | 论文、idea、claim、experiment 之间的关系边。 |

## 报告与运行产物目录

### `reports/`

阶段性实验报告，适合直接阅读。

| 文件 | 用途 |
| --- | --- |
| `B0_teacher.md` | Teacher simulator、数据生成和 sanity 阶段报告。 |
| `B3_baselines.md` | physics-only 与 black-box baseline scaffold 报告。 |
| `B3_base_hybrid.md` | base hybrid scaffold 报告。 |
| `B4_ablations.md` | 组件 ablation scaffold 汇总报告。 |
| `B4_ablation_summary.json` | ablation 汇总的机器可读 JSON。 |
| `B5_cross_generalization.md` | 跨车 / 跨配置泛化 scaffold 汇总报告。 |
| `B5_cross_generalization_summary.json` | B5 汇总的机器可读 JSON。 |
| `B6_fine_tune.md` | FT0-FT6 × FTD0-FTD5 target fine-tune 数据效率 scaffold 汇总报告。 |
| `B6_fine_tune_summary.json` | B6 汇总的机器可读 JSON。 |

### `runs/`

每次实验运行的实际输出。目录名通常为 `Rxxx_实验名/`。

每个 run 目录通常包含：

| 子目录 | 用途 |
| --- | --- |
| `artifacts/` | 数据集、指标、summary、metadata、split 等运行产物。 |
| `logs/` | 日志目录。当前 scaffold run 中不一定都有大量日志。 |
| `checkpoints/` | checkpoint 目录。当前多数 run 是 scaffold，没有真实深度模型权重。 |

当前已完成的主要 run：

| Run 范围 | 内容 |
| --- | --- |
| `R000*` | Teacher simulator 和 DS0/DS1/DS1 proxy 数据生成。 |
| `R001-R004*` | 数据/物理 sanity、proxy sanity。 |
| `R005-R008` | baseline scaffold。 |
| `R009-R014` | base hybrid scaffold。 |
| `R015-R033` | 组件 ablation scaffold。 |
| `R034-R036` | cross-vehicle / cross-config generalization scaffold。 |
| `R037` | final single model scaffold checkpoint descriptor。 |
| `R038-R045` | target fine-tune data-efficiency scaffold。 |

注意：`runs/` 是运行结果目录，不是手写源码。重新运行实验可能覆盖或新增其中的文件。

## 测试目录

### `tests/`

自动化测试目录。

| 文件 | 用途 |
| --- | --- |
| `test_teacher_simulator.py` | Teacher simulator 相关单元测试和回归测试。 |
| `__init__.py` | Python package 标记。 |

运行方式：

```bash
conda run -n deep-sim python -m unittest tests.test_teacher_simulator
```

`tests/__pycache__/` 是 Python 自动生成的缓存目录，不属于源码。

## 根目录文件

| 文件 | 用途 |
| --- | --- |
| `MANIFEST.md` | 自动维护的产物清单，按时间记录技能、文件、阶段和说明。 |
| `DIRECTORY_GUIDE.md` | 当前文件，按目录解释仓库结构和用途。 |
| `environment.yml` | Miniforge/conda 环境定义。实验活动应使用 `deep-sim` 环境。 |
| `requirements.txt` | Python 依赖的轻量记录。当前主要依赖 `numpy`。 |
| `.gitignore` | Git 忽略规则。 |

## 哪些目录应该重点维护

日常开发和实验时，优先关注：

1. `refine-logs/`：方案、规格、tracker、结果是否同步。
2. `configs/runs/` 和 `configs/teacher/`：实验是否可复现。
3. `experiments/`：实验 runner 和评估逻辑。
4. `teacher_simulator/`：数据生成和 teacher 物理逻辑。
5. `reports/`：阶段性结果是否能支持当前 claim。
6. `tests/`：修改 teacher 或实验逻辑后是否有基本回归测试。

通常不要手动改：

1. `.git/`：Git 内部数据。
2. `__pycache__/`：Python 缓存。
3. `runs/*` 中的历史产物：除非明确是在清理或重新生成实验结果。
