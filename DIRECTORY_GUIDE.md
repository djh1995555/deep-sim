# 目录说明

本文件说明当前仓库各目录的内容和用途。当前状态截至 2026-05-21。

## 顶层目录总览

| 路径                    | 类型                   | 内容                                                                          | 用途                                                                                       |
| --------------------- | -------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `AGENTS.md`           | 运行环境说明               | `/run-experiment` 可读取的本地 conda 环境、GPU 状态和 smoke run 命令                      | 当前声明本地 `deep-sim` CUDA smoke 可用，GPU 为 NVIDIA RTX A4500。                                  |
| `.agents/`            | 内部工具                 | 本仓库随附的 agent skills 定义                                                      | 给 Codex/ARIS 工作流使用，例如 `experiment-bridge`、`research-wiki`、`experiment-plan`。不是车辆动力学模型源码。 |
| `.git/`               | Git 元数据              | 提交历史、对象库、分支引用                                                               | Git 内部目录，不需要手动修改。                                                                        |
| `EXPERIMENT_USAGE.md` | 手动实验使用指南             | 单 run、队列、矩阵报告、实车 CSV 适配、验证命令                                                | 给人工手动执行实验时使用，是最直接的操作入口。                                                                  |
| `configs/`            | 实验配置                 | Teacher 数据集配置和每个 run 的 YAML 配置                                              | 定义实验入口参数，是复现实验的主要配置来源。                                                                   |
| `data/`               | 数据入口                 | canonical dataset 真实目录                                                      | 给 PyTorch 训练阶段提供稳定数据路径，当前保留 `ds1_v1` 和 `ds1_proxy_ft_v1` 两个数据集。                          |
| `experiments/`        | 实验执行代码               | runner、sanity、baseline、hybrid、PyTorch smoke、ablation 报告代码                   | `python -m experiments.run --config ...` 的执行入口和实验逻辑。                                     |
| `idea-stage/`         | 早期研究资料               | 文献调研、精读、idea 报告、最终设计决策                                                      | 记录从需求到方案形成的早期推理链，偏研究探索。                                                                  |
| `refine-logs/`        | 方案与规格文档              | 实验计划、模块设计、数据设计、Teacher 设计、运行规格、结果总结                                         | 当前方案设计和实验执行状态的权威文档目录。                                                                    |
| `reports/`            | 实验报告                 | B0/B3/B4 阶段报告、PyTorch 开发报告、矩阵报告和 ablation 汇总 JSON                           | 面向阅读的实验阶段输出，通常由 run 或汇总脚本生成/更新。                                                          |
| `research-wiki/`      | 研究知识库                | 论文卡片、idea 卡片、claim/gap/query 记录                                             | 持久化研究知识库，用于追踪证据、文献和想法之间的关系。                                                              |
| `runs/`               | 运行时产物目录              | 当前不存在，重新执行实验时由 runner 自动创建                                                     | 只用于保存新 run 的输出；canonical 数据已经迁移到 `data/`。                                               |
| `student_model/`      | PyTorch 模型源码         | Student model 的数据接口、常量、E1/E2/E3 encoder、residual heads、adapter trainability | 正式训练阶段的模型实现入口；训练由 `experiments/torch_training.py` 调度。                                    |
| `teacher_simulator/`  | Teacher simulator 源码 | 高保真车辆动力学 teacher 的当前 v0/scaffold 实现                                         | 生成 DS0/DS1/DS1 proxy 数据，支撑 sanity、baseline 和 hybrid scaffold 实验。                         |
| `tests/`              | 测试代码                 | teacher simulator、canonical data、student model、PyTorch runner 测试            | 验证数据生成、采样、基础物理逻辑和训练入口没有被破坏。                                                              |

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
| `ds2_extreme_v0.yaml` | DS2 极限操控 scaffold 数据配置，用于 B7 / MoE tire residual 后续实验入口。 |

### `configs/runs/`

每个实验 run 的入口配置。配置中的 `logging.output_dir` 通常指向 `runs/Rxxx_...`；`runs/` 当前不在仓库中，执行实验时会自动创建。

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
| `R046-R047` | DS2 extreme dataset smoke 和 T3-MoE tire residual forward smoke。 |
| `R100-R115` | PyTorch training smoke/dev runs：data loader、forward/loss、tiny overfit、rollout、checkpoint、CUDA gates、one-step train、resume、black-box baseline、小规模 base train、公平对比、组件变体、fine-tune adapter、K=3 ensemble。历史上已通过，产物已清理。 |
| `configs/torch_matrix/` | 由 `experiments.torch_config_matrix` 生成的完整 PyTorch ablation/fine-tune 配置矩阵：`R200-R216` 和 `R300-R334`。配置已生成，等待队列运行。 |

运行方式统一使用 Miniforge/conda 环境：

```bash
conda run -n deep-sim python -m experiments.run --config configs/runs/R009.yaml
```

完整手动操作流程见根目录 `EXPERIMENT_USAGE.md`。

## 数据目录

### `data/`

canonical 数据集入口。这里保存训练和测试直接读取的数据，不再通过软链接指向 `runs/`。

| 路径 | 内容 | 用途 |
| --- | --- | --- |
| `data/README.md` | 数据入口说明和验证命令 | 说明当前 canonical dataset 的来源和使用方式。 |
| `data/ds1_v1/` | DS1 scaffold 数据集，120 个 episode | base training、held-out road / vehicle eval、PyTorch smoke/dev run 的默认数据源。 |
| `data/ds1_proxy_ft_v1/` | DS1 proxy fine-tune 数据集，135 个 episode | FT0-FT6 target time-window / data-efficiency fine-tune 实验的数据源。 |

每个数据集目录通常包含：

| 文件/目录 | 用途 |
| --- | --- |
| `manifest.json` | 数据集索引，列出 episode 文件和 metadata。 |
| `split_manifest.json` | train / validation / test / held-out / fine-tune 等 split 定义。 |
| `episodes/*.npz` | 数值数组，供训练和评估读取。 |
| `episodes/*.json` | episode sidecar metadata。 |
| `scenario_matrix.json` | 生成该数据集使用的工况组合。 |
| `dataset_qa_report.json`、`scenario_coverage_report.json` | 数据质量和覆盖率报告。 |

验证当前数据是否可读：

```bash
conda run -n deep-sim python -m experiments.materialize_data
```

注意：`data/` 是稳定输入目录；新实验的输出不要写进这里，应写入 `runs/`。

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
| `torch_training.py` | PyTorch runner，覆盖 R100-R115 的 data loader、loss、tiny overfit、rollout、checkpoint、CUDA、训练、resume、black-box baseline、公平对比、fine-tune 和 ensemble 检查。 |
| `torch_config_matrix.py` | 生成 trainable PyTorch ablation / fine-tune 配置矩阵。 |
| `torch_dev_report.py` | 聚合 R112-R115 结果到 `reports/PYTORCH_DEV_REPORT.md/json`。 |
| `experiment_queue.py` | 本地批量实验队列。支持读取 run config 或 `configs/torch_matrix/MANIFEST.json`、dry-run、跳过已成功 run、失败重试、逐次日志、队列状态 JSON 和可选 post-rollout eval。 |
| `matrix_report.py` | 汇总 `R200-R216` / `R300-R334` 矩阵运行状态、验证指标和 rollout 指标，生成 `reports/PYTORCH_MATRIX_REPORT.md/json`。 |
| `real_data_adapter.py` | 将 CSV 实车 episode 转换为 canonical dataset，写出 manifest、episode `.npz`、sidecar 和 adapter summary。 |
| `__init__.py` | Python package 标记。 |

`experiments/__pycache__/` 是 Python 自动生成的缓存目录，不属于源码。

### `student_model/`

正式 PyTorch Student Model 的源码入口。

| 文件 | 用途 |
| --- | --- |
| `constants.py` | 状态、控制、context 字段顺序定义。 |
| `data.py` | canonical dataset 读取、episode array 解析、context vector 编码、可选 Torch dataset wrapper。 |
| `torch_model.py` | 当前 final single skeleton：`E2 + T1 + F1 + S1 + M0-fixed + V2-small + U0`。 |

当前限制：这里仍是小规模训练开发链路，不是完整训练结论。训练 loss、optimizer、scheduler、early stopping、best checkpoint、rollout、checkpoint save/load、resume、direct black-box baseline、fair comparison、fine-tune adapter 和 ensemble 已在 `experiments/torch_training.py` 中接入；R100-R115 历史上已通过，但运行产物已清理，需要时按配置重新执行。

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
| `EXPERIMENT_TRACKER.md` | 当前 run 执行跟踪表，记录 DONE/READY 等状态。 |
| `EXPERIMENT_RESULTS.md` | 最新实验结果汇总。 |
| `EXPERIMENT_CODE_REVIEW.md` | 当前实现和实验产物的本地 code review 记录。 |
| `EXPERIMENT_RUN_SPEC.md` | 实验运行规范，包含 conda 环境要求和运行命令约定。 |
| `DATA_DESIGN.md` | 数据集设计细节，包括场景组合、字段、split、采样比例和 fine-tune 数据桶。 |
| `DATA_SCHEMA_SPEC.md` | 数据 schema 规格。 |
| `MODULE_DESIGN.md` | Student model 组件设计，包括 encoder、residual、MuHead、fine-tune adapter 等。 |
| `STUDENT_MODEL_SPEC.md` | Student model 的实现规格。 |
| `TEACHER_SIMULATOR_DESIGN.md` | Teacher simulator 设计文档。 |
| `TEACHER_SIMULATOR_SPEC.md` | Teacher simulator 实现规格。 |
| `PYTORCH_IMPLEMENTATION_STATUS.md` | 从 scaffold 进入 PyTorch 训练实现的状态记录。 |
| `*_2026*.md` | 历史快照或阶段性 checkpoint；当前 timestamped result/review/tracker 快照已清理，只保留最新权威文档。 |

手动运行实验时优先读根目录 `EXPERIMENT_USAGE.md`；需要理解 run 的验收契约时再读 `EXPERIMENT_RUN_SPEC.md`。

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
| `B7_extreme_moe.md` | DS2 extreme 数据和 T3-MoE tire residual 开发 smoke 报告。 |
| `PYTORCH_DEV_REPORT.md/json` | R112-R115 PyTorch 开发 gate 汇总。 |
| `PYTORCH_MATRIX_REPORT.md/json` | R200-R216 ablation 和 R300-R334 fine-tune 矩阵状态/结果汇总。矩阵 run 当前仍是 pending。 |

### `runs/`

每次实验运行的实际输出。目录名通常为 `Rxxx_实验名/`。

当前仓库已经做过运行产物瘦身：历史 R000-R115 产物、timestamped result/review/tracker 快照和 Python `__pycache__` 已删除。`runs/` 目录当前不存在；重新执行实验时，runner 会按配置自动创建新的 `runs/{run_id}_{run_name}/`。

canonical 数据已经移动到 `data/` 下，不再通过 symlink 指向 `runs/`：

| 数据目录 | 用途 |
| --- | --- |
| `data/ds1_v1` | 主 DS1 scaffold 数据集，供 base training / held-out eval 使用。 |
| `data/ds1_proxy_ft_v1` | fine-tune target-window proxy 数据集，供 FT 数据效率实验使用。 |

其余历史 run 的结果摘要仍保留在 `reports/`、`refine-logs/EXPERIMENT_RESULTS.md` 和 `refine-logs/EXPERIMENT_TRACKER.md` 中；如需重新生成某个 run，使用对应的 `configs/runs/Rxxx.yaml`。

每个 run 目录通常包含：

| 子目录 | 用途 |
| --- | --- |
| `artifacts/` | 数据集、指标、summary、metadata、split 等运行产物。 |
| `logs/` | 日志目录。当前 scaffold run 中不一定都有大量日志。 |
| `checkpoints/` | checkpoint 目录。当前多数 run 是 scaffold，没有真实深度模型权重。 |

已完成过、但历史产物已清理的主要 run：

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
| `R046-R047` | DS2 extreme dataset smoke 和 T3-MoE tire residual forward smoke；历史上已通过，产物已清理。 |
| `R100` | PyTorch data loader smoke；历史上已通过，产物已清理。 |
| `R101-R104` | PyTorch forward/loss、tiny overfit、rollout、checkpoint smoke；历史上已通过，产物已清理。 |
| `R105` | PyTorch CUDA-required forward/loss smoke；历史上已通过，产物已清理。 |
| `R106` | PyTorch CUDA-required tiny-overfit smoke；历史上已通过，产物已清理。 |
| `R107-R111` | PyTorch 小规模训练、rollout eval、resume/eval-only、black-box baseline 和 base hybrid small training；历史上已通过，产物已清理。 |
| `R112-R115` | PyTorch fair comparison、组件变体、fine-tune adapter 和 K=3 ensemble 开发 gate；历史上已通过，产物已清理。 |
| `R200-R216` | 真实可训练 PyTorch 单因素 ablation 矩阵；配置已生成，运行状态 pending。 |
| `R300-R334` | FT0-FT6 × FTD1-FTD5 fine-tune 数据效率矩阵；配置已生成，运行状态 pending。 |

注意：`runs/` 是运行结果目录，不是手写源码。当前可以不存在；重新运行实验会重新创建它，实验结束后也可以再次删除。

## 测试目录

### `tests/`

自动化测试目录。

| 文件 | 用途 |
| --- | --- |
| `test_teacher_simulator.py` | Teacher simulator 相关单元测试和回归测试。 |
| `test_canonical_data.py` | canonical dataset 入口和 student-visible array/context 读取测试。 |
| `test_student_model.py` | PyTorch student model forward smoke；缺少 PyTorch 时自动 skip。 |
| `test_torch_training.py` | PyTorch smoke runner/config 测试；缺少 PyTorch 时验证 blocked 路径。 |
| `test_experiment_engineering.py` | experiment queue、matrix report 和 real data adapter 的回归测试。 |
| `__init__.py` | Python package 标记。 |

运行方式：

```bash
conda run -n deep-sim python -m unittest
```

`tests/__pycache__/` 是 Python 自动生成的缓存目录，不属于源码。

## 根目录文件

| 文件 | 用途 |
| --- | --- |
| `MANIFEST.md` | 自动维护的产物清单，按时间记录技能、文件、阶段和说明。 |
| `DIRECTORY_GUIDE.md` | 当前文件，按目录解释仓库结构和用途。 |
| `EXPERIMENT_USAGE.md` | 手动实验操作手册，说明如何运行单个实验、批量矩阵、报告汇总和实车 CSV 适配。 |
| `AGENTS.md` | `/run-experiment` 环境配置说明，记录本地 `deep-sim`、GPU 状态和 smoke 命令。 |
| `environment.yml` | Miniforge/conda 环境定义。实验活动应使用 `deep-sim` 环境。 |
| `requirements.txt` | Python 依赖的轻量记录。PyTorch CUDA build 由 `environment.yml` 通过 conda 管理。 |
| `.gitignore` | Git 忽略规则。 |

## 哪些目录应该重点维护

日常开发和实验时，优先关注：

1. `refine-logs/`：方案、规格、tracker、结果是否同步。
2. `configs/runs/` 和 `configs/teacher/`：实验是否可复现。
3. `experiments/`：实验 runner 和评估逻辑。
4. `teacher_simulator/`：数据生成和 teacher 物理逻辑。
5. `reports/`：阶段性结果是否能支持当前 claim。
6. `tests/`：修改 teacher 或实验逻辑后是否有基本回归测试。
7. `EXPERIMENT_USAGE.md`：手动流程变化后同步更新操作命令。

通常不要手动改：

1. `.git/`：Git 内部数据。
2. `__pycache__/`：Python 缓存。
3. 新生成的 `runs/*` 产物：除非明确是在清理、归档或重新生成实验结果。
