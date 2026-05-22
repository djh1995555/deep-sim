# 目录说明

本文件说明当前仓库各目录的内容和用途。当前状态截至 2026-05-22。

## 顶层目录总览

| 路径                    | 类型                   | 内容                                                                          | 用途                                                                                       |
| --------------------- | -------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `AGENTS.md`           | 运行环境说明               | `/run-experiment` 可读取的本地 conda 环境、GPU 状态和 smoke run 命令                      | 当前声明本地 `deep-sim` CUDA smoke 可用，GPU 为 NVIDIA RTX A4500。                                  |
| `.agents/`            | 内部工具                 | 本仓库随附的 agent skills 定义                                                      | 给 Codex/ARIS 工作流使用，例如 `experiment-bridge`、`research-wiki`、`experiment-plan`。不是车辆动力学模型源码。 |
| `.git/`               | Git 元数据              | 提交历史、对象库、分支引用                                                               | Git 内部目录，不需要手动修改。                                                                        |
| `EXPERIMENT_USAGE.md` | 手动实验使用指南             | 单 run、队列、矩阵报告、实车 CSV 适配、验证命令                                                | 给人工手动执行实验时使用，是最直接的操作入口。                                                                  |
| `configs/`            | 实验配置                 | 每个训练 run 的 YAML 配置                                      | 定义训练实验入口参数，是复现实验的主要配置来源；数据集生成配置由外部 simulator 项目维护。                                                                   |
| `data/`               | 数据入口                 | canonical dataset 真实目录                                                      | 给 PyTorch 训练阶段提供稳定数据路径，当前保留 `ds1_v1` 和 `ds1_proxy_ft_v1` 两个数据集。                          |
| `experiments/`        | 实验执行代码               | runner、sanity、baseline、hybrid、PyTorch smoke、ablation 报告代码                   | `python -m experiments.run --config ...` 的执行入口和实验逻辑。                                     |
| `idea-stage/`         | 早期研究资料               | 文献调研、精读、idea 报告、最终设计决策                                                      | 记录从需求到方案形成的早期推理链，偏研究探索。                                                                  |
| `refine-logs/`        | 方案与规格文档              | 实验计划、模块设计、数据设计、运行规格、结果总结                                         | 当前方案设计和实验执行状态的权威文档目录。                                                                    |
| `output/`             | 运行输出根目录              | 训练实验输出、队列状态和训练报告                                                   | 当前训练工程的新运行产物统一写入这里，避免重新引入 `runs/`。                                                             |
| `research-wiki/`      | 研究知识库                | 论文卡片、idea 卡片、claim/gap/query 记录                                             | 持久化研究知识库，用于追踪证据、文献和想法之间的关系。                                                              |
| `student_model/`      | PyTorch 模型源码         | Student model 的数据接口、常量、E1/E2/E3 encoder、residual heads、adapter trainability | 正式训练阶段的模型实现入口；训练由 `experiments/torch_training.py` 调度。                                    |
| `tests/`              | 测试代码                 | canonical data、student model、PyTorch runner、实验工程测试                       | 验证数据读取、采样、模型 forward、训练入口和队列/报告逻辑没有被破坏。                                                              |

## 配置目录

### `configs/`

实验配置根目录。

### `configs/experiments/`

每个训练实验、数据生成实验、smoke run 和矩阵 run 的入口配置。文件名按实验块和语义命名；配置内部仍保留 `run.id`，例如 `R111`、`R200`，用于历史 gate、队列筛选和报告索引。重新执行实验时会自动创建 `output/training/{run_id}_{run_name}/`。

当前子目录：

| 子目录 | 作用 |
| --- | --- |
| `b0_data_generation/` | B0 数据生成和数据 schema sanity。 |
| `b1_sanity/` | B1 schema、物理一致性、dt 对齐、派生量、tiny learnability 和 physics rollout smoke。 |
| `b2_proxy/` | B2 sim-to-real proxy 扰动、target time window 和分布 sanity。 |
| `b3_baselines_base/` | B3 physics-only、black-box baseline、公平性审计、base hybrid 训练/评估和多 seed 复现。 |
| `b4_ablation_scaffold/` | B4 E/T/F/S/M/V/U 组件 ablation scaffold。 |
| `b5_generalization/` | B5 跨车 / 跨配置泛化 scaffold。 |
| `b6_finetune_scaffold/` | B6 FT0-FT6 target fine-tune 数据效率 scaffold。 |
| `b7_extreme_moe/` | B7 DS2 extreme dataset smoke 和 T3-MoE tire residual forward smoke。 |
| `p3_smoke/` | PyTorch CPU smoke：data loader、forward/loss、tiny overfit、rollout、checkpoint。 |
| `p4_gpu_smoke/` | PyTorch GPU smoke：CUDA forward/loss 和 tiny overfit。 |
| `p5_training_dev/` | PyTorch 训练开发：one-step training、rollout eval、resume/eval-only 和 black-box baseline。 |
| `p6_model_dev/` | PyTorch base model small training、公平对比和组件变体 smoke。 |
| `p7_adapter_ensemble/` | Fine-tune adapter 和 K=3 ensemble smoke。 |
| `ablation/` | `experiments.torch_config_matrix --write` 生成的 R200-R216 可训练 PyTorch 单因素 ablation 配置。 |
| `finetune/` | `experiments.torch_config_matrix --write` 生成的 R300-R334 FT0-FT6 × 数据量矩阵配置。 |
| `matrix/` | 矩阵队列 manifest，当前为 `MANIFEST.json`。 |

运行方式统一使用 Miniforge/conda 环境：

```bash
conda run -n deep-sim python -m experiments.run --config configs/experiments/b3_baselines_base/b3_5_base_hybrid_training.yaml
```

完整手动操作流程见根目录 `EXPERIMENT_USAGE.md`。

## 数据目录

### `data/`

canonical 数据集入口。这里保存训练和测试直接读取的数据，不再通过软链接指向历史运行输出目录。

| 路径 | 内容 | 用途 |
| --- | --- | --- |
| `data/README.md` | 数据入口说明和验证命令 | 说明当前 canonical dataset 的来源和使用方式。 |
| `data/ds1_v1/` | DS1 scaffold 数据集，120 个 episode | base training、held-out road / vehicle eval、PyTorch smoke/dev run 的默认数据源。 |
| `data/ds1_proxy_ft_v1/` | DS1 proxy fine-tune 数据集，135 个 episode | FT0-FT6 target time-window / data-efficiency fine-tune 实验的数据源。 |

数据集目录内容取决于 `scenario_set`。所有生成的数据集至少包含：

| 文件/目录 | 用途 |
| --- | --- |
| `manifest.json` | 数据集索引，列出 episode 文件和 metadata。 |
| `episodes/*.npz` | 数值数组，供训练和评估读取。 |
| `episodes/*.json` | episode sidecar metadata。 |
| `generation_summary.json` | 本次数据生成摘要，包括配置路径、输出路径、episode 数量和 scenario set。 |

`ds1`、`ds1_proxy`、`ds2_extreme` 还会包含：

| 文件/目录 | 用途 |
| --- | --- |
| `split_manifest.json` | train / validation / test / held-out / fine-tune 等 split 定义。 |
| `scenario_matrix.json` | 生成该数据集使用的工况组合。 |
| `dataset_qa_report.json`、`scenario_coverage_report.json` | 数据质量和覆盖率报告。 |

`ds1_proxy` 还会包含 `perturbation_profiles.json`、`proxy_target_windows.json`、`proxy_distribution_report.json`，用于记录 proxy perturbation 和目标时间窗分布。

验证当前数据是否可读：

```bash
conda run -n deep-sim python -m experiments.materialize_data
```

注意：`data/` 是稳定输入目录；新训练实验输出不要写进这里，应写入 `output/training/`。

## 实验代码目录

### `experiments/`

实验执行和评估逻辑。

| 文件 | 用途 |
| --- | --- |
| `run.py` | 统一实验入口。读取 run config，调度外部数据生成、sanity、baseline、hybrid、ablation 等任务。 |
| `sanity.py` | 数据和物理 sanity 检查，包括 schema、dt、派生量、learnability 和 rollout smoke。 |
| `baselines.py` | physics-only baseline 和 black-box baseline scaffold 逻辑。 |
| `hybrid.py` | base hybrid、组件 ablation、评估指标和 residual 审计的 scaffold 实现。 |
| `ablation_report.py` | 汇总 R015-R033 组件 ablation 结果，生成 markdown/JSON 报告。 |
| `torch_training.py` | PyTorch runner，覆盖 R100-R115 的 data loader、loss、tiny overfit、rollout、checkpoint、CUDA、训练、resume、black-box baseline、公平对比、fine-tune 和 ensemble 检查。 |
| `torch_config_matrix.py` | 生成 trainable PyTorch ablation / fine-tune 配置矩阵。 |
| `torch_dev_report.py` | 聚合 R112-R115 结果到 `output/training/reports/PYTORCH_DEV_REPORT.md/json`。 |
| `experiment_queue.py` | 本地批量实验队列。支持读取 run config 或 `configs/experiments/matrix/MANIFEST.json`、dry-run、跳过已成功 run、失败重试、逐次日志、队列状态 JSON 和可选 post-rollout eval。 |
| `matrix_report.py` | 汇总 `R200-R216` / `R300-R334` 矩阵运行状态、验证指标和 rollout 指标，生成 `output/training/reports/PYTORCH_MATRIX_REPORT.md/json`。 |
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

## 外部 Simulator

车辆物理模型和闭环仿真器已从当前训练工程中抽出，独立工程位置：

```text
/home/mi/vibe/research/simulator
```

当前训练工程与 simulator 的关系是：

```text
/home/mi/vibe/research/simulator
  -> 生成多车型、多工况 canonical dataset
  -> 写入当前工程 data/ 或临时输出目录
  -> 当前工程读取 dataset 训练车辆动力学神经网络模型
```

当前工程不再维护 `simulator/` 源码、`configs/simulator/` 仿真 request 或闭环仿真报告文档。需要修改车辆物理模型、controller 闭环仿真、工况生成、debug 可视化时，到 `/home/mi/vibe/research/simulator` 中修改。

数据集生成配置也由外部 simulator 项目维护：

```text
/home/mi/vibe/research/simulator/configs/datasets/
```

当前训练工程的 `configs/experiments/*.yaml` 可以通过绝对路径引用这些外部配置作为数据生成 provenance；训练运行本身主要读取 `data/` 下已经生成好的 canonical dataset。

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

### `output/`

统一运行输出根目录，当前训练工程只维护训练实验输出。

| 路径 | 内容 | 用途 |
| --- | --- | --- |
| `output/training/` | `experiments.run` 单 run 输出、队列状态、队列日志、checkpoint、metrics、post-rollout eval 和训练汇总报告 | 训练实验和数据生成类运行产物的默认位置。 |
| `output/training/reports/` | B0/B3/B4/B5/B6、PyTorch dev report、matrix report 等可再生成报告 | 新的阶段汇总报告默认写入这里。 |

### 历史 `reports/`

根目录 `reports/` 已不再作为新产物目录维护。历史报告可通过对应脚本重新生成到 `output/training/reports/`，例如 B0/B3/B4/B5/B6、PyTorch dev report 和 matrix report。

### `output/training/`

每次训练实验运行的实际输出。目录名通常为 `Rxxx_实验名/`。

当前仓库已经做过运行产物瘦身：历史 R000-R115 产物、timestamped result/review/tracker 快照和 Python `__pycache__` 已删除。重新执行实验时，runner 会按配置自动创建新的 `output/training/{run_id}_{run_name}/`。

canonical 数据已经移动到 `data/` 下，不再通过 symlink 指向历史运行目录：

| 数据目录 | 用途 |
| --- | --- |
| `data/ds1_v1` | 主 DS1 scaffold 数据集，供 base training / held-out eval 使用。 |
| `data/ds1_proxy_ft_v1` | fine-tune target-window proxy 数据集，供 FT 数据效率实验使用。 |

其余历史 run 的结果摘要仍保留在 `refine-logs/EXPERIMENT_RESULTS.md` 和 `refine-logs/EXPERIMENT_TRACKER.md` 中；如需重新生成某个 run，使用对应的 `configs/experiments/<实验块>/<语义化实验名>.yaml`，新结果会写入 `output/training/`。

每个 run 目录通常包含：

| 子目录 | 用途 |
| --- | --- |
| `artifacts/` | 数据集、指标、summary、metadata、split 等运行产物。 |
| `logs/` | 日志目录。当前 scaffold run 中不一定都有大量日志。 |
| `checkpoints/` | checkpoint 目录。当前多数 run 是 scaffold，没有真实深度模型权重。 |

已完成过、但历史产物已清理的主要 run：

| Run 范围 | 内容 |
| --- | --- |
| `R000*` | 外部 simulator 生成 DS0/DS1/DS1 proxy 数据并执行数据导出检查。 |
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

注意：`output/training/` 是训练运行结果目录，不是手写源码。当前可以不存在；重新运行实验会重新创建它，实验结束后也可以再次删除。

## 测试目录

### `tests/`

自动化测试目录。

| 文件 | 用途 |
| --- | --- |
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
2. `configs/experiments/`：训练实验是否可复现；外部数据生成需求在 `/home/mi/vibe/research/simulator/configs/datasets/` 维护。
3. `experiments/`：实验 runner 和评估逻辑。
4. `data/` 和外部 `/home/mi/vibe/research/simulator` 的数据生成契约：训练输入是否与当前 schema/模型匹配。
5. `output/training/reports/`：阶段性结果是否能支持当前 claim。
6. `tests/`：修改外部数据生成接口或实验逻辑后是否有基本回归测试。
7. `EXPERIMENT_USAGE.md`：手动流程变化后同步更新操作命令。

通常不要手动改：

1. `.git/`：Git 内部数据。
2. `__pycache__/`：Python 缓存。
3. 新生成的 `output/training/*` 产物：除非明确是在清理、归档或重新生成实验结果。
