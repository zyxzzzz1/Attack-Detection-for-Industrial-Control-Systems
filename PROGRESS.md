# 进度记录

> 资产映射与漏洞数据来自实验环境，仅用于演示；融合部分不声称提升检测指标（约束 7）。
>
> 本文件中引用的约束和决策编号（包括写作"CLAUDE.md 约束 / 决策 …"的地方）的内容见 `docs/design_decisions.md`。

## 阶段状态

| 阶段 | 内容 | 状态 | 提交 |
|---|---|---|---|
| M7 | OpenVAS 冒烟扫描 + parse_report | 完成 | 9fb7da0 |
| M8 | KEV/EPSS 威胁情报富化 intel.py | 完成 | 3a706c2 |
| M6+M9 | 资产图校验 + 融合打分 + mock 报警 | 完成 | 01fb115 |
| E1 | 提交 M6+M9、pandas Timestamp、PROGRESS.md | 完成 | d64bb25 |
| E2 | M10 告警卡片（Markdown/HTML + 索引页） | 完成 | a5e72ed |
| E3 | 第二台靶机（hmi_main） | 已跳过，仅完成准备部分 | 4365eb7 |
| E4 | 一键流水线 `python -m src.pipeline [--offline]` | 完成 | fd5d109 |
| E5 | Streamlit 演示 `app/streamlit_app.py` | 完成 | 3184e48 |
| E6 | 扩展收尾（README、reports/extension/） | 完成 | E6 提交 |
| M0 | 主线:环境与数据(下载核对、逐文件统计、`reports/main/data_summary.md`) | 完成 | M0 提交 |
| M1 | 预处理与滑窗(`src/data/`、`src/eval/labels.py`、标签隔离测试) | 完成 | M1 提交 |
| C0 | 集群运行包(Minerva):运行入口、manifest、作业脚本、同步脚本、runbook | 完成(2026-10-08 在集群上搭好环境,CPU 与 GPU 上的完整 pytest 均通过) | 6227268 |
| M2a | 评估框架与基线(PCA、IsolationForest) | 完成(2026-10-08:集群作业 274399,主表前两行与 val_thr 诊断已生成) | M2a 集群提交 |
| M2b | LSTM-AE | 完成(2026-10-08:计时作业 274400、完整训练 274401,产物已检查) | M2b/M2c 提交 |
| M2c | LSTM-VAE 与 Deep SVDD | 完成(2026-10-09:集群作业 274566、274567,产物已检查,主表五行齐全) | M2c 集群提交 |
| M3 | 主结果表(5 个模型)与分数曲线图 | 完成(2026-10-09:全部来自集群完整运行) | 56be910 |
| M4 | LSTM-AE 消融 | 完成(2026-10-09:集群作业 274568,消融表与图已生成) | db357d4 |
| M5 | 归因(LSTM-AE、LSTM-VAE、PCA) | 完成(2026-10-09:全部来自集群完整运行;传感器级 Hit@k 未做,原因见 M5 一节) | ef278e7 |
| I1 | 报警导出与集成 | 完成(2026-10-09:真实 alerts.json、融合排序评估、流水线与 Streamlit 默认读真实报警、3 个案例) | c50dec5 |
| R1 | 收尾 | 完成(2026-10-09:`reports/main/` 索引与关键数字、README 主线部分、方法草稿的待填项、本文件顶部的关键数字) | e5a6fed |
| L1 | 界面和报告素材英文化(`ui.lang`:en / zh,默认 en) | 完成(2026-10-09:Streamlit、卡片、索引页、案例、全部图表;数值与中文版一致) | c164de1 |
| L2 | 代码全部英文化:注释、显示文字、提示语、生成的报告;删除 zh 选项与译文表 | 完成(2026-10-25,见下方 L2 一节) | — |

## 关键数字（主线）

写报告时直接引用。数字由 `python -m scripts.make_key_numbers` 从来源文件读出（本节原为 `reports/main/key_numbers.zh.md` 的副本；2026-10-25 起只生成英文版 `key_numbers.md`，本节不再自动同步，数字以英文版为准）；检测、消融、归因的数字全部来自集群上的完整运行（git_dirty=false），冒烟测试的数字没有进入这里。
来源文件全部入库：`reports/main/runs/` 是各次集群运行的小文件（指标、manifest、阈值、预处理摘要、训练记录）的副本，由本脚本从产物目录复制，主机名等集群信息已替换为占位符。

### 检测

| 数字 | 口径 | 来源 |
|---|---|---|
| **不使用 point-adjust**；阈值 = val_thr 分数的 99% 分位数；报警规则为分数 > 阈值 | 全部检测指标的共同口径；逐点 F1 偏低是预期结果 | CLAUDE.md 约束 3、4，决策 6b |
| pca：precision 0.0946，recall 0.8405，F1 0.1700，AUROC 0.9014；事件 48/50；延迟中位数 6 秒；误报 7.16 次/小时（781 段） | 单个随机种子（42）；逐点指标排除未打分的 295 秒；集群完整运行 | `reports/main/main_results.csv` |
| iforest：precision 0.4144，recall 0.4828，F1 0.4460，AUROC 0.8347；事件 35/50；延迟中位数 28 秒；误报 6.84 次/小时（746 段） | 单个随机种子（42）；逐点指标排除未打分的 295 秒；集群完整运行 | `reports/main/main_results.csv` |
| lstm_ae：precision 0.2765，recall 0.8040，F1 0.4115，AUROC 0.9339；事件 50/50；延迟中位数 6 秒；误报 14.95 次/小时（1631 段） | 单个随机种子（42）；逐点指标排除未打分的 295 秒；集群完整运行 | `reports/main/main_results.csv` |
| lstm_vae：precision 0.0606，recall 0.8111，F1 0.1128，AUROC 0.8800；事件 48/50；延迟中位数 4 秒；误报 21.22 次/小时（2315 段） | 单个随机种子（42）；逐点指标排除未打分的 295 秒；集群完整运行 | `reports/main/main_results.csv` |
| deep_svdd：precision 0.1683，recall 0.8287，F1 0.2797，AUROC 0.9055；事件 49/50；延迟中位数 3 秒；误报 69.33 次/小时（7564 段） | 单个随机种子（42）；逐点指标排除未打分的 295 秒；集群完整运行 | `reports/main/main_results.csv` |
| **LSTM-AE 5 个种子**：F1 **0.4290 ± 0.0184**，AUROC **0.9335 ± 0.0016**，事件检出率 **0.9920 ± 0.0110**（平均 49.6/50），误报 **13.19 ± 1.68** 次/小时；precision 0.2929 ± 0.0168，recall 0.8026 ± 0.0055 | 主配置 W=60、hidden=64；种子 42, 43, 44, 45, 46；均值 ± 样本标准差（ddof=1）。报告主表中 LSTM-AE 一行用这组数；其余模型只有单种子 | `reports/main/main_results_seeds.csv`、`main_results.md` |
| LSTM-AE 与 IsolationForest 的逐点 F1：0.4290 对 0.4460，相差 0.0170 = 0.93 个标准差；5 个种子中 1 个高于 IsolationForest | 差距在随机波动范围内；IsolationForest 自己的种子波动没有评估 | 同上 |
| test 上报警的秒占比：pca 19.8%，iforest 2.6%，lstm_ae 6.5%，lstm_vae 29.8%，deep_svdd 11.0% | 已打分的秒中分数 > 阈值的比例；val_thr 上按阈值的定义为 1%。所有模型都偏高（阈值迁移问题） | `reports/main/runs/<模型>/eval_metrics.json` |
| iforest 分子系统事件检出：P1 31/34，P2 12/22，P3 4/8 | 事件涉及多个子系统时在每个子系统下各计一次 | `reports/main/per_subsystem_detection.csv` |
| lstm_ae 分子系统事件检出：P1 34/34，P2 22/22，P3 8/8 | 事件涉及多个子系统时在每个子系统下各计一次 | `reports/main/per_subsystem_detection.csv` |
| 消融 9 个组合：F1 0.084~0.617，AUROC 0.899~0.946，误报 0.33~74.66 次/小时；最高 W=60、hidden=32（F1 0.617），最低 W=60、hidden=128（F1 0.084）；主模型配置按 F1 排第 5 | LSTM-AE，W ∈ {30, 60, 120} × hidden ∈ {32, 64, 128}，单种子；只作敏感性分析，不据此修改主模型（约束 9） | `reports/main/ablation.csv`、`ablation.md`、`ablation_heatmap.png` |
| 阈值分位敏感性（主模型配置）：99% 分位 F1 0.411、事件检出率 1.00，99.5% 分位 F1 0.629、事件检出率 0.96，99.9% 分位 F1 0.682、事件检出率 0.84；99.9% 分位下 9 个组合的 F1 为 0.63~0.68 | 只作敏感性分析，不据此选阈值；主表仍用 99% 分位 | `reports/main/ablation_quantiles.csv` |
| lstm_ae 训练：50 个 epoch，最好第 48 个，未触发早停；每个 epoch 平均 1.18 秒；参数 68,473 个 | 集群 NVIDIA L4，种子 42；max_epochs=50、patience=5 事先固定 | `reports/main/runs/lstm_ae/run_manifest.json`、`train_log.csv`；图 `reports/main/training_curves.png` |
| lstm_vae 训练：50 个 epoch，最好第 50 个，未触发早停；每个 epoch 平均 1.36 秒；参数 64,441 个 | 集群 NVIDIA L4，种子 42；max_epochs=50、patience=5 事先固定 | `reports/main/runs/lstm_vae/run_manifest.json`、`train_log.csv`；图 `reports/main/training_curves.png` |
| deep_svdd 训练：18 个 epoch，最好第 13 个，第 18 个 epoch 早停；每个 epoch 平均 1.01 秒；参数 33,024 个 | 集群 NVIDIA L4，种子 42；max_epochs=50、patience=5 事先固定 | `reports/main/runs/deep_svdd/run_manifest.json`、`train_log.csv`；图 `reports/main/training_curves.png` |
| Deep SVDD 中心：替换前 \|c\| 最小 0.00025、中位数 0.0102、最大 0.0366；32/32 维被替换为 ±0.1。坍塌判据 collapsed=false（score_cv 236，被 1 个极端窗口拉高）；val_thr 分数中位数 1.7e-05，阈值 7.8e-05 | 中心 = 初始网络在全部训练窗口上输出的均值，接近 0 的分量设为 ±eps（决策 5、6b） | `reports/main/runs/deep_svdd/run_manifest.json`、`threshold.json` |
| PCA 在 val_thr 上占比最大的传感器：P3_FIT01 9.0%（上限 30%，未超限） | 只用 val_thr 的重尾诊断，在第一次评估 test 之前确定缩放方式（StandardScaler，不设 std 下限） | `reports/main/val_sensor_contrib.csv`、`.png` |

### 归因

| 数字 | 口径 | 来源 |
|---|---|---|
| lstm_ae：报警段 54/72 = 0.750 [0.639, 0.836]；事件·第一个报警段 43/50 = 0.860 [0.738, 0.930]；事件·合并 45/50 = 0.900 [0.786, 0.957]；同一批报警段上总是猜 P1 的命中率 0.681；Top-1 在 P4 6/72 | 子系统命中 = Top-1 传感器所属子系统 ∈ 事件的子系统集合；只计与攻击重叠的报警段；方括号为 95% Wilson 区间。按事件：主口径取事件的第一个重叠报警段，合并口径为补充。单种子 | `reports/main/attribution_summary.csv`、`.md`、`attribution.csv` |
| lstm_vae：报警段 52/57 = 0.912 [0.811, 0.962]；事件·第一个报警段 47/48 = 0.979 [0.891, 0.996]；事件·合并 47/48 = 0.979 [0.891, 0.996]；同一批报警段上总是猜 P1 的命中率 0.702；Top-1 在 P4 4/57 | 子系统命中 = Top-1 传感器所属子系统 ∈ 事件的子系统集合；只计与攻击重叠的报警段；方括号为 95% Wilson 区间。按事件：主口径取事件的第一个重叠报警段，合并口径为补充。单种子 | `reports/main/attribution_summary.csv`、`.md`、`attribution.csv` |
| pca：报警段 37/53 = 0.698 [0.565, 0.805]；事件·第一个报警段 36/48 = 0.750 [0.612, 0.851]；事件·合并 34/48 = 0.708 [0.568, 0.818]；同一批报警段上总是猜 P1 的命中率 0.679；Top-1 在 P4 0/53 | 子系统命中 = Top-1 传感器所属子系统 ∈ 事件的子系统集合；只计与攻击重叠的报警段；方括号为 95% Wilson 区间。按事件：主口径取事件的第一个重叠报警段，合并口径为补充。单种子 | `reports/main/attribution_summary.csv`、`.md`、`attribution.csv` |
| 与多个事件重叠的报警段 0 个；无子系统标签事件 0 个；不与攻击重叠的报警段：lstm_ae 1631，lstm_vae 2315，pca 781 | 主口径与并集口径因此完全相同；不与攻击重叠的报警段不计入命中率 | `reports/main/attribution_summary.csv` |
| **LSTM-AE 5 个种子**：报警段 **0.777 ± 0.033**，事件·第一个报警段 **0.875 ± 0.026**，事件·合并 0.919 ± 0.014；Top-1 在 P4 的比例 0.114 ± 0.031 | 每个种子的运行各算一遍归因；均值 ± 样本标准差。LSTM-VAE、PCA 只有单种子 | `reports/main/attribution_seeds.csv` |
| lstm_ae 按报警段长度：<10 秒 7/20 = 0.350 [0.181, 0.567]；10–60 秒 1/4 = 0.250 [0.046, 0.699]；>60 秒 46/48 = 0.958 [0.860, 0.988] | 与攻击重叠的报警段按长度分组；10–60 秒含两端。短报警段归因明显更差（LSTM-AE） | `reports/main/attribution_by_duration.csv` |
| lstm_vae 按报警段长度：<10 秒 2/4 = 0.500 [0.150, 0.850]；10–60 秒 0/3 = 0.000 [0.000, 0.561]；>60 秒 50/50 = 1.000 [0.929, 1.000] | 与攻击重叠的报警段按长度分组；10–60 秒含两端。短报警段归因明显更差（LSTM-AE） | `reports/main/attribution_by_duration.csv` |
| pca 按报警段长度：<10 秒 2/5 = 0.400 [0.118, 0.769]；10–60 秒 1/2 = 0.500 [0.095, 0.905]；>60 秒 34/46 = 0.739 [0.597, 0.844] | 与攻击重叠的报警段按长度分组；10–60 秒含两端。短报警段归因明显更差（LSTM-AE） | `reports/main/attribution_by_duration.csv` |
| （探索性）lstm_ae 按 val_thr 归一化后：报警段 54/72 → 61/72；事件（第一个报警段） 43/50 → 43/50；事件（合并） 45/50 → 46/50 | **事后分析，不替换主结果**：段内误差先除以各传感器在 val_thr 上的平均误差再排 Top-k；基线只来自 val_thr，但这一步是看过测试结果后才做的，数字偏乐观 | `reports/main/attribution_exploratory_valnorm.csv` |
| （探索性）lstm_vae 按 val_thr 归一化后：报警段 52/57 → 52/57；事件（第一个报警段） 47/48 → 44/48；事件（合并） 47/48 → 44/48 | **事后分析，不替换主结果**：段内误差先除以各传感器在 val_thr 上的平均误差再排 Top-k；基线只来自 val_thr，但这一步是看过测试结果后才做的，数字偏乐观 | `reports/main/attribution_exploratory_valnorm.csv` |
| （探索性）pca 按 val_thr 归一化后：报警段 37/53 → 44/53；事件（第一个报警段） 36/48 → 41/48；事件（合并） 34/48 → 42/48 | **事后分析，不替换主结果**：段内误差先除以各传感器在 val_thr 上的平均误差再排 Top-k；基线只来自 val_thr，但这一步是看过测试结果后才做的，数字偏乐观 | `reports/main/attribution_exploratory_valnorm.csv` |
| 传感器级 Hit@k：未做 | HAI 21.03 的数据文件没有逐传感器的攻击目标，官方攻击目标表未取得；写进局限性 | `reports/main/attribution_summary.md` |

### 融合（原型演示；不声称提升检测指标，约束 7）

| 数字 | 口径 | 来源 |
|---|---|---|
| 真实报警 1703 条，其中 72 条与攻击事件重叠（基准比例 0.042） | LSTM-AE 主模型运行（种子 42）在 test 上的全部报警段，不按标签筛选；is_mock=false | `data/processed/alerts.json`、`reports/main/fusion_ranking_eval.csv` |
| 排序 raw_score：precision@10 = 10/10，precision@20 = 20/20，precision@50 = 45/50；precision@50 的 95% 区间 [0.786, 0.957] | 前 k 条报警中与攻击事件重叠的比例；raw_score 按最大异常分数，alpha=0 / alpha=1 为融合优先级。三种排序没有可测量的差别：排序的作用来自异常分数，不来自漏洞上下文 | `reports/main/fusion_ranking_eval.csv`、`.md` |
| 排序 alpha=0：precision@10 = 10/10，precision@20 = 20/20，precision@50 = 45/50；precision@50 的 95% 区间 [0.786, 0.957]；前 50 条中 46 条与原始分数排序相同 | 前 k 条报警中与攻击事件重叠的比例；raw_score 按最大异常分数，alpha=0 / alpha=1 为融合优先级。三种排序没有可测量的差别：排序的作用来自异常分数，不来自漏洞上下文 | `reports/main/fusion_ranking_eval.csv`、`.md` |
| 排序 alpha=1：precision@10 = 10/10，precision@20 = 20/20，precision@50 = 45/50；precision@50 的 95% 区间 [0.786, 0.957]；前 50 条中 44 条与原始分数排序相同 | 前 k 条报警中与攻击事件重叠的比例；raw_score 按最大异常分数，alpha=0 / alpha=1 为融合优先级。三种排序没有可测量的差别：排序的作用来自异常分数，不来自漏洞上下文 | `reports/main/fusion_ranking_eval.csv`、`.md` |
| S 封顶的报警 61 条 | S = min(1, 最大分数 ÷ 阈值 − 1)，分数达到阈值 2 倍即为 1；这些报警之间不再按分数区分。只写成局限和未来工作，不改公式 | `data/processed/alerts.json`、`reports/main/cases.md` |
| 卡片与演示只渲染优先级最高的前 50 条 | 融合打分仍对全部报警进行；条数为 `fusion.max_rendered_alerts` | `config/config.yaml` |
| 3 个案例：归因正确的真实攻击、归因错误的真实攻击、高优先级的误报 | 按融合优先级从前往后各取第一条，不人工挑选；对照标签的说明来自事后评估 | `reports/main/cases.md`、`cases/` |
| 漏洞与资产侧的数字（540 条漏洞、KEV 命中 9 条、min_qod=70 后保留 66 条、E_sub 只有 0 / 1 两个取值） | 真实扫描只有一台靶机 ews_01；资产映射为人为构造 | `reports/extension/key_numbers.md`、`kev_table.md` |

### 数据问题

| 数字 | 口径 | 来源 |
|---|---|---|
| 官方 train1 末尾 43,201 秒与 test3 开头 43,201 秒是同一份数据；另有 train1 第 1 行 = test2 最后 1 行。两处都已从训练集删除，test 保持原样 | 按 79 个传感器的逐行内容哈希比较全部文件对；这段数据没有攻击。不处理会低估 test3 上的误报 | `reports/main/data_summary.md` 第 3 节；`config/config.yaml` 的 `ml.data.train_overlap_drops` |
| 划分：train 702,721 / val_es 87,840 / val_thr 87,840 行；特征 57 个（删除 22 个常量列） | 删除重叠行之后按时间切分，最后 20% 对半分为 val_es（早停）和 val_thr（定阈值）；scaler 只在 train 部分 fit | `reports/main/runs/lstm_ae/preprocess.json`；`PROGRESS.md` 的 M1 一节 |
| test：402,005 秒，攻击 8,947 秒（2.23%），50 个攻击事件（长度 17~422 秒） | 事件 = attack=1 的连续区间 | `reports/main/data_summary.md`、`attack_events.csv` |
| 未打分 295 秒（5 个文件 × 59 秒），其中攻击秒 0 个 | 每个测试文件的前 W−1 秒没有分数（W=60），不参与任何指标；W=30 / 120 的消融组合为 145 / 595 秒 | `reports/main/main_results.csv` 的 n_unscored 列 |
| attack=1 但三个子系统标签全为 0 的有 179 秒，分布在 12 个事件内部；按并集规则 50 个事件的子系统集合全部非空 | 事件的子系统集合 = 事件内所有秒的 attack_P* 并集；检测指标只用 attack 列，不受影响 | `reports/main/attack_events.csv`、`data_summary.md` |
| LSTM-AE 各测试文件报警的秒占比：test1 4.9%，test2 3.6%，test3 2.9%，test4 20.8%，test5 9.0% | val_thr 上为 1%。各文件差别很大，推测 test 的正常工况相对训练数据有漂移（未单独验证是哪些传感器） | `reports/main/runs/lstm_ae/eval_metrics.json`；图 `reports/main/scores_test*.png`、`score_distribution.png` |
| P4 没有子系统标签 | P4 是 HIL 仿真部分；Top-1 落在 P4 的归因必然计为未命中 | `data/samples/hai_columns.txt`、决策 7 |

### 写报告时必须一并说明的事项

- 逐点 F1 偏低是不使用 point-adjust 的预期结果;评估方法、阈值分位、主模型配置都没有为提高指标而改动,没有在 test 上调参。
- 与预期不符的结果(都已如实记录在下文各节):LSTM-AE 的逐点 F1 没有超过 IsolationForest(差距在随机波动内);LSTM-VAE、Deep SVDD 不如 LSTM-AE,
  LSTM-VAE 的 F1 低于 PCA;主模型配置在消融中按 F1 排第 5;主模型的归因命中率低于 LSTM-VAE;融合排序相对原始分数没有可测量的差别。
- 各处给出的"可能的原因"(正常工况漂移、短报警段被背景误差主导等)是推测,除"短报警段归因更差"一条有分组数据支持外,都没有单独验证。
- 用于报警导出、归因和图的主模型运行(种子 42)是 5 个种子里 F1 最低的一个;"50/50 事件全部检出"只在 3/5 个种子上成立。
- 未做 / 局限:传感器级 Hit@k(没有官方攻击目标表);只有 LSTM-AE 主配置有多种子结果;单一数据集;只有一台靶机被扫描;
  S 在 2 倍阈值处封顶(只写成局限和未来工作);Streamlit 没有做目视检查。
- 来源可查(2026-10-09):关键数字引用的各次运行的小文件(指标、manifest、阈值、预处理摘要、训练记录)已复制到 `reports/main/runs/` 并入库
  (18 次运行:5 个模型、9 个消融组合、4 个额外种子;共 88 个文件、约 0.6 MB)。复制由 `python -m scripts.make_key_numbers` 完成
  (内部调用 `scripts.snapshot_runs`),关键数字改为从这份入库副本读取;分数、逐传感器误差、模型权重、checkpoint 和日志仍不入库。
  - 脱敏(仓库可能公开):`run_manifest.json` 中键名为 hostname、node 的值(计算节点主机名,共 90 处)替换为 `<cluster-node>`;
    另按 `cluster.yaml` 的用户名、集群路径、登录主机名和 `/data/users/<名字>`、`/home/<名字>` 形式的路径做字符串替换(这些在 88 个文件里实际没有出现)。
    产物目录里的原件不动。
  - 测试核对:入库副本 = 对 `artifacts/` 中原件做同样脱敏的结果;除 manifest 外的 70 个文件与原件逐字节相同;
    入库副本中不出现用户名、集群路径、主机名。原件不在时(集群、全新克隆)这项核对跳过。
- 仓库脱敏(2026-10-09,为将来公开做准备;不改任何计算和结果):
  - 登录用户名和集群上的绝对路径从入库的 `config/cluster.yaml` 移到不入库的 `config/cluster.local.yaml`(示例 `cluster.local.example.yaml`);
    入库的 `cluster/*.sbatch`、`cluster_env.sh` 按 TODO 路径生成(只用于本地 --smoke 和测试)。
  - 真实的作业脚本由 `python -m scripts.make_cluster_jobs` 生成到不入库的 `cluster/local/`,内容与脱敏前入库的版本逐字节相同;
    `push_to_cluster.sh` 推送代码后把它们放到集群的 `cluster/` 下,所以集群上的命令(`sbatch cluster/train_one.sbatch ...`)不变。
    `push_to_cluster.sh`、`pull_artifacts.sh` 从 `cluster/local/cluster_env.sh` 读取集群路径。
  - **没有处理的**:git 提交历史(b28454f 起的提交里仍有用户名和路径,作者邮箱也在历史里;这些提交还没有推送到远端)。
    重写历史会改变所有提交号,而 run_manifest.json 和本文件引用了这些提交号,需要单独决定。集群名称 Minerva / Chalmers 仍出现在文档中。
  - 推送逻辑的这一改动只做了本地测试(生成结果、TODO 时拒绝、dirty 检查),**没有在集群上实际推送过**;下次推送时先用 `--dry-run` 看一遍。

### L2 代码全部英文化(2026-10-25)

- 代码、模板、配置、测试里不再有中文:注释和 docstring、显示文字、终端输出、报错、生成报告用的句子模板全部改为英文;`tests/test_english_only.py` 检查这一点。
- 删除 `ui.lang` / `UI_LANG`、`src/utils/i18n.py`、`config/i18n_en.yaml`、`scripts/i18n_keys.py`、`tests/test_i18n.py` 和 `key_numbers.zh.md`;免责声明统一为 `src/utils/disclaimer.py` 中的常量。asset_map 与 mock 报警中的显示文字改为英文。
- 用本地已同步的集群产物重新生成了 `reports/` 下的表和文档:数值与改动前逐项一致(只有措辞中的个别数字写法不同),PNG、alerts.json 和案例卡片没有变化。
- 保留中文的:`data/samples/hai_columns.txt` 的表头(生成它的脚本已改为英文,重新生成时表头会变)、`data/samples/smoke_report.xml`(OpenVAS 原始报告)、各中文文档(本文件、`docs/`、`*.zh.md`)。
- 下面 L1 一节描述的 zh / en 切换机制已删除,仅作历史记录。

### L1 界面和报告素材英文化(2026-10-09,R1 之后新增的一步)

- **配置项** `config.yaml` 的 `ui.lang`(en / zh,**默认 en**),环境变量 `UI_LANG` 可临时覆盖。中文版本保留:设为 zh 即可。
- **做法**:代码和模板里的显示文字以中文为源写成 `t("中文")` / `{{ _("中文") }}`,zh 原样输出,en 查译文表 `config/i18n_en.yaml`(202 条)。
  带数字的文字用占位符,数字不进译文。英文缺译时直接报错,不会悄悄退回中文。实现在 `src/utils/i18n.py`;`python -m scripts.i18n_keys` 检查译文表有没有缺少或多余的键。
- **范围**:Streamlit 界面、告警卡片模板(md / html)、索引页、`cases.md`、免责声明和各类提示语、`reports/main/` 与 `reports/extension/` 下全部 14 张图的标题、坐标轴、图例。
  资产映射里的子系统名称、说明、资产角色和 mock 报警的场景说明在显示时翻译,`asset_map.yaml` 和 JSON 本身不改。
- **免责声明**逐句对应,没有弱化:"Asset mapping and vulnerability data come from a lab environment and are for demonstration only."
- **没有改动的**:任何计算、字段名、数字;代码注释仍是中文。`reports/main/` 下的各张表(csv)和 Markdown 文档(主表、归因汇总、关键数字、融合排序评估等)
  以及 `reports/extension/` 的两份 Markdown 不在这次范围内,仍是中文;命令行日志也仍是中文。
- **数值一致性的核对**:
  - 改模板之前先用旧代码渲染了一份中文卡片;改完后的中文输出与之逐字节相同(只有索引页 html 里一处引号的转义写法不同),`cases.md` 的中文版也逐字节相同。
  - 真实报警前 50 条和 mock 报警的卡片、索引页(共 114 个文件),中文版与英文版里出现的数字完全相同;重新生成的英文 `cases.md` 和三张案例卡片
    与之前入库的中文版相比数字完全相同(各 256~280 个数字),没有中文残留。
  - 重新生成图表后,`reports/` 下只有 14 张 PNG、`cases.md` 和三张案例卡片有变化,所有 csv / json / 其他 md 没有变化。
  - 测试(新增 10 个,共 230 个,全部通过):融合打分的 JSON 在两种语言下除免责声明外逐项相同;卡片、索引页、案例、Streamlit 表格里的数字相同;
    同一批产物生成的 csv / json 逐字节相同,图上的文字对象数量相同、其中的数字相同;译文表与代码一一对应、占位符和数字不增不减。
    已有测试断言的是中文文字,所以测试里默认把显示语言固定为 zh(`tests/conftest.py`),英文由 `tests/test_i18n.py` 专门覆盖。
- 英文较长造成的版式调整(只影响显示):资产拓扑图里方框内的文字一行放不下时折行。
- 已用默认语言(英文)重新生成:14 张图、`reports/cards/`(前 50 条)、`reports/cards_mock/`、`reports/main/cases.md` 与三张案例卡片。
  和以前一样,Streamlit 只做了无界面的自动测试,**没有目视检查**;截图前请实际打开看一遍。

#### L1 目视检查后的显示修正(2026-10-09;只改显示,不改计算)

- **告警列表**(Streamlit 的列表和索引页 md / html):S 前面新增 "Score / threshold"(最大分数 ÷ 阈值,两位小数);"priority" 改为 "Priority";
  "Needs review" 改为 "KEV vulns needing review"(中文"需核实的 KEV 漏洞",资产表里的同名列一并改);子系统一列不显示权重按显示精度四舍五入后为 0% 的子系统
  (前 50 条真实报警中有 3 处;卡片的"推断受影响的子系统"一节和 JSON 不变)。
- **并列时的排序规则**:原来就有确定的规则,没有改,只是在表格下方写明——按 priority(未四舍五入的值)降序,相同时按 S 降序,仍相同时按告警编号升序
  (`prioritize()` 的排序键)。Streamlit、卡片、索引页、`cases.md` 用的都是 `prioritize()` 给出的同一个 rank。
  真实报警 1,703 条中 priority 完全相等的有 2 组共 11 条(2.000 的 9 条、1.800 的 2 条,都在前 50 条内),组内 S 都相同,实际由告警编号(即时间先后)决定。
  **没有**改成按原始分数降序:那样会让 11 条报警的排名变化,案例 1(LSTMAE-00112)从第 1 变为第 3。
- **E_sub 的标注**:路径上的资产全部未扫描时写 "0.000 (not assessed: no upstream asset scanned)"(中文"0.000（未评估：上游资产均未扫描）"),
  "possibly underestimated" 只用于部分资产未扫描的情况。判断只在显示层做(`render_card.exposure()`),JSON 字段不变。
  Streamlit 列表里 E_sub 一列保持数值(便于排序),文字写在 "Path risk" 一列。卡片"可能的入侵路径"下的那句提示语是已定决策的原文,没有改。
- `.streamlit/config.toml`:`client.toolbarMode = "minimal"`,隐藏 Deploy 按钮。
- 测试新增 8 个,共 238 个,全部通过。重新生成了 `reports/cards/`、`reports/cards_mock/` 和三张案例卡片;三个案例的排名不变(1 / 50 / 28),`cases.md` 没有变化。
  三张案例卡片里另有一处与这次修改无关的变化:P4 的名称由 "turbine power generation model" 变为 "turbine generator model"——译文表在 c164de1 中已经是后者,入库的卡片当时没有重新生成。
- 修掉了一个不稳定测试:`test_fusion_output_and_cards_have_identical_numbers_in_both_languages` 比较的数字里原来包含生成时间,
  中英文两次渲染恰好跨过整秒时会失败;现在渲染前把生成时间换成固定值,不参与比较。
- 目视检查后的四项决定(2026-10-09):并列规则不改;入侵路径下的 "may be underestimated" 提示语保留;Streamlit 列表的 E_sub 保持数值列;修掉上面的不稳定测试。

#### L1 截图前的第二轮显示修正(2026-10-09;只改显示,不改计算)

- **资产拓扑图**:没有 IP 的资产只写角色("Historian"、"PLC"),不再写 " · -"。顺带删掉了 `make_extension_report.py` 里一段被后面同名函数覆盖、从未执行的残留代码。
- **P4 的 E_sub**(Streamlit 子系统表):写 "0.000 (no control chain: HIL simulation)"(中文"0.000（无控制链：HIL 仿真）"),不再写"未评估"——P4 没有任何资产,不是未评估。
- **子系统表**改用 `st.table`,单元格自动换行,"Unassessed upstream" 一列不再被截断。
- **P4 的名称**:对照 HAI 官方说明(GitHub 仓库 README)——四个过程是 "boiler process, turbine process, water treatment process and HIL simulation";
  HIL 仿真器 "emulates steam-turbine power generation and pumped-storage hydropower generation"。原来的"汽轮机发电模型 / turbine generator model"漏掉了抽水蓄能发电
  (21.03 的 P4 列里也有 P4_HT_* 一组),与官方描述不一致。名称改为官方的过程名:"HIL simulation"(中文"HIL 仿真",`asset_map.yaml` 的 name 一并改,只是显示名称)。
- **P4 的说明补充 P3**(同日定):官方说明里水处理过程(P3)的泵和阀由抽水蓄能发电模型控制,P4 与 P1、P2、P3 都有耦合。
  `asset_map.yaml` 中 P4 的 note 改为建议同时排查 P1/P2/P3 相关资产,`cases.md` 里关于案例 2 的那条观察同步修改。
- **卡片里 P4 的 E_sub** 与子系统表统一:"E_sub = 0.000 (no control chain: HIL simulation)"(原为 "E_sub = 0 (no associated assets)")。
- 拓扑图的版式(P3 的名称略宽于方框、ews_01 的文字贴近下边框)不改。**界面到此不再调整。**
- 测试新增 2 个,共 240 个,全部通过。重新生成了拓扑图、`reports/cards/`、`reports/cards_mock/` 和案例卡片(案例 2、3 的卡片里只有 P4 的名称变化;排名不变,`cases.md` 不变)。

## 主线(M0 → R1)

分工:本地虚拟机只做开发、单元测试和 1% 数据的冒烟测试;完整训练、打分、评估、消融都在学校集群上跑。
主表、消融、归因的数字只来自集群上的完整运行。

### M0 环境与数据(2026-10-07)

- 本地:i5-13500H(VMware,4 核)、内存 7.7 GiB(可用约 3.9 GiB)、磁盘剩 7.6 G、无 GPU。
  torch 2.14.1+cpu、scikit-learn 1.7.2,site-packages 中没有 nvidia-* 包。
- 数据:`python -m scripts.download_hai` 下载 8 个文件共 187.2 MB,大小和 git blob sha 都与 GitHub contents API 一致
  (核对记录在数据目录的 download_manifest.json)。`--verify-only` 可在集群上复用。
- 统计:`python -m scripts.hai_data_summary` → `reports/main/data_summary.md`(约 15 秒,峰值内存约 660 MB)。
  - train 921,603 行(train1 216,001 / train2 226,801 / train3 478,801),test 402,005 行;列全部与 hai_columns.txt 一致。
  - 每个文件内部逐秒连续,无缺秒、重复、乱序、缺失值。train 文件之间间隔 442 小时和 33 小时。
  - test 共 8,947 攻击秒(2.23%)、50 个攻击事件(test1~5:5 / 20 / 8 / 5 / 12),事件长度 17~422 秒。
  - 22 个传感器在三个 train 文件中都是常量(P1 8 个、P2 12 个、P3 2 个),M1 删除后预计剩 57 个特征。
  - 验证集(最后 20% = 184,321 行)全部落在 train3 内。
- **发现:train1 与 test3 有 12 小时是同一份数据。** train1 的最后 43,201 秒(2020-07-13 00:00:00 ~ 12:00:00)
  与 test3 的前 43,201 秒时间戳相同、79 个传感器取值逐行完全相同,这段没有攻击。另外 test2 的最后 1 秒与 train1 的第 1 秒相同。
  不处理的话,test3 前 40% 的正常数据在训练时被见过,误报率会被低估。
- 标签:`attack`=1 但三个子系统标签全为 0 的有 179 秒(test2 154 / test3 3 / test5 22);同时命中多个子系统的有 1,091 秒;
  子系统标签在事件内断续出现(最短 1 秒),所以事件只按 `attack` 的连续区间定义。
- 新增 `config.yaml` 的 `ml:` 段、`src/utils/paths.py`(HAI_DATA_DIR / ARTIFACT_DIR 覆盖)、`src/utils/rowhash.py`、11 个测试(共 92 个,全部通过)。

- **重叠检查扩大到全部文件(2026-10-07)**:按 79 个传感器的逐行内容哈希比较 28 对文件(含时间错位的部分重叠)。
  结果与按时间戳检查一致,没有新的大段重叠:只有 train1×test3 的 43,201 行和 train1 第 1 行 = test2 最后 1 行;
  train×train、test×test 均为 0,文件内部也没有内容重复的行。注意同一列在不同文件里可能解析成 int64 或 float64,
  哈希前必须统一转 float64(否则连 train1×test3 都查不出来)。
- **已决定**:训练集去掉 train1 末尾 43,201 秒,test 保持原样;M1 加 config 开关(默认开启),删除后再判常量列、再 fit scaler,
  测试按内容哈希断言训练窗口不含 test3 前 43,201 秒的行。
- **179 秒无子系统标签的分类**:分布在 12 / 50 个事件中,全部在事件内部断续出现(不贴首尾),没有整个事件无标签的情况。
  按"事件子系统集合 = 事件内 attack_P* 并集"的规则,50 个事件并集全部非空(P1 23、P2 11、P3 2、P1+P2 8、P1+P3 3、P2+P3 3),
  "无子系统标签事件"为 0 个。明细见 `reports/main/attack_events.csv`。

### M1 预处理与滑窗(2026-10-07)

- 代码:`src/data/load_hai.py`(读取、删重叠行、切分)、`preprocess.py`(常量列 + scaler)、`windowing.py`(按需切片的滑窗)、
  `src/eval/labels.py`(标签唯一入口)。摘要命令:`python -m src.data.preprocess [--fraction 0.01]`。
- 数据组织成"段"(逐秒连续的一块):文件边界、train/val 切分点、val 内部切分点都是段边界,滑窗只在段内进行。
  标签列不在读取的 usecols 里,src/data/ 从不读入标签;时间列只用于核对逐秒连续和记录段的起始时间。
- 处理顺序:删除与 test 相同的行 → 切分 → 在 train 部分判定常量列 → fit scaler。
- 划分(删除 train1 首 1 行 + 末 43,201 行后共 878,401 行):train 702,721 / val_es 87,840 / val_thr 87,840 行;
  val_es、val_thr 都在 train3 内(切分点 2020-08-08 10:12:01 和 2020-08-09 10:36:01)。
- 特征 57 个(删除 22 个常量列,清单与删除重叠行之前相同)。
- 窗口数(W=60):train 70,256(stride 10)、val_es 87,781、val_thr 87,781(stride 1)。
- 本地读取 + 预处理全量 train 约 6 秒;不做磁盘缓存,每次运行重新读取,scaler 随模型保存到 artifacts/{model}/。
- 观察(不据此改动):有几个特征在训练部分的标准差很小(P4_ST_FD 0.0016、P4_HT_FD 0.0020、P2_24Vdc 0.0033),
  标准化后正常数据里就有 |z| > 10 的点(train 最大 31.9,val_thr 最大 26.8,涉及 P1_FT01Z、P1_PIT01、P4_HT_FD、P4_ST_FD、P4_ST_PT01)。
  这些重尾特征可能主导 MSE 分数和归因,M2/M5 汇报时留意。
- 测试:新增 25 个(共 117 个,全部通过)。覆盖:窗口不跨越三类边界(stride 1/3/10)、常量列删除(只看 train 部分、删除重叠行之后)、
  scaler 只 fit train 部分、每个文件前 W−1 秒为 NaN、标签隔离白名单、真实数据上按内容哈希断言训练窗口不含任何 test 行
  (关掉开关时能查到 43,202 行,证明断言不是空检查)。
- 标签隔离白名单:`src/eval/labels.py`、`src/evaluate.py`、`src/attribution.py`;其余 src/ 文件不得出现 "attack" 或引用标签模块。

- 已确认(2026-10-07):train1 第 1 行保持删除(head_rows = 1),原则是训练数据中不出现任何 test 行。
- 重尾特征先不改动;M2a 增加只用 val_thr 的诊断(PCA 各传感器对 MSE 的贡献占比,单个超过 30% 时停下汇报),
  任何缩放调整必须在第一次评估 test 之前定下并写进 CLAUDE.md。

### C0 集群运行包(2026-10-07)

集群:Minerva(Chalmers CSE,SLURM,Ubuntu 24.04,Python 3.12,L4 / L40s)。尚未获得访问权限,以下全部只在本地验证。

- `config/cluster.yaml`(未知项为 TODO)、`docs/cluster_notes.md`(规则与待确认清单)、`docs/cluster_runbook.md`(按顺序的全部命令,
  严格遵守"登录节点不跑计算":pytest、冒烟、核对都走 srun / sbatch)。
- 运行入口 `python -m src.run_model --model M [--resume] [--smoke] [--set k=v] [--ablation-index N]`:按阶段执行,
  目前只有 `prepare`(读取、切分、fit 预处理,保存 scaler.joblib / preprocess.json);训练、打分、阈值、评估从 M2a 起加入阶段表。
- `src/utils/run.py`:配置覆盖与哈希、随机种子与 cuDNN 确定性、设备检测、线程数与 num_workers 按 SLURM_CPUS_PER_TASK 截断、
  `run_manifest.json`(git 提交、配置哈希、超参数、版本、设备、主机、起止时间、final / early-stop epoch、SLURM_JOB_ID、
  SLURM_ARRAY_TASK_ID、节点、nvidia-smi 结果;多段续训逐段记录在 segments)。日志写到运行目录的 run.log。
- 作业脚本由 `python -m scripts.make_cluster_jobs` 按 cluster.yaml 生成到 `cluster/`:train_one.sbatch(L4 × 1、4 CPU、32G)、
  baselines.sbatch(16 CPU、64G、无 GPU)、ablation_array.sbatch(`--array=0-8%3`)、pytest.sbatch(GPU 环境检查)。
  都带 `--resume`;不带 `--smoke` 且不在 SLURM 作业里时拒绝运行;集群路径还是 TODO 时拒绝运行;作业时长为 TODO 时不写 `--time` 行,提交时必须显式给出。
- 消融任务号:0~8 = W {30, 60, 120} × hidden {32, 64, 128} 行优先(任务 4 = 主模型配置);运行目录 `ablation/lstm_ae_w<W>_h<H>/`。
  阈值分位 {99, 99.5, 99.9} 只需用保存的 val_thr 分数重新取分位,不占 array 任务。
- 同步:`scripts/push_to_cluster.sh [--dry-run|--data]`(只推 git 跟踪的文件,写 .git_commit)、
  `scripts/pull_artifacts.sh [--with-checkpoints] [运行名...]`;都先测 ssh 别名 minerva,连不上提示检查 Chalmers VPN。
- 数据:`scripts.download_hai` 新增 `--offline`(按随数据同步的 download_manifest.json 核对,不联网)。
- 依赖:`env/requirements-cluster.txt` 由本地 pip freeze 生成(80 个包,版本与本地一致,torch 为同版本的 CUDA 构建),说明在 `env/README.md`。
- **Python 3.12 验证(本地)**:用 3.12.14 新建环境按锁定版本安装,全部有 wheel;在只含 git 跟踪文件的干净副本上跑完整 pytest:
  135 通过、3 跳过(跳过的是需要代码目录下 .venv 的本地冒烟,集群上会运行)。PyPI 上 torch 2.14.1 的 CUDA 构建在 3.12 上可解析
  (依赖 CUDA 13.0 系列运行库,需要驱动足够新;驱动版本待查,不匹配时的装法见 env/README.md)。
- 本地冒烟:`bash cluster/{train_one,baselines,ablation_array}.sbatch --smoke ...` 直接执行通过(真实数据 1%,只到 prepare 阶段)。
- 测试:新增 21 个(共 138 个,全部通过,约 70 秒)。
- 未验证(需要集群):sbatch 指令在 Minerva 上是否被接受、L4 上的 CUDA torch、srun 下的 pytest、rsync 同步。

### M2a 评估框架与基线(2026-10-07,本地部分)

**还没有任何可写进报告的数字**:主表和 val_thr 诊断都要等集群上的完整运行。下面只有代码和流程。

- 评估:`src/eval/segments.py`(`to_segments(binary, merge_gap_s=0)`,报警段与事件共用)、`src/eval/metrics.py`
  (逐点 P / R / F1 / AUROC,事件检出率、延迟、误报率;报警规则为分数 > 阈值;不使用 point-adjust)。
  约定:没有报警时 precision 为 NaN;没有正样本时 recall、F1、AUROC 为 NaN;正常时长只计已打分且标签为 0 的秒;
  检测延迟按事件内第一秒报警计算(报警在事件开始前就已持续时延迟为 0)。
- 模型接口 `fit(train_ds, val_es_ds)` / `score(ds) -> (scores, per_sensor_err | None)`,注册表在 `src/models/__init__.py`。
  - **PCA 输入选逐秒向量**(57 维),不用展平窗口(3,420 维):展平窗口的主成分大部分在描述时间平移;逐秒 PCA 是不含时间信息的基线,
    与 LSTM 的差别正好反映时间建模的作用;逐传感器误差可直接得到。窗口分数 = 窗口内 60 秒逐秒重构 MSE 的均值(与 LSTM-AE 口径相同),
    per_sensor_err = 窗口最后一秒各传感器的平方误差。
  - IsolationForest:每个窗口 57 × 5 = 285 个统计特征,100 棵树、max_samples 256,用全部训练窗口(约 7 万,不下采样)。
- 阶段(`run_model`,每个阶段也有单独的入口 `python -m src.{train,diagnose,score,threshold,evaluate}`,可单独重跑):
  prepare → train → **diagnose** → score → threshold → evaluate。
  - diagnose 只用 val_thr:各传感器对重构 MSE 的平均占比 → `val_sensor_contrib.csv`。单个传感器超过 30%(`ml.diagnosis.max_sensor_share`)时
    运行**在读取 test 之前停下**(退出码 3,status = stopped_for_review;基线作业里 IsolationForest 也不会开始)。
    决定之后用 `--set ml.diagnosis.allow_over_limit=true` 放行。冒烟运行只警告不停下。
  - score 产物:`scores_val_thr.npy`、`scores_test.npy`(5 个 test 文件拼接,T = 402,005)、`per_sensor_err_test.npy`、`test_index.json`(各文件偏移)。
  - threshold:`threshold.json` 含阈值、val_thr 分数的均值 / 方差(M2c 检查坍塌用)和 99 / 99.5 / 99.9 分位(M4 可选的分位敏感性直接用)。
  - evaluate:`eval_metrics.json`(含主表的一行 main_row、分文件逐点指标)、`eval_events.csv`(每个事件的子系统集合、是否检出、延迟)、
    `eval_alarms.csv`。
- 报告脚本 `python -m scripts.make_main_report` → `reports/main/main_results.csv/.md`、`val_sensor_contrib.csv/.png`;
  只接受非冒烟、git_dirty=false、已完成的运行,其他一律拒绝并说明原因。
- config 新增 `ml.train`(Adam、lr 1e-3、batch 256、50 epoch、patience 5)、`ml.threshold`、`ml.eval.merge_gap_s`、`ml.diagnosis`、
  `ml.models.pca` / `iforest` 的参数。
- 标签隔离:只有 `src/evaluate.py` 导入标签模块,白名单未改动。
- 本地冒烟(`bash cluster/baselines.sbatch --smoke`,1% 数据):两个模型六个阶段全部跑通,共约 23 秒。
  1% 的 test 数据里没有攻击,所以冒烟只检查了流程;带攻击事件的端到端路径由合成数据的测试覆盖。
- 测试:新增 35 个(共 173 个,全部通过)。覆盖手工样例上的指标(无报警、全报警、无攻击事件、事件不跨文件、未打分的秒)、
  PCA 窗口分数的定义、IsolationForest 的确定性、诊断超限时不产生任何 test 产物、单阶段重跑、报告脚本拒绝冒烟 / dirty 运行。
- 顺带修复:`test_sync_scripts_report_vpn_when_unreachable` 在 push 脚本加了工作区检查之后会失败(临时目录不是 git 仓库),
  改为只测 pull 脚本,push 的连接提示由 dirty 检查的测试覆盖。
- 未验证(需要集群):全量数据上的耗时和内存。估计 PCA 几秒;IsolationForest 给 402,005 个窗口算特征和打分是最慢的一步。

**本地诊断运行(2026-10-07,经同意;全量 train,只跑 prepare → train → diagnose,不读 test;只用于决策,不进报告)**

- 命令 `python -m src.run_model --model pca --local-diagnostic`,产物在 `artifacts/local_diagnostic/pca/`,manifest 带 local_diagnostic=true,
  `make_main_report` 拒绝这类运行。耗时 17 秒(PCA 拟合 8.8 秒);PCA 保留 21 个主成分(方差 95.5%)。
- **结论:没有传感器超过 30%,缩放方式保持 StandardScaler 不变(已写进 CLAUDE.md 决策 6b)。**
  val_thr 上占比前 5:P3_FIT01 9.0%、P1_FCV02D 7.9%、P3_PIT01 7.5%、P2_CO_rpm 5.7%、P4_ST_TT01 5.2%;前 10 合计 50.4%。
  按子系统:P1 50.8%(38 个传感器中保留的部分)、P3 20.8%、P2 16.1%、P4 12.3%。
- M1 记下的重尾 / 小标准差特征没有主导 PCA 误差:P4_HT_FD 0.12%(val_thr 最大 |z| 23.3)、P4_ST_FD 0.05%(26.8)、
  P1_PIT01 1.3%(15.1)、P4_ST_PT01 1.5%(14.6)、P1_FT01Z 2.8%(11.0)、P2_24Vdc 0.01%。
  注意这只说明 PCA 的情况;LSTM 类模型的误差分布可能不同,M2b / M5 汇报时仍要看逐传感器误差。
- 集群完整运行会重新跑诊断;`make_main_report` 会自动与本地结果比较(占比相差超过 0.5 个百分点报"不一致")。

**指标约定(2026-10-07 定)**:没有报警时 precision 记 NaN,有攻击则 recall = 0、F1 = 0;AUROC 只依赖分数,只有标签全为同一类时才是 NaN;
正常时长只计已打分的秒。现有实现已符合,补了两个专门的测试。

**冒烟数据**:test 部分除每个文件前 1%,另取 test1 第一个事件(第 2111~2302 行)前后各 600 秒(`ml.smoke.extra_test_clips`),
事件级指标在冒烟中真正跑一遍。

### C0 集群准备与 M2a 集群运行(2026-10-08)

**集群环境**(详见 `config/cluster.yaml`):
- home 为 `/data/users/<CID>`,无用户级配额;`/data` 全体共享 59T,剩 2.8T。代码 `openvas/`、数据 `hai-21.03/`、产物 `openvas_artifacts/` 都在其下。
- 默认分区 `short`(30 分钟),`long` 1 天;登录节点和计算节点都能上网;没有单独的 scratch。
- 显卡驱动 595.71.05,装 PyPI 默认的 torch 2.14.1(CUDA 13.0 构建),L4 上验证可用。venv 占 3.8 GB。
- HAI 数据在计算节点上下载,8 个文件的大小和 sha 与官方仓库一致。
- 集群上完整 pytest:CPU 版与 GPU 版(L4)都是 206 个全部通过。

**在集群上发现并修复的问题**:
- 没申请 GPU 的作业也能看到节点上的显卡。作业脚本现在对没申请 GPU 的作业设置 `CUDA_VISIBLE_DEVICES=""`。
- GPU 上断点续训失败:checkpoint 被读到 GPU,恢复 CPU 随机数状态时报错。已改为读到 CPU(5ec3feb)。这是提交训练作业之前发现的。
- 带 `--gres=gpu` 的 `srun` 不写任务数时启动了两个任务,同时写同一个运行目录。所有作业和 srun 命令现在显式 `--ntasks=1`(6227268)。
  `sbatch` 提交的正式作业不受影响。
- 一个本地测试在路径补全后会真的连接集群,已改掉。
- 小瑕疵(未改):CPU 作业的 manifest 里 `gpu_nvidia_smi` 仍列出节点上的全部显卡;判断设备以 `device` 字段为准。

**M2a 主表前两行**(集群作业 274399,提交 6227268,git_dirty=false,设备 cpu,总耗时 58 秒;不使用 point-adjust):

| model | precision | recall | f1 | auroc | event_recall | delay_median_s | false_alarms_per_hour | n_unscored |
|---|---|---|---|---|---|---|---|---|
| pca | 0.0946 | 0.8405 | 0.1700 | 0.9014 | 0.96(48/50) | 6.0 | 7.16 | 295 |
| iforest | 0.4144 | 0.4828 | 0.4460 | 0.8347 | 0.70(35/50) | 28.0 | 6.84 | 295 |

- val_thr 诊断(PCA):占比最大的传感器 P3_FIT01 9.0%,与本地诊断相差 0.000 个百分点,未超过 30%,缩放方式不变。
- **与预期不符之处**:PCA 的 precision 只有 0.095。按阈值的定义 val_thr 上有 1% 的秒报警,而 test 上有 19.8% 的秒报警
  (IsolationForest 为 2.6%),攻击秒只占约 2%。可能的原因:test 采集于训练数据之后,正常工况的分布有漂移,
  PCA 的逐秒重构误差对此敏感;AUROC 0.90 说明分数的排序仍有区分度,问题在阈值的可迁移性上。
  不据此调整阈值或缩放方式(约束 2、3,决策 6b)。写进报告的讨论部分。

### M2b LSTM-AE 集群运行(2026-10-08)

- 计时作业 274400:全量数据 1 个 epoch,训练 1.1 秒 + 验证 0.4 秒(L4)。据此估算完整训练约 2 分钟、消融 9 个组合远低于 6 小时,不需要串接和缩减。
- 完整训练 274401:提交 64ede38,git_dirty=false,设备 cuda:0(NVIDIA L4,CUDA 13.0),无配置覆盖,一段跑完,作业总耗时 80 秒。
  配置 W=60、hidden=64、1 层,参数 68,473 个。
- 训练曲线:跑满 50 个 epoch,**没有触发早停**;最好的是第 48 个 epoch(val_es 0.1157),最终权重取自它。
  train 损失 0.437 → 0.118,val_es 0.274 → 0.116,两条曲线同步下降,没有过拟合迹象。
  第 28~37 个 epoch 损失有一次明显的再下降(val_es 0.148 → 0.123)。到第 50 个 epoch 损失仍在缓慢下降,
  说明 max_epochs=50 时训练还没有完全收敛;max_epochs 是事先固定的配置,不因此修改。
- 每个 epoch 耗时:训练 0.68~1.10 秒,验证 0.37~0.42 秒,平均合计 1.18 秒。
- 结果(不使用 point-adjust,排除未打分 295 秒):P 0.2765,R 0.8040,F1 0.4115,AUROC 0.9339;
  事件 50/50 全部检出,延迟中位数 6 秒;误报 14.95 次/小时(1,631 段)。阈值 0.2369。
- test 上 6.5% 的秒报警(val_thr 上按定义为 1%),与 PCA 相同方向的现象但程度轻得多(PCA 为 19.8%)。
- 误报段数是 PCA 的两倍左右(14.95 对 7.16 次/小时),但误报的总秒数少于 PCA(precision 0.28 对 0.09):LSTM-AE 的误报更碎。
  merge_gap_s 固定为 0(决策 6),不因此修改。
- val_thr 上占比最大的传感器:P2_24Vdc 11.6%(只用于报告,决策 6b)。
- 分数未坍塌(score_cv 0.35)。

### M2c LSTM-VAE 集群运行(2026-10-09)

- 作业 274566:提交 64ede38,git_dirty=false,status=completed,设备 cuda:0(NVIDIA L4,CUDA 13.0),无配置覆盖,一段跑完,作业总耗时 111 秒。
  配置 W=60、hidden=64、latent=32、β=1.0,参数 64,441 个。
- 训练曲线:跑满 50 个 epoch,**没有触发早停**;最好的是第 50 个 epoch(val_es 损失 305.84),到最后损失仍在下降,
  和 LSTM-AE 一样没有完全收敛;max_epochs 不改。train 损失 996.5 → 315.0,val_es 671.0 → 305.8,两条曲线同步下降,没有过拟合迹象。
  重构 MSE:train 0.567 → 0.162,val_es 0.368 → 0.157(LSTM-AE 的 val_es MSE 为 0.116)。
  KL(每个窗口):第 6 个 epoch 升到约 48,之后降到 37~38 并保持,没有后验坍塌(KL 没有趋于 0)。
- 每个 epoch 耗时:训练 0.84~1.35 秒,验证 0.37~0.43 秒,平均合计 1.36 秒。
- 结果(不使用 point-adjust,排除未打分 295 秒):P 0.0606,R 0.8111,F1 0.1128,AUROC 0.8800;
  事件 48/50 检出(漏检 2 个),延迟中位数 4 秒、均值 16.8 秒;误报 21.22 次/小时(2,315 段)。阈值 0.2790。
- **与预期不符:LSTM-VAE 明显不如 LSTM-AE**(F1 0.113 对 0.412,AUROC 0.880 对 0.934),precision 低于 PCA(0.095),
  误报段数是四个模型里最多的。test 上 29.8% 的秒报警(val_thr 上按定义为 1%;LSTM-AE 6.5%,PCA 19.8%)。
  - 分文件看,报警比例:test1 69.9%、test2 3.4%、test3 31.5%、test4 80.7%、test5 21.2%
    (LSTM-AE 为 4.9 / 3.6 / 2.9 / 20.8 / 9.0%,PCA 为 17.2 / 5.8 / 23.1 / 67.6 / 14.8%)。
    三个重构类模型都在 test4 上报警最多、test2 上最少,方向一致,VAE 最严重。
  - 分文件的 AUROC 并不低(0.925 / 0.894 / 0.947 / 0.948 / 0.946),合并后只有 0.880:
    各文件正常分数的整体水平不同,文件内排序尚可,跨文件用同一个阈值就不行了。
  - 可能的原因:① test 的正常工况相对训练数据有漂移(与 PCA 同一现象),VAE 的重构更粗(val_es MSE 0.157 对 0.116),
    正常分数分布更集中(val_thr 上 score_cv 0.32),漂移后整段分数越过阈值;② β=1 且 KL 按窗口求和,正则偏强,
    限制了重构能力;③ 训练未收敛。这些是推测,没有做实验验证。
  - β、max_epochs、阈值分位都是事先固定的,不据此修改,不在 test 上调参(约束 2、3)。
- val_thr 上占比最大的传感器:P2_24Vdc 9.8%(只用于报告,决策 6b);前 5 个传感器与 LSTM-AE 的前 5 相同(P2_24Vdc、P2_VT01、P4_ST_FD、P2_HILout、P2_VYT02),顺序略有不同。
- 分数未坍塌(score_cv 0.32)。

### M2c Deep SVDD 集群运行(2026-10-09)

- 作业 274567:提交 64ede38,git_dirty=false,status=completed,设备 cuda:0(NVIDIA L4,CUDA 13.0),无配置覆盖,一段跑完,运行耗时 35 秒。
  配置 W=60、hidden=64、rep_dim=32、eps=0.1,参数 33,024 个。
- **中心 c 的统计量**(`deep_svdd_center`,用全部 70,256 个训练窗口的初始输出求均值):替换前 |c| 最小 0.00025、中位数 0.0102、最大 0.0366;
  **32 维全部小于 eps,全部被替换为 ±0.1**。与冒烟时的观察一致:c 只保留了初始输出的符号,几乎不携带数据信息(‖c‖² = 0.32)。规则未改(决策 6b)。
- 训练曲线:第 18 个 epoch 早停,最好的是第 13 个(val_es 2.17e-5)。train 损失 0.0369 → 9.4e-6,一路下降;
  val_es 损失前 13 个 epoch 与 train 同步下降,**第 14 个 epoch 起突然升高一个数量级以上并持续上升**(2.8e-4 → 1.3e-3),train 损失没有同样的变化。
  val_es 损失是距离的均值,对少数距离极大的窗口很敏感(val_thr 上也有 1 个窗口的分数达到 7.55,而中位数是 1.7e-5),
  推测是这类少数窗口在后续 epoch 被映射得更远造成的;没有进一步核实。最终权重取自第 13 个 epoch。
- 每个 epoch 耗时:训练 0.67~0.81 秒,验证 0.32 秒,平均合计 1.01 秒。
- **坍塌检查**:按事先定的判据(val_thr 分数的变异系数低于 0.001)**没有坍塌**,`collapsed=false`,score_cv = 236。需要如实补充两点:
  - score_cv 这么大是被那 1 个极端窗口(7.55)拉高的,去掉最高的 0.1% 之后变异系数为 0.81,四分位距 ÷ 中位数为 0.69;分数确实有离散度,不是常数。
  - 但分数的绝对量级非常小:val_thr 中位数 1.7e-5,阈值 7.8e-5,而 ‖c‖² = 0.32。即几乎所有正常窗口都被映射到离 c 极近的位置,
    模型在向"所有输入映射到 c"的方向收缩(Deep SVDD 的典型坍塌方向),只是没有收缩到分数失去排序能力:AUROC 0.906。
- 结果(不使用 point-adjust,排除未打分 295 秒):P 0.1683,R 0.8287,F1 0.2797,AUROC 0.9055;
  事件 49/50 检出(漏检 1 个),延迟中位数 3 秒、均值 13.9 秒;**误报 69.33 次/小时(7,564 段)**,是五个模型里最多的,约为 LSTM-AE 的 4.6 倍。阈值 7.84e-5。
- test 上 11.0% 的秒报警(val_thr 上按定义为 1%)。分文件:test1 17.7%、test2 3.4%、test3 7.2%、test4 41.3%、test5 9.0%,
  仍是 test4 最多、test2 最少,与其他模型方向一致。
- 可能的原因(推测):分数的量级极小、正常分数挤在很窄的绝对范围内,阈值附近的分数来回越过阈值,报警段非常碎(7,673 段,merge_gap_s 固定为 0);
  加上与其他模型相同的 test 正常工况漂移。c 不携带数据信息可能也有影响。中心规则、eps、阈值分位都不改,不在 test 上调参。
- 不输出逐传感器误差,不参与归因(决策 5)。

**五个模型的主表**(集群完整运行,不使用 point-adjust;完整版见 `reports/main/main_results.md`):

| model | precision | recall | f1 | auroc | event_recall | delay_median_s | false_alarms_per_hour |
|---|---|---|---|---|---|---|---|
| pca | 0.0946 | 0.8405 | 0.1700 | 0.9014 | 0.96 | 6.0 | 7.16 |
| iforest | 0.4144 | 0.4828 | 0.4460 | 0.8347 | 0.70 | 28.0 | 6.84 |
| lstm_ae | 0.2765 | 0.8040 | 0.4115 | 0.9339 | 1.00 | 6.0 | 14.95 |
| lstm_vae | 0.0606 | 0.8111 | 0.1128 | 0.8800 | 0.96 | 4.0 | 21.22 |
| deep_svdd | 0.1683 | 0.8287 | 0.2797 | 0.9055 | 0.98 | 3.0 | 69.33 |

- LSTM-AE 的 AUROC 和事件检出率最高;逐点 F1 最高的是 IsolationForest(0.446),LSTM-AE 第二(0.412)。
- 所有模型在 test 上的报警比例都高于 val_thr 上的 1%,阈值从 val_thr 迁移到 test 的问题是共性的,写进报告的讨论部分。

### M4 LSTM-AE 消融集群运行(2026-10-09)

- 作业 274568(`--array=0-8%3`):9 个任务全部 COMPLETED,提交 64ede38,git_dirty=false,设备 NVIDIA L4,各一段跑完;
  覆盖项只有 `ml.window.length` 和 `ml.models.lstm_ae.hidden`。单个任务 1 分 09 秒 ~ 4 分 45 秒(最长是 W=120、hidden=128),
  9 个任务的运行时间合计约 18 分钟,远低于 6 小时,不需要缩减。`jobs.ablation.time` 填为 10 分钟。
- **可复现性**:任务 4(W=60、hidden=64)与主模型运行 274401 的全部指标和阈值逐位相同(F1 0.4115、阈值 0.236868)。
- 训练:8 个组合跑满 50 个 epoch 未早停,最好的 epoch 在 45~50;只有 W=120、hidden=128 在第 50 个 epoch 恰好满足早停条件(最好第 45 个)。
  和主模型一样,多数组合到第 50 个 epoch 仍未完全收敛。没有组合坍塌(score_cv 0.21~0.43)。
- 结果(不使用 point-adjust;完整表见 `reports/main/ablation.md`,图 `ablation_heatmap.png`):

| W | hidden | precision | recall | f1 | auroc | event_recall | delay_median_s | false_alarms_per_hour | val_es 最好损失 |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 32 | 0.4547 | 0.7366 | 0.5623 | 0.8994 | 0.96 | 5.0 | 7.47 | 0.148 |
| 30 | 64 | 0.1157 | 0.8045 | 0.2023 | 0.9089 | 1.00 | 3.0 | 58.72 | 0.086 |
| 30 | 128 | 0.2109 | 0.7820 | 0.3322 | 0.9242 | 1.00 | 4.5 | 32.50 | 0.061 |
| 60 | 32 | 0.5084 | 0.7859 | 0.6174 | 0.9251 | 0.96 | 5.5 | 1.53 | 0.160 |
| **60** | **64**(主模型) | 0.2765 | 0.8040 | 0.4115 | 0.9339 | 1.00 | 6.0 | 14.95 | 0.116 |
| 60 | 128 | 0.0441 | 0.9395 | 0.0843 | 0.9166 | 1.00 | 0.0 | 74.66 | 0.085 |
| 120 | 32 | 0.4882 | 0.7705 | 0.5977 | 0.9430 | 0.96 | 17.0 | 0.33 | 0.176 |
| 120 | 64 | 0.5022 | 0.7913 | 0.6145 | 0.9462 | 0.96 | 14.0 | 0.51 | 0.155 |
| 120 | 128 | 0.1494 | 0.8600 | 0.2546 | 0.9387 | 1.00 | 6.5 | 20.10 | 0.142 |

- **结论:结果对配置非常敏感,而且不单调。** 逐点 F1 在 0.084 ~ 0.617 之间,误报率在 0.33 ~ 74.7 次/小时之间(相差两个数量级);
  AUROC 相对稳定(0.899 ~ 0.946)。差别主要来自阈值处的 precision,而不是分数的排序能力。
- **与预期不符:主模型配置(W=60、hidden=64)不是其中表现好的**,逐点 F1 在 9 个组合中排第 5,有 4 个组合明显更好
  (60/32、120/64、120/32、30/32,F1 0.56~0.62,误报 0.3~7.5 次/小时)。按约束 9,主模型配置事先固定,**不据此修改**;如实写进报告。
- **val_es 重构损失越低,test 上的 precision 往往越差**:同一个 W 下 hidden=128 的 val_es 损失最低,但误报最多(W=60、hidden=128 的 precision 只有 0.044)。
  可能的原因(推测,未验证):容量大的模型把正常数据重构得更精细,val_thr 上的分数分布更窄、阈值更低(0.17 对 0.29),
  test 上正常工况稍有漂移,整段分数就越过阈值;容量小的模型对漂移不敏感。这与 PCA、LSTM-VAE、Deep SVDD 上看到的阈值迁移问题是同一类现象。
  这也意味着 val_es 损失不能用来预测检测效果。
- W 的影响:W=120 的 AUROC 最高、误报最少,但检测延迟更长(中位数 14~17 秒),未打分的秒数也更多(595 对 295)。
- 阈值分位的敏感性(`ablation_quantiles.csv`,只作敏感性分析,不据此选阈值):分位数提高到 99.9% 时 9 个组合的 F1 收敛到 0.63~0.68,
  事件检出率降到 0.82~0.92。说明各组合之间的差别大部分来自 99% 分位阈值的迁移,而不是模型本身的区分能力。
  主模型在 99.5% / 99.9% 分位下 F1 为 0.629 / 0.682;**主表仍按约束 3 用 99% 分位**。
- 这些数字来自单个随机种子(42),组合之间的差别有多少来自种子没有评估,写进报告的局限性。

### M3 主结果表与图(2026-10-09)

全部由集群完整运行的产物在本地生成(`scripts.make_main_report`、`scripts.make_figures`);五个运行的 manifest 都是
git_dirty=false、status=completed、非冒烟、提交 6227268(基线)或 64ede38(LSTM 类,设备 NVIDIA L4)。画图脚本不读标签,攻击区间取自 `eval_events.csv`。

- 产物(`reports/main/`):`main_results.csv/.md`(5 行,数字见上面"五个模型的主表")、`scores_test1~5.png`(每个测试文件一张,
  五个模型各一个子图,对数纵轴、阈值线、攻击区间阴影)、`score_distribution.png`、`training_curves.png`、`per_subsystem_detection.csv`。
- 未打分:每个模型 295 秒(5 个文件 × 59 秒),不参与任何指标,其中没有攻击秒。
- 分子系统的事件检出(一个事件涉及多个子系统时在每个子系统下各计一次:P1 34、P2 22、P3 8 个):

| model | P1 | P2 | P3 | 全部 |
|---|---|---|---|---|
| pca | 33/34 | 22/22 | 7/8 | 48/50 |
| iforest | 31/34 | 12/22 | 4/8 | 35/50 |
| lstm_ae | 34/34 | 22/22 | 8/8 | 50/50 |
| lstm_vae | 32/34 | 22/22 | 8/8 | 48/50 |
| deep_svdd | 33/34 | 22/22 | 8/8 | 49/50 |

  - IsolationForest 漏掉的主要是 P2(10 个)和 P3(4 个)的事件;四个按秒重构或按距离打分的模型在 P2 上全部检出。
  - PCA、LSTM-VAE、Deep SVDD 漏检的事件都在 test2(事件 10、19、24),其中事件 19(P1,152 秒)被 PCA 和 LSTM-VAE 同时漏掉,
    事件 10(P1,123 秒)被 LSTM-VAE 和 Deep SVDD 同时漏掉。
- 图中能直接看到的现象(与表中数字一致,不是新的结论):
  - **分数曲线**:以 test4 为例,攻击区间内五个模型都有明显的尖峰;但 PCA、LSTM-VAE、Deep SVDD 的正常段基线从文件开始约 1.5 小时后
    整体抬高到阈值之上并一直保持,LSTM-AE 的基线贴着阈值上下波动。也就是说误报主要不是零星尖峰,而是正常分数的整体水平高于 val_thr。
    这支持"test 的正常工况相对训练数据有漂移、阈值迁移不好"的解释,但没有做进一步实验(例如定位是哪些传感器漂移),仍属推测。
  - **分数分布**:test 正常分数的分布相对 val_thr 整体右移(PCA、LSTM-AE、LSTM-VAE 最明显),阈值落在 test 正常分布的主体之内;
    攻击分数有很长的右尾,但主体与 test 正常分数重叠。IsolationForest 的 test 正常分布只是变宽,攻击分布的峰在阈值之上,所以它的 precision 最高。
  - **训练曲线**:LSTM-AE、LSTM-VAE 的 train 与 val_es 同步下降到第 50 个 epoch;Deep SVDD 的 val_es 从第 14 个 epoch 起急升(见上)。
- 不使用 point-adjust,逐点 F1 偏低是预期结果;评估方法、阈值分位、主模型配置都没有改动。

### LSTM-AE 多随机种子(2026-10-09)

- 决定(CLAUDE.md 决策 6c):只对 LSTM-AE 主配置(W=60、hidden=64)做 5 个种子,主表该行给出 F1、AUROC 的均值 ± 标准差,其余模型维持单种子;
  目的是判断 LSTM-AE 与 IsolationForest 的 F1 差距是否在随机波动范围内。
- 实现:种子 42 直接用已有的主模型运行(消融任务 4 已证明同一种子逐位可复现),另跑 43~46;标准差为样本标准差(ddof=1)。
  `config.yaml` 的 `ml.seed_study`;`scripts/make_main_report.py` 要求每次运行除 `ml.seed` 外没有别的覆盖项,种子不齐时维持单种子。
  主表 csv 不变(仍是种子 42),逐种子结果在 `main_results_seeds.csv`。新增 1 个测试(共 207 个)。
- 集群作业 274758~274761:4 个都是 COMPLETED,提交 64ede38,git_dirty=false,设备 NVIDIA L4,覆盖项只有 `ml.seed=<种子>`,各一段跑完(约 1.5 分钟)。
  都跑满 50 个 epoch 未早停,最好的 epoch 49 或 50,val_es 最好损失 0.118~0.119(种子 42 为 0.116),训练情况与主模型运行一致;没有坍塌(score_cv 0.35~0.37)。
- 逐种子结果(不使用 point-adjust,各排除未打分 295 秒):

| seed | precision | recall | f1 | auroc | 事件检出 | false_alarms_per_hour | threshold |
|---|---|---|---|---|---|---|---|
| 42(主模型运行) | 0.2765 | 0.8040 | 0.4115 | 0.9339 | 50/50 | 14.95 | 0.2369 |
| 43 | 0.3047 | 0.7997 | 0.4413 | 0.9331 | 50/50 | 10.63 | 0.2485 |
| 44 | 0.2847 | 0.7964 | 0.4194 | 0.9311 | 50/50 | 14.18 | 0.2519 |
| 45 | 0.3162 | 0.8111 | 0.4550 | 0.9354 | 49/50 | 13.64 | 0.2421 |
| 46 | 0.2824 | 0.8018 | 0.4177 | 0.9341 | 49/50 | 12.54 | 0.2475 |
| 均值 ± 标准差 | 0.293 ± 0.017 | 0.803 ± 0.006 | **0.4290 ± 0.0184** | **0.9335 ± 0.0016** | 0.992 ± 0.011 | 13.19 ± 1.68 | |

- **结论 1:LSTM-AE 与 IsolationForest 的逐点 F1 差距在随机波动范围内。** IsolationForest(单种子)0.4460,比 LSTM-AE 的均值高 0.017,
  不到 1 个标准差(0.93);5 个种子里有 1 个(种子 45,0.4550)高于 IsolationForest。按逐点 F1 不能说哪个更好。
  保留意见:只有 5 个种子;IsolationForest 自己也依赖随机种子,它的波动没有评估。
- **结论 2:AUROC 的差距不是随机波动。** LSTM-AE 0.9335 ± 0.0016,IsolationForest 0.8347、PCA 0.9014、Deep SVDD 0.9055,差距是标准差的几十倍
  (其余模型为单种子)。事件检出同理:LSTM-AE 各种子 49~50/50,IsolationForest 35/50。
- **结论 3:消融里的大差别不能用种子噪声解释。** 主配置下 F1 的种子标准差是 0.018,而消融组合之间相差 0.1~0.5
  (例如 60/32 的 0.617、60/128 的 0.084)。保留意见:其他组合的种子波动没有测,可能与主配置不同。
- 需要如实说明的两点:
  - 用于报警导出、归因和图的主模型运行(种子 42)恰好是 5 个种子里 F1 最低的一个;按决策 6c 仍只用它,不换种子。
  - "50 个事件全部检出"只在 3 / 5 个种子上成立,种子 45、46 各漏检 1 个,漏掉的都是 test2 的事件 10(P1,123 秒;LSTM-VAE、Deep SVDD、IsolationForest 也漏掉了它);报告里写 49~50 / 50。
- 误报率在各种子上都高于 PCA 和 IsolationForest(10.6~15.0 对 7.2、6.8 次/小时),这一点不随种子改变。
- 2026-10-09 补充:主表 `.md` 中 LSTM-AE 一行写成均值 ± 标准差的列扩大为四个:F1 0.4290 ± 0.0184、AUROC 0.9335 ± 0.0016、
  事件检出率 0.9920 ± 0.0110(49.6 / 50)、误报率 13.19 ± 1.68 次/小时。该行其余各列和主表 csv 仍是种子 42 的运行。
- 已定(2026-10-09):S 封顶问题只写成局限和未来工作,不改公式、不做补充排序;I1 中的入库策略和 mock 输出目录照现在的做法。

### M5 归因(2026-10-09)

由集群完整运行的产物在本地生成(`python -m src.attribution`;三个运行 git_dirty=false、status=completed,LSTM-AE 用种子 42 的主模型运行)。
产物(`reports/main/`):`attribution.csv`(每个报警段一行,含 Top-3 / Top-5)、`attribution_summary.csv/.md`、
`attribution_events_<model>.csv`(按事件汇总)、`attribution_heatmap_<model>.png`(事件 × 子系统的误差占比,圆圈为标签中的受攻击子系统)。

| model | 报警段 | 与攻击重叠 | 命中 | 报警段命中率 | 同一批报警段上"总是猜 P1"的命中率 | 按事件命中 | Top-1 在 P4 |
|---|---|---|---|---|---|---|---|
| lstm_ae | 1,703 | 72 | 54 | **0.750** | 0.681 | 45/50(0.900) | 6/72(8.3%) |
| lstm_vae | 2,372 | 57 | 52 | **0.912** | 0.702 | 47/48(0.979) | 4/57(7.0%) |
| pca | 834 | 53 | 37 | **0.698** | 0.679 | 34/48(0.708) | 0/53 |

- 口径:只对与攻击事件重叠的报警段算命中率(Top-1 所属子系统 ∈ 事件的子系统集合)。**没有任何报警段与多个事件重叠**,主口径与并集口径完全相同;
  无子系统标签的事件为 0 个。不与攻击重叠的报警段(LSTM-AE 1,631、LSTM-VAE 2,315、PCA 781)单列,不计入命中率。
  "按事件命中"是把事件内全部报警秒的误差合并后取 Top-1,只含被检出的事件。
- **必须一起读的参照线**:50 个事件里 34 个涉及 P1,所以不看数据、总是回答 P1 就有约 68% 的命中率。这一列是我在同一批报警段上算的参照,不是模型输出。
  - **PCA 的命中率(0.698)与这条参照线基本相同**:它的 Top-1 有 41/53 落在 P1。单独攻击 P2 的 11 个事件只命中 2 个,
    错的都指向 P1_PCV02Z 或 P3_PIT01。也就是说 PCA 的归因在这份数据上几乎没有信息量。
  - **LSTM-AE(0.750)只比参照线高 7 个百分点**;按事件看好一些(0.900 对 0.68)。它的错误方向与 PCA 相反:P2 的事件全部命中(14/14 个报警段、11/11 个事件),
    错的 18 个报警段里 14 个是单独攻击 P1 的事件被指到 P2(9 个)或 P4(5 个),其余 4 个是 P3 或 P1+P3 的事件被指到 P2、P4;误指最多的传感器是 P4_HT_FD、P2_VT01、P2_24Vdc、P2_VXT02。
  - **LSTM-VAE 最高(0.912,按事件 47/48)**,明显高于参照线。
- **与预期不符:主模型 LSTM-AE 的归因不如 LSTM-VAE**(0.750 对 0.912),尽管它的检测指标好得多。可能的原因(推测,未验证):
  LSTM-AE 的报警段更碎(72 段对 57 段,段长中位数 135 对 157 秒),短的报警段落在事件边缘时,误差由平时就偏高的 P2 / P4 传感器主导
  (val_thr 上占比最高的正是 P2_24Vdc、P4_ST_FD、P2_HILout、P2_VT01);VAE 的重构更粗,攻击造成的大偏差更容易压过这些背景误差。
  注意三个模型的命中率各自算在自己的报警段上,分母不同,不是同一批样本上的比较。
- 按决策 9,`alerts.json` 仍只由 LSTM-AE 导出,不因归因结果改用 LSTM-VAE。
- **P4**:Top-1 落在 P4 的比例 LSTM-AE 8.3%、LSTM-VAE 7.0%、PCA 0%。P4 没有子系统标签,落在 P4 必然算不命中;LSTM-AE 的 18 个未命中里有 6 个是这种情况。
  P4 是 HIL 仿真部分,与 P1、P2 物理耦合,指到 P4 不一定是错的,但现有标签无法判断。
- **误报段的 Top-1(不计入命中率,但对 I1 有影响)**:LSTM-AE 的 1,631 个误报段里 Top-1 在 P2 的占 61.7%、P4 28.9%、P1 8.0%、P3 1.3%;
  最常见的传感器是 P4_HT_FD(16.1%)、P2_VT01(12.8%)、P2_CO_rpm(12.6%)、P2_24Vdc(11.2%)、P4_ST_FD(10.5%)。
  这几个传感器大概率就是 test 上发生漂移的那些(与 M3 的推测一致,仍未单独验证)。PCA 的误报集中在 P1_PCV02Z(42.1%)、P1_FCV02D(22.3%)、P3_PIT01(16.9%);
  LSTM-VAE 集中在 P1_PCV02Z(51.0%)。
- **传感器级 Hit@k 未做**:HAI 21.03 的 CSV 没有逐传感器的攻击目标,官方攻击目标表尚未提供(决策 7)。找到后可补做;否则写进报告的局限性。
- 局限:命中率的分母很小(53~72 个报警段、48~50 个事件),单个种子;P3 的事件只有 2 个单独攻击 P3 的,结论对 P3 基本没有支撑。

### M5 补充分析(2026-10-09,全部在本地;决定写入 CLAUDE.md 决策 7)

上面"M5 归因"一节中"按事件命中"一列是补充口径(合并);本节给出新的主口径和其余补充。输出见 `reports/main/attribution_summary.md`。

**1. 按事件的命中率:两种口径**(分母都是被检出且有子系统标签的事件)
- 主口径:每个事件的**第一个与之重叠的报警段**(按报警段开始时间),看它的 Top-1 所属子系统是否在事件的子系统集合内;对应运维人员最先看到的那条报警。
- 补充口径(原口径):事件区间内全部报警秒的误差合并后取 Top-1;要等事件里的报警都出现后才算得出。

**3. 全部命中率与 95% Wilson 置信区间**

| model | 报警段 | 事件·第一个报警段(主) | 事件·合并(补充) |
|---|---|---|---|
| lstm_ae | 54/72 = 0.750 [0.639, 0.836] | 43/50 = 0.860 [0.738, 0.930] | 45/50 = 0.900 [0.786, 0.957] |
| lstm_vae | 52/57 = 0.912 [0.811, 0.962] | 47/48 = 0.979 [0.891, 0.996] | 47/48 = 0.979 [0.891, 0.996] |
| pca | 37/53 = 0.698 [0.565, 0.805] | 36/48 = 0.750 [0.612, 0.851] | 34/48 = 0.708 [0.568, 0.818] |

- 区间很宽。LSTM-AE 与 LSTM-VAE 在报警段口径上的区间略有重叠(上限 0.836 对下限 0.811),按事件的两种口径也重叠;LSTM-AE 与 PCA 的区间大部分重叠。
  "LSTM-VAE 归因最好"的方向在三种口径下一致,但单凭这批样本达不到很强的把握。
- "总是猜 P1"的参照线(约 0.68)落在 LSTM-AE 报警段口径的区间之内、LSTM-VAE 的区间之外。
- LSTM-AE 的 5 个种子(`attribution_seeds.csv`,均值 ± 样本标准差):报警段 **0.777 ± 0.033**(0.750~0.817),
  事件·第一个报警段 **0.875 ± 0.026**(0.857~0.918),事件·合并 **0.919 ± 0.014**;Top-1 在 P4 的比例 0.114 ± 0.031。
  种子 42(主模型运行)在报警段口径上是 5 个种子里并列最低的。LSTM-VAE 和 PCA 只有单种子。

**2. 按报警段长度分组(验证"短报警段归因更差")**:只含与攻击重叠的报警段。

| model | <10 秒 | 10–60 秒 | >60 秒 |
|---|---|---|---|
| lstm_ae | 7/20 = 0.350 [0.181, 0.567] | 1/4 = 0.250 [0.046, 0.699] | 46/48 = 0.958 [0.860, 0.988] |
| lstm_vae | 2/4 = 0.500 [0.150, 0.850] | 0/3 = 0.000 [0.000, 0.561] | 50/50 = 1.000 [0.929, 1.000] |
| pca | 2/5 = 0.400 [0.118, 0.769] | 1/2 = 0.500 [0.095, 0.905] | 34/46 = 0.739 [0.597, 0.844] |

- **推测成立(对 LSTM-AE)**:超过 60 秒的报警段命中 46/48,不到 10 秒的只有 7/20,两个区间不重叠。
  LSTM-AE 的 18 个未命中里 16 个来自 60 秒以内的报警段。
- LSTM-AE 与 LSTM-VAE 的总体差距主要来自短报警段的数量(20 个对 4 个),而不是长报警段上的归因质量(0.958 对 1.000):
  LSTM-AE 的报警更碎,多出来的短报警段拉低了命中率。
- PCA 不同:长报警段也只有 0.739,它的问题不是报警段短,而是总指向 P1。
- 10–60 秒一组每个模型只有 2~4 个报警段,不能下结论。

**4. 探索性分析:按 val_thr 平均误差归一化(事后分析,不替换主结果)**
- 做法:报警段内各传感器的平均误差先除以它在 val_thr 上的平均误差,再排 Top-k(比较"相对平时的倍数")。基线只来自 val_thr
  (各运行目录的 `val_sensor_contrib.csv`),没有读取测试标签;标签只用于事后评估。
- **这一步是看过测试集上的命中率之后才决定做的,结果偏乐观,不能当作事先确定的方法来引用。** 主结果、alerts.json、卡片都不使用。

| model | 报警段:主结果 → 归一化 | 事件·第一个报警段 | 事件·合并 | Top-1 在 P4 |
|---|---|---|---|---|
| lstm_ae | 54/72 → 61/72(0.750 → 0.847 [0.747, 0.912]) | 43/50 → 43/50 | 45/50 → 46/50 | 6 → 3 |
| lstm_vae | 52/57 → 52/57(0.912 → 0.912) | 47/48 → 44/48 | 47/48 → 44/48 | 4 → 2 |
| pca | 37/53 → 44/53(0.698 → 0.830 [0.708, 0.908]) | 36/48 → 41/48 | 34/48 → 42/48 | 0 → 1 |

  - LSTM-AE:报警段命中率提高(主要在短报警段:<10 秒 7/20 → 13/20,10–60 秒 1/4 → 4/4;>60 秒 46/48 → 44/48),
    但按事件的主口径没有变化(43/50)。
  - PCA 提高最多(三种口径都提高),说明它"总指向 P1"有相当一部分是 P1 的几个传感器平时误差就大造成的。
  - **LSTM-VAE 变差**(按事件 47/48 → 44/48,长报警段 50/50 → 46/50):归一化不是对所有模型都有利。
  - 结论只能写成:"归一化可能减轻背景误差对短报警段归因的影响,但效果不一致,需要在独立数据上事先确定后再验证"。

### I1 的准备:报警导出与融合排序评估(2026-10-09;决定写入 CLAUDE.md 决策 8b)

- `python -m scripts.export_alerts` → `data/processed/alerts.json`:LSTM-AE(种子 42 的主模型运行)的全部 **1,703** 条报警,is_mock=false,
  不按标签筛选;字段契约测试通过。卡片和 Streamlit 只渲染优先级前 50 条(条数做成 config 参数)在 I1 的其余部分实现。
- 融合排序评估 `python -m src.eval.fusion_ranking` → `reports/main/fusion_ranking_eval.csv/.md`。
  **标签隔离**:真值(报警段是否与攻击事件重叠)取自 evaluate 阶段写出的 `eval_alarms.csv`;这个模块不读标签、不在白名单内(白名单未改),
  export_alerts 和 alerts.json 里也没有真值。报警与 eval_alarms.csv 逐条核对起止时间,对不上就报错。
- 结果(1,703 条报警中 72 条与攻击重叠,基准比例 4.2%):

| 排序 | precision@10 | precision@20 | precision@50 | 前 50 条中与原始分数排序相同的 |
|---|---|---|---|---|
| 原始分数 | 10/10 | 20/20 | 45/50 = 0.90 [0.786, 0.957] | — |
| α=0 | 10/10 | 20/20 | 45/50 = 0.90 | 46 |
| α=1 | 10/10 | 20/20 | 45/50 = 0.90 | 44 |

  - **三种排序的 precision@k 完全相同**。它们选出的报警并不完全一样(前 50 条里有 4~6 条不同),但真报警的条数恰好相同。
    也就是说在这份数据上,融合打分(无论是否使用漏洞暴露风险)相对于直接按异常分数排序**没有可测量的差别**,既没有变好也没有变差。
  - 原因很直接:分数最高的报警几乎都是真攻击(攻击处的分数是阈值的几十到几千倍),任何以分数为主的排序前 20 条都是真的;
    而 S 在分数达到阈值 2 倍时封顶,61 条报警的 S = 1(其中 46 条是真的),融合排序在这 61 条之内只按子系统权重、关键度和 E_sub 区分。
    E_sub 只有 0 / 1 两个取值(P1、P2 为 1,P3、P4 为 0,风险饱和,见下文 M9),区分能力有限。
  - 排序本身是有用的(前 50 条 precision 0.90,基准 0.042),但这个作用来自异常分数,不来自漏洞上下文。
  - 按约束 7,这个评估只描述排序公式在这份数据上的表现;资产映射是人为构造的,不能据此说漏洞上下文对检测有帮助或没有帮助。
- 案例(待 I1):选 3 个——归因正确的真实攻击、归因错误的真实攻击、高优先级的误报(前 50 条里有 5 条误报可选)。
- 测试:新增 5 个(共 212 个,全部通过)。

### I1 集成(2026-10-09,本地)

- **报警来源切换**(`config.yaml` 的 `fusion` 段):`alerts_json` 现在指向真实报警 `data/processed/alerts.json`(1,703 条,is_mock=false);
  mock 报警改由 `mock_alerts_json` 指定。mock 的输出单独放(`enriched_alerts_mock.json`、`reports/cards_mock/`),与真实报警的输出互不覆盖。
  `scripts.make_mock_alerts` 的默认输出同步改为 `mock_alerts_json`(否则会覆盖真实报警)。
- **alerts.json 的字段契约没有改动**;扩展部分的打分公式、资产图、漏洞处理也没有改动。改的只是读哪个文件、渲染多少条。
- **只渲染前 N 条**:`fusion.max_rendered_alerts`(默认 50)。融合打分仍对全部 1,703 条进行,`enriched_alerts.json` 含全部报警;
  卡片、索引页、Streamlit 的列表和卡片下拉框只含优先级最高的前 50 条,索引页写明"共 1703 条,只列出前 50 条",卡片上的"第 x / n"按全部报警计。
  渲染前会清掉输出目录里上一次生成的卡片,避免新旧混在一起。真实报警的索引页和 Streamlit 都提示"模型的原始输出,未经人工确认,其中包含误报"。
- **一键流水线**:`python -m src.pipeline --offline` 用真实报警跑通(540 条漏洞、KEV 命中 9 条、1,703 条报警、102 个文件);`--mock` 切回 mock。
  真实数据快照 vulns.json / vulns_enriched.json 内容未变,没有被改写。
- **Streamlit**:默认读真实报警,侧边栏"报警来源"可切回 mock;AppTest 覆盖默认真实、切回 mock、调参三种情况。
  和之前一样,只做了无界面的 AppTest,**没有做目视检查**。
- **入库策略**(可逆,按"真实数据的快照提交、可重新生成的派生输出不提交"处理):`data/processed/alerts.json` 入库;
  `enriched_alerts.json`(23 MB)和 `reports/cards/` 的 100 张卡片不入库,一条命令可以重新生成;三个案例的卡片入库在 `reports/main/cases/`。
- **案例** `python -m scripts.make_cases` → `reports/main/cases.md`。按固定规则挑选,不人工挑:按融合优先级从前往后,各取第一条。

| 案例 | 报警 | 排名 | 分数 ÷ 阈值 | 对照标签 |
|---|---|---|---|---|
| 1 归因正确的真实攻击 | LSTMAE-00112 | 1 | 6,721 倍 | 事件 #6(P2,83 秒),Top-1 P2_VT01,命中;卡片子系统 P2 100% |
| 2 归因错误的真实攻击 | LSTMAE-00472 | 50 | 23.2 倍 | 事件 #34(P1,254 秒),Top-1 P4_ST_PT01(P4),未命中;但卡片按 Top-5 加权后 P1 占 68% |
| 3 高优先级的误报 | LSTMAE-01505 | 28 | 2.2 倍 | 不与任何事件重叠;前一个事件在 396 分钟之前,后一个在 72 分钟之后 |

- 案例暴露的问题(写进报告的讨论部分):
  - **S 的封顶点太低**:S = min(1, 分数 ÷ 阈值 − 1) 在 2 倍阈值处封顶,2.2 倍的误报(案例 3)和 23 倍的真实攻击(案例 2)S 都是 1,
    误报反而排在前面(第 28 对第 50)。公式是扩展部分事先定下的,没有改动;作为局限写明,改进留作未来工作。
  - **漏洞上下文不区分报警**:三张卡片的漏洞都来自 ews_01,排在最前的是同一条;只有一台靶机被评估,涉及 P1 或 P2 的报警得到几乎相同的漏洞上下文。
  - **卡片不能帮助判断误报**:误报的卡片与真实攻击的卡片形式上完全一样。
  - "Top-1 是否命中"比卡片实际展示的信息更严格(案例 2);P4 没有子系统标签,指向 P4 必然算未命中。
- 测试:新增 3 个(共 215 个,全部通过)。

### M2b LSTM-AE 与 M2c LSTM-VAE、Deep SVDD(2026-10-07,本地部分)

经同意不等集群权限,连续完成了两个阶段的本地部分。**没有任何可写进报告的数字**;训练曲线、每个 epoch 的耗时、
Deep SVDD 是否坍塌都要等集群运行。集群命令按依赖顺序写在 `docs/cluster_queue.md`。

- 共用训练流程 `src/models/torch_base.py`:Adam、lr 1e-3、batch 256、最多 50 个 epoch、patience 5(看 val_es 损失),
  训练 stride 10、验证 stride 1;设备自动检测。每个 epoch 写 `checkpoints/last.pt`(模型、优化器、早停状态、随机数状态)、
  `epoch_NNN.pt`,最好的权重存 `best.pt`,训练结束写成 `model.pt`;`train_log.csv` 每个 epoch 一行(含训练 / 验证耗时)。
  manifest 记录 final_epoch、early_stop_epoch、best_epoch。
- 断点续训:打乱顺序只由(种子, epoch)决定,随机数状态随 checkpoint 保存。测试中在第 2 个 epoch 后中断再续训,
  三个模型的权重、训练记录、分数都与一次跑完的**逐位一致**(CPU)。已完成的运行用相同配置带 `--resume` 再次提交时直接结束("无事可做"),
  用于 `--dependency=afterany` 串接。
- LSTM-AE(`lstm_ae.py`,hidden 64、1 层,68,473 个参数):编码器最后隐状态重复 W 次 → 解码器 LSTM → Linear;
  分数 = 窗口 MSE,per_sensor_err = 最后一秒各传感器的平方误差。主模型配置 W=60、hidden=64 由测试固定。
- LSTM-VAE(`lstm_vae.py`,latent 32,β = 1.0):损失 = 重构 + β·KL,两项都按每个窗口求和
  (重构 = 0.5 × 窗口内平方误差之和,β = 1 时即负 ELBO);训练时采样,验证和打分时取 z = μ,分数 = 窗口 MSE。
  (可逆的实现选择:如果重构用逐元素均值而 KL 用求和,KL 会压倒重构项导致后验坍塌,所以两项统一用求和。)
- Deep SVDD(`deep_svdd.py`,rep_dim 32,eps 0.1):LSTM 与投影层都没有 bias,没有归一化层;中心 c = 初始网络在全部训练窗口上输出的均值,
  |c| < eps 的分量设为 ±eps,之后固定(由测试确认训练中不变)。分数 = 到 c 的平方距离,不输出逐传感器误差。
  **观察(冒烟,需在集群结果中复核)**:无 bias 的初始网络输出很接近 0,c 的 32 维全部被设为 ±0.1,即 c 实际上只保留了符号。
  这符合已定的规则,但意味着 c 几乎不携带数据信息;如果完整训练出现坍塌或效果差,这是首先要说明的原因。
- 坍塌检查:threshold 阶段记录 val_thr 分数的方差和变异系数 `score_cv`,低于 `ml.collapse.min_score_cv`(0.001)时
  `threshold.json` 的 `collapsed` 为 true 并写警告日志。
- 诊断闸门只对 PCA 生效(`ml.diagnosis.model`);LSTM-AE / VAE 也输出 `val_sensor_contrib.csv`,但不会因此停下。
- 计时作业只跑 `--stages prepare,train`(1 个 epoch 的模型不去读 test)。
- 本地冒烟(1% 数据、1 个 epoch,`bash cluster/train_one.sbatch --smoke <模型>`):三个模型六个阶段都跑通,各约 13 秒,
  含 test1 事件片段的事件级指标也跑到了。
- 测试:新增 18 个(共 194 个,全部通过,约 105 秒)。

### M3 / M4 / M5 / I1 的脚本(2026-10-07,只用冒烟产物开发和测试)

**没有任何正式结果。** 这些脚本的正式输出要等集群产物;现在只确认了流程能跑通、规则有测试。
冒烟产物生成的表、图、alerts 全部在 `artifacts/smoke_reports/`(不入库),没有写进 `reports/main/`,也没有生成 `data/processed/alerts.json`。

- 来源规则统一在 `src/utils/reporting.py`:正式模式只读集群完整运行、写 `reports/main/`;`--smoke` 只读 `artifacts/smoke/` 下的冒烟运行、
  写 `artifacts/smoke_reports/`。两种模式不混用,测试确认正式模式会拒绝放在正式位置的冒烟产物。`make_main_report` 的拒绝规则未变(改为复用同一个函数)。
- M3 `scripts/make_figures.py`:每个测试文件一张分数曲线图(各模型一个子图,对数纵轴、阈值线、攻击区间阴影)、分数分布
  (val_thr / test 正常 / test 攻击)、训练曲线、`per_subsystem_detection.csv`。攻击区间取自 evaluate 写出的 `eval_events.csv`,画图脚本不读标签。
- M4 `scripts/make_ablation_report.py`:`ablation.csv` / `.md`、W × hidden 的色块图(主模型配置加框)、`ablation_quantiles.csv`
  (阈值分位 99 / 99.5 / 99.9 的敏感性;由 evaluate 阶段用同一套指标重算,写在 `eval_metrics.json` 的 threshold_sensitivity)。缺少的组合留空并列出。
- M5 `src/attribution.py`(标签白名单内):每个报警段的 Top-3 / Top-5、子系统命中率、Top-1 落在 P4 的比例、
  只与无子系统标签事件重叠的报警段单列、不与攻击重叠的报警段单列;另按事件汇总并画"事件 × 子系统"误差占比热力图。
  报警段与多个事件重叠时(2026-10-07 定,写入决策 7):主口径按重叠时间最长的事件判断(相同取较早的),
  并集口径作为补充一并输出(`hit_union`、`subsystem_hit_rate_union`),汇总里有与多个事件重叠的报警段数量(`n_multi_event_segments`)。
  **传感器级 Hit@k 未做**:HAI 21.03 的 CSV 没有逐传感器的攻击目标。官方攻击目标表由用户去找,找到后补做;找不到就不做,写进报告的局限性。
- I1 `scripts/export_alerts.py`:只从 LSTM-AE 的产物导出,不读标签;字段与 mock 完全一致(多 `model`、少 `mock_case`,`is_mock=false`)。
  契约测试(pydantic,禁止多余字段)对 mock 和导出结果都通过,并通过扩展部分实际使用的 `Alert` 模型;另有测试确认导出的报警与归因用的是同一批报警段。
  share 是相对全部传感器的占比(Top-5 之和不超过 1;mock 里是和为 1)。扩展部分的融合打分会自行归一化,接口未改。
  告警卡片(Markdown / HTML;Streamlit 复用同一模板)的列标题改为"占全部传感器误差的比例"。
  一键流水线和 Streamlit 切到真实报警属于 I1 的后续,等真实 alerts.json 生成后再做。
- 公共部分:`src/eval/alarms.py`(报警段、占比、Top-k,归因与导出共用,内部调用 `to_segments`)、`src/eval/run_io.py`、`src/utils/plotstyle.py`。
- Deep SVDD:`run_manifest.json` 的 `stages.train.deep_svdd_center`(从头训练的那一段同时写在顶层)记录替换前 |c| 的最小值、中位数、最大值和被替换的维数;
  随 checkpoint 保存,续训后不丢。规则未改。
- 方法部分草稿 `reports/main/draft/methods.md`:数据与重叠处理、预处理、五个模型、阈值、指标约定、归因规则;不含实验数字,需要结果的位置标了【待填】。
- 测试:新增 12 个(共 206 个,全部通过)。
- 拿到集群产物后从同步到生成全部表图的命令顺序见 `docs/cluster_queue.md` 末尾。

### 待决定 / 待提供

0. 集群步骤全部在 `docs/cluster_queue.md`:基线 → LSTM-AE 计时作业 → LSTM-AE 完整训练 → LSTM-VAE → Deep SVDD。
   M3 及之后的阶段需要集群产物,本地做到 M2c 为止。
1. 已决定(2026-10-07):代码传输用 rsync;push 只推 git 跟踪的文件;作业时长为 TODO 时不写 `--time` 行。
   push 前检查工作区,有未提交的修改就拒绝并列出文件,`--allow-dirty` 才放行(manifest 标记 dirty=true);
   用于报告的产物,其 run_manifest.json 的 git_dirty 必须为 false。
2. cluster.yaml 已全部补全(2026-10-09 填入 `jobs.ablation.time`)。集群上的代码仍是 64ede38,其作业脚本没有 `--time` 行;再提交作业前先推送,或显式带 `--time`。

## 环境

- Greenbone Community Containers（Docker Compose），gvmd socket bind mount 到 `/tmp/gvm/gvmd/gvmd.sock`。
- 实验网：host-only `192.168.56.0/24`；宿主 192.168.56.10；Metasploitable2 在 192.168.56.11（映射为 ews_01）。
- 其余资产（ews_02、hmi_main、historian）暂无靶机，`scanned: false`；PLC 按 OT 实践不扫描。

## 已定决策

- 数据集 HAI 21.03。列名见 `data/samples/hai_columns.txt`：`time` + 79 个传感器（P1 38 / P2 22 / P3 7 / P4 12）
  + 4 个标签列（attack、attack_P1~P3）；train1 中标签全为 0。
- P4 为 HIL 仿真（名称按 HAI 官方说明；官方描述：仿真汽轮机发电与抽水蓄能发电）：assets 为空，E_sub 恒为 0，criticality 0.6；卡片附说明，建议同时排查 P1/P2/P3(P4 与三者都有物理耦合)。
- 上游资产 = 子系统 assets ∪ 沿 edges 反向可达；入侵路径 = 以子系统 PLC 为终点、以无入边资产为起点的极大简单路径。
- 跨子系统加权：w_sub 按 Top-k 误差占比聚合后在可识别传感器上归一化；未知前缀传感器不参与，列入 unmapped_sensors；
  全部未知时 priority = 0。
- QoD：parse_report 不过滤；融合时 min_qod=70；低于阈值但命中 KEV 的列入 needs_review（卡片显示"需人工核实"）。
- EPSS 无记录的 CVE 记 0.0，缓存空行超过 7 天重新查询。
- 漏洞排序：r_v 相同（截断到 1 时常见）依次按 KEV、CVSS、EPSS 排序；按 EPSS 优先会把 CVSS 10 的后门（EPSS 0）挤出前 10。
- 告警卡片：`python -m src.fusion.render_card` → `reports/cards/`（NN_ID.md/.html + index.md/.html），
  建议动作 = 优先排查资产 → 需核实 KEV → 未扫描资产 → 子系统说明；修复建议按修复文本去重取前 5 条。
- 多份报告：config.yaml 的 `vuln.report_xmls` 为列表（每台靶机一份），parse_report 逐份做三数核对后合并，
  跨报告同样按 (asset_ip, port, nvt_oid) 去重；vulns.json 以 `sources` 记录每份报告的 id、文件、主机。
- **E3 跳过（2026-09-30 决定）**：不搭建第二台靶机，hmi_main 保持 `scanned: false`。原因：只有一台靶机，
  接受风险饱和，作为局限性写入报告。准备部分保留为未来工作：多报告合并、`scripts/compare_fusion.py`、
  搭建说明 `docs/e3_hmi_main_setup.md`（Ubuntu 22.04 全量更新，只开 22/80/443，静态 192.168.56.20）。
- 未扫描资产：风险计算中按 0 处理，但在 enriched_alerts 中 `R_asset: null`、`assessed: false`，卡片显示"未评估"；
  路径上存在未评估资产时，卡片提示"该路径存在未评估资产，实际风险可能被低估"。
  子系统上游资产全部未评估时（如 P3），E_sub 显示为"未评估（按 0 计算）"。
- 一键流水线：`python -m src.pipeline [--offline] [--format md|html|both]`，依次 parse → intel → prioritize → render，
  任一步失败即停（退出码 parse 4 / intel 3 / 其他 2）。输出 JSON 除 generated_at 外内容未变时不改写，
  避免每次运行都改动已提交的快照。
- Streamlit：侧边栏调 α/β/γ/min_qod 实时重算，只影响页面、不写文件；告警列表显示相对默认参数的排名变化；
  卡片复用 `alert_card.md.j2`。页面由 AppTest 冒烟测试覆盖；headless 截图无法等到 websocket 渲染，未做目视检查。
- 报告素材：`python -m scripts.make_extension_report` → `reports/extension/`（asset_topology.png、qod_distribution.png、
  kev_table.md、key_numbers.md），全部基于真实数据，入库；`e3_compare.md` 为 mock 派生，不入库。
  配色：蓝 = 保留 / 已评估，灰 + 斜线纹理 = 丢弃 / 未评估（灰色对比度不足，按规则加纹理与数值标签补救）。
- 提交策略：真实数据快照（vulns.json、vulns_enriched.json）提交；mock 派生输出（enriched_alerts.json、reports/cards/）不提交。

## 关键数字与发现(扩展部分)

### M7 导出漏掉 101 条 Critical（已修复）
- 现象：首次导出的报告漏掉了全部 101 条 Critical（severity ≥ 9.0）结果。
- 原因：GMP 报告过滤串的 `levels` 只写了 `hml`（High/Medium/Low）。新版 gvmd 新增了 Critical 级别（c，severity ≥ 9.0），
  不在 levels 里的结果被静默丢弃，报告里也不报错。
- 修复：`scripts/smoke_scan.py` 使用 `apply_overrides=0 min_qod=0 rows=-1 first=1 levels=chmlg`；
  parse_report 读取前做三数核对（实际 result 条数 == result_count/filtered == result_count/full），不一致直接失败。
- 修复后：630 条结果，丢弃 Log 90 条，去重 0 条，写入 540 条，其中 Critical 101 条（测试 `test_all_critical_kept` 固定此数）。

### M8 威胁情报（KEV 2026.09.29，EPSS 日期 2026-09-29）
- 540 条漏洞涉及 1086 个唯一 CVE（非预估的 ~799）；EPSS 分 11 批查询（每批 ≤100 个、cve 参数 ≤2000 字符）；5 个 CVE 无 EPSS 记录。
- EPSS 分布（按漏洞，取 CVE 最大值）：>0 493 条，>0.01 448，>0.1 208，>0.5 89，>0.9 33；无 CVE 的漏洞 47 条。
- 命中 KEV 9 条（8 个 CVE），全部 solution_type = VendorFix：

| QoD | KEV CVE | 漏洞 |
|---|---|---|
| 99 | CVE-2020-1938 | Tomcat AJP Ghostcat（主动检测） |
| 95 | CVE-2012-1823 | PHP-CGI（主动检测） |
| 30 | CVE-2009-1151 ×2 | phpMyAdmin 代码注入 |
| 30 | CVE-2019-11043 | PHP-FPM RCE |
| 30 | CVE-2024-38475 | Apache httpd < 2.4.60 |
| 30 | CVE-2021-40438 | Apache httpd < 2.4.49 |
| 30 | CVE-2024-4577 | PHP-CGI 参数注入 |
| 30 | CVE-2020-11023 | jQuery XSS |

### QoD 过滤
- min_qod=70 后剩 **66 / 540** 条（丢弃 88%），其中命中 KEV **2** 条；其余 7 条 KEV 命中（QoD 30，仅版本横幅比对）进入 needs_review。
- 写报告时需说明：QoD 30 为远程横幅比对，Metasploitable2 上多为真阳性，但无法排除发行版回移补丁造成的误报。

### M9 融合打分（真实 vulns_enriched.json + mock 报警）
- R_asset：ews_01 = 1.000（66 条参与计算，7 条需核实）；其余资产 0（未扫描）。
- E_sub：P1 = 1.000、P2 = 1.000（均来自 ews_01），P3 = 0，P4 = 0。
- mock 报警排序：MOCK-004（严重，P2）2.000 > MOCK-002（P2+P1）1.544 > MOCK-001（P1）0.900
  > MOCK-005（P4+P1）0.324 > MOCK-003（刚过阈值，P3）0.014。
- **风险饱和**：ews_01 上有 18 条漏洞仅凭 CVSS 10 即达到 r_v = 1.0（如 Ingreslock 后门），R_asset/E_sub 封顶，
  E_sub 只剩 0/1 两个取值，排序实际上只由 S、criticality 以及是否连到 ews_01 决定。
  这是公式 min(1, ·) 与单靶机共同导致的。E3 已跳过，接受饱和，报告"局限性"一节需写明：
  只有 ews_01 一台靶机被评估，其余上游资产均未评估，E_sub 可能被低估，且无法区分 P1 与 P2 的暴露程度。
