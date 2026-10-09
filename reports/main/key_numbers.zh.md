## 关键数字（主线）

写报告时直接引用。数字由 `python -m scripts.make_key_numbers` 从来源文件读出（本节是 `reports/main/key_numbers.zh.md` 的副本）；检测、消融、归因的数字全部来自集群上的完整运行（git_dirty=false），冒烟测试的数字没有进入这里。
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
