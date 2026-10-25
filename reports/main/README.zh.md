# reports/main：主线报告素材

[English](README.md) | **中文**

全部由脚本从集群完整运行的产物生成（非冒烟、`git_dirty=false`、已完成）；冒烟测试的数字没有进入这里。
检测指标**不使用 point-adjust**，阈值为 val_thr 分数的 99% 分位数。写报告时先看 `key_numbers.md`。

| 文件 | 内容 | 生成命令 |
|---|---|---|
| `key_numbers.md` | 关键数字汇总：检测 / 归因 / 融合 / 数据问题，每个数字带口径和来源 | `python -m scripts.make_key_numbers` |
| `data_summary.md`、`attack_events.csv` | 原始数据统计；train 与 test 的重叠；50 个攻击事件及其子系统 | `python -m scripts.hai_data_summary` |
| `main_results.csv/.md` | 主结果表（5 个模型）。`.md` 中 LSTM-AE 一行的 F1、AUROC、事件检出率、误报率是 5 个种子的均值 ± 标准差；csv 是种子 42 的运行 | `python -m scripts.make_main_report` |
| `main_results_seeds.csv` | LSTM-AE 主配置 5 个种子的逐种子结果与均值、标准差 | 同上 |
| `val_sensor_contrib.csv/.png` | PCA 在 val_thr 上各传感器的误差占比（缩放方式的事先检查） | 同上 |
| `scores_test1~5.png` | 每个测试文件的分数曲线（阈值线、攻击区间阴影） | `python -m scripts.make_figures` |
| `score_distribution.png`、`training_curves.png` | 分数分布（val_thr / test 正常 / test 攻击）；训练曲线 | 同上 |
| `per_subsystem_detection.csv` | 分子系统的事件检出 | 同上 |
| `ablation.csv/.md`、`ablation_heatmap.png`、`ablation_quantiles.csv` | LSTM-AE 消融（W × hidden）与阈值分位敏感性；只作敏感性分析 | `python -m scripts.make_ablation_report` |
| `attribution_summary.csv/.md` | 归因汇总：子系统命中率（报警段 / 事件两种口径）、95% Wilson 区间、P4 比例 | `python -m src.attribution` |
| `attribution.csv` | 每个报警段一行（三个模型），含 Top-3 / Top-5 | 同上 |
| `attribution_events_<模型>.csv`、`attribution_heatmap_<模型>.png` | 按事件汇总；事件 × 子系统的误差占比热力图 | 同上 |
| `attribution_by_duration.csv` | 按报警段长度分组的命中率 | 同上 |
| `attribution_seeds.csv` | LSTM-AE 5 个种子的归因命中率 | 同上 |
| `attribution_exploratory_valnorm.csv` | **探索性的事后分析**：按 val_thr 平均误差归一化前后的命中率；不替换主结果 | 同上 |
| `fusion_ranking_eval.csv/.md` | 融合排序评估：precision@k（原始分数、α=0、α=1） | `python -m src.eval.fusion_ranking` |
| `cases.md`、`cases/` | 3 个案例及其告警卡片（按固定规则挑选） | `python -m scripts.make_cases` |
| `runs/<运行名>/` | 各次集群运行的小文件的入库副本：`eval_metrics.json`、`run_manifest.json`、`threshold.json`、`preprocess.json`、`train_log.csv`（5 个模型、9 个消融组合、4 个额外种子）。主机名等集群信息已替换为占位符，其余内容与产物目录中的原件相同。关键数字从这里读取 | `python -m scripts.make_key_numbers`（内部调用 `scripts.snapshot_runs`） |
| `draft/methods.md` | 报告"方法"一章的草稿 | 手写 |

相关文件：真实报警 `data/processed/alerts.json`（`python -m scripts.export_alerts`）；扩展部分的素材在 `reports/extension/`。
从同步集群产物到生成全部表图的命令顺序见 `docs/cluster_queue.md` 末尾。

上面各脚本生成的图表、表格和文档全部是英文；本文件是 `README.md` 的中文副本。

这些文件中引用的约束和决策编号（如"约束 7""决策 6b"，或写作"CLAUDE.md 约束 …"）的内容见
[`docs/design_decisions.md`](../../docs/design_decisions.md)（英文）。

## 读这些结果时要注意

- 逐点 F1 偏低是不使用 point-adjust 的预期结果，评估方法没有为提高指标而改动。
- 所有模型在 test 上的报警比例都高于 val_thr 上的 1%：阈值从验证集迁移到测试集的效果不好，推测是正常工况的漂移（没有单独验证）。
- 主模型配置（W=60、hidden=64）事先固定，在消融的 9 个组合里按 F1 排第 5；没有据此修改。
- 除 LSTM-AE 主配置（5 个种子）外，所有结果都来自单个随机种子。
- 归因命中率的分母很小（53~72 个报警段、48~50 个事件），置信区间很宽；"总是猜 P1"就有约 0.68 的命中率，要对照这条参照线来读。
- 融合部分的资产映射是人为构造的，漏洞数据来自一台实验靶机，与 HAI 中的攻击没有因果关系；不声称提升检测指标。
