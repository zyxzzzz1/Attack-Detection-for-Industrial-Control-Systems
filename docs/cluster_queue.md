# 集群提交清单（按依赖顺序）

拿到 Minerva 权限后按顺序提交。每个阶段的本地部分完成时把命令追加到这里。
所有命令都在**登录节点**上执行，只做提交（`sbatch`）；标 **[本地]** 的在虚拟机上执行。
环境搭建、数据准备、查看状态和日志的细节见 `docs/cluster_runbook.md`。

## 0. 前置（一次性）

1. 按 runbook 第 1 步查询集群信息，补全 `config/cluster.yaml` 的 `paths.*`，重新生成作业脚本并**提交**
   （`python -m scripts.make_cluster_jobs`，`git commit`）。工作区不干净时 push 会被拒绝，`run_manifest.json` 的
   `git_dirty` 必须为 false 才能用于报告。
2. runbook 第 2~4 步：推送代码、建环境、准备数据。
3. runbook 第 5 步：`srun` 跑完整 pytest，**全部通过后才允许提交下面的作业**；再用 `sbatch cluster/pytest.sbatch` 确认 GPU 可用。
4. runbook 第 6 步：冒烟（可选，但建议第一次做）。

下面的命令默认已经执行过：

```bash
cd "$CLUSTER_CODE_DIR" && source cluster/cluster_env.sh
```

`cluster.yaml` 的 `jobs.*.time` 还没填时，每条 `sbatch` 都要显式带 `--time`（下面已写出）。

## 1. 基线：PCA + IsolationForest（M2a）

```bash
sbatch --time=01:00:00 cluster/baselines.sbatch
```

- 资源：CPU 16 核、64G，不申请 GPU。本地估计 PCA 不到 1 分钟；IsolationForest 给 40 万个窗口算特征和打分较慢，1 小时是保守值。
- PCA 的 `diagnose` 阶段会重新计算 val_thr 上各传感器的误差占比。本地诊断的结果是最大 9.0%（P3_FIT01），远低于 30%，
  预期不会停下。如果作业以退出码 3 停下（`run_manifest.json` 的 status 为 `stopped_for_review`），**不要放行**，拉回产物交给 Claude。
- 跑完后 **[本地]**：

  ```bash
  scripts/pull_artifacts.sh pca iforest
  .venv/bin/python -m scripts.make_main_report      # 主表前两行 + 诊断图；会自动与本地诊断结果比较并报告是否一致
  ```

- 把 `sacct -j <作业号> --format=JobID,Elapsed,MaxRSS,State` 的结果一并告诉 Claude（用来填 `jobs.baselines.time`）。

## 2. LSTM-AE 计时作业（M2b，依赖：第 1 步完成且诊断未停下）

```bash
sbatch --time=01:00:00 cluster/train_one.sbatch lstm_ae --run-name timing/lstm_ae \
    --stages prepare,train --set ml.train.max_epochs=1
```

- 只训练 1 个 epoch 的全量数据，只执行 prepare 和 train（不打分、不读取 test）。
- 跑完后 **[本地]** `scripts/pull_artifacts.sh timing/lstm_ae`，把结果交给 Claude：它根据 `train_log.csv` 里的
  `train_seconds`、`val_seconds` 估算完整训练（最多 50 个 epoch）和消融（9 个组合）的时长，判断是否超过单作业时限，
  并给出第 3 步的 `--time`。**估算消融总耗时超过 6 小时时，Claude 会给出缩减方案，由你决定。**

## 3. LSTM-AE 完整训练（M2b，依赖：第 2 步的估算）

```bash
sbatch --time=<第 2 步估算值> cluster/train_one.sbatch lstm_ae
```

- 资源：`-p long --gres=gpu:L4:1 --cpus-per-task=4 --mem=32G`（写在脚本里）。
- 估算超过 1 天时按 runbook 第 9 步用 `--dependency=afterany` 串接多段；前一段已经跑完时，后一段发现无事可做会立即结束。
- 被中断（到时限、节点故障）时原样重新提交同一条命令，会从最近的 checkpoint 接着训练。
- 跑完后 **[本地]** `scripts/pull_artifacts.sh lstm_ae`。

## 4. LSTM-VAE（M2c，依赖：第 2 步的估算；与第 3 步互不依赖，可以同时提交）

```bash
sbatch --time=<与第 3 步相同> cluster/train_one.sbatch lstm_vae
```

- 跑完后 **[本地]** `scripts/pull_artifacts.sh lstm_vae`。

## 5. Deep SVDD（M2c，依赖同上，可以同时提交）

```bash
sbatch --time=<与第 3 步相同> cluster/train_one.sbatch deep_svdd
```

- 只有编码器，每个 epoch 比 LSTM-AE 快，用同样的 `--time` 足够。
- 跑完后 **[本地]** `scripts/pull_artifacts.sh deep_svdd`。`threshold.json` 里的 `collapsed`、`score_cv` 用来检查是否坍塌。

## 6. LSTM-AE 消融（M4，依赖：第 2 步的估算；估算总耗时超过 6 小时时先等缩减方案的决定）

```bash
sbatch --time=<第 2 步估算值> cluster/ablation_array.sbatch      # --array=0-8%3，9 个 (W, hidden) 组合
```

- 任务 4（W=60、hidden=64）与主模型配置相同，但单独训练一次，放在 `ablation/` 下，不替代第 3 步的主模型运行。
- 只重跑某个任务：`sbatch --time=... --array=4 cluster/ablation_array.sbatch`。
- 跑完后 **[本地]** `scripts/pull_artifacts.sh ablation`。

## 7. LSTM-AE 多随机种子（2026-10-09 追加，依赖：第 3 步；与本地的 M5 互不依赖）

只对主模型配置（W=60、hidden=64）做，种子见 `config.yaml` 的 `ml.seed_study.seeds`。种子 42 就是第 3 步的主模型运行，不重跑；
其余 4 个种子各提交一个作业，只带一个覆盖项 `ml.seed=<种子>`，运行目录 `seeds/lstm_ae_s<种子>/`：

```bash
for s in 43 44 45 46; do
    sbatch --time=00:15:00 cluster/train_one.sbatch lstm_ae --run-name seeds/lstm_ae_s$s --set ml.seed=$s
done
```

- 集群上的代码不需要更新（运行名和覆盖项都是已有的参数）。
- 跑完后 **[本地]** `scripts/pull_artifacts.sh seeds`，再运行 `.venv/bin/python -m scripts.make_main_report`：
  5 个种子都可用时主表 `.md` 中 LSTM-AE 一行的 f1、auroc 写成"均值 ± 标准差"，并生成 `main_results_seeds.csv`；
  缺任何一个种子、或某次运行带了别的覆盖项时维持单种子并打印原因。
- 报警导出、归因和图仍只用种子 42 的主模型运行；其余模型维持单种子。

## 全部跑完后 [本地]：从同步到生成全部表图

按顺序执行。每个脚本只接受集群上的完整运行（非冒烟、非本地诊断、`git_dirty=false`、`status=completed`），
不合格的运行会被跳过并打印原因；看到 `[跳过]` 时先把输出交给 Claude，不要绕过。

```bash
# 1. 同步产物（默认不含 checkpoint 和模型权重）
scripts/pull_artifacts.sh

# 2. M3：主结果表 + val_thr 诊断（并与本地诊断结果比较，报告是否一致）
.venv/bin/python -m scripts.make_main_report
# 3. M3：每个测试文件的分数曲线、分数分布、训练曲线、分子系统检出表
.venv/bin/python -m scripts.make_figures
# 4. M4：消融表与图
.venv/bin/python -m scripts.make_ablation_report
# 5. M5：归因表、子系统命中率、P4 比例、热力图（LSTM-AE、LSTM-VAE、PCA）
.venv/bin/python -m src.attribution
# 6. I1：导出真实报警 data/processed/alerts.json（只用 LSTM-AE）
.venv/bin/python -m scripts.export_alerts
# 7. I1：融合排序评估 precision@k → reports/main/fusion_ranking_eval.csv/.md
.venv/bin/python -m src.eval.fusion_ranking
# 8. I1：一键流水线（真实报警，卡片只生成优先级前 50 条）→ reports/cards/
.venv/bin/python -m src.pipeline --offline
# 9. I1：按固定规则挑 3 个案例 → reports/main/cases.md、reports/main/cases/
.venv/bin/python -m scripts.make_cases
# 10. R1：把各次运行的小文件脱敏后复制到 reports/main/runs/（入库），再汇总关键数字 → reports/main/key_numbers.md
#     （一条命令完成；PROGRESS.md 顶部的"关键数字"是它的副本，改动后要同步）
.venv/bin/python -m scripts.make_key_numbers
```

输出位置：第 2~5 步写到 `reports/main/`（入库），第 6 步写到 `data/processed/alerts.json`（入库）。
做完后告诉 Claude，由它检查每个运行的 `run_manifest.json`（git_dirty=false、status=completed、设备为 L4）、训练曲线、
每个 epoch 的耗时、Deep SVDD 的中心统计量与是否坍塌，再写 M2a~M5 的汇报；I1 的后续（一键流水线切到真实报警、
Streamlit 默认读真实数据、案例）在那之后进行。

开发和检查流程时可以给第 3~6 步加 `--smoke`：读 `artifacts/smoke/` 下的冒烟产物，写到 `artifacts/smoke_reports/`，
不会碰 `reports/main/` 和 `data/processed/`。冒烟的数字不得写进报告。
