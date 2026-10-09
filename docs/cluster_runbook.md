# 集群操作手册（Minerva）

按顺序执行。标 **[本地]** 的在虚拟机上执行，标 **[登录节点]** 的在 `ssh minerva` 之后执行。

**规则：登录节点上只做 venv 创建、pip 安装、rsync、git、提交和查看作业。pytest、冒烟测试、训练、打分、评估、
数据核对全部通过 `srun` 或 `sbatch`。** 下面每条命令都遵守这一条；不要把 `srun` 去掉直接运行。

当前状态（M2c 本地部分）：五个模型的全部阶段已实现（prepare → train → diagnose → score → threshold → evaluate）。
**提交顺序和每一步的具体命令见 `docs/cluster_queue.md`**；本文件说明环境搭建、数据准备和通用操作。

> **个人信息不入库（2026-10-09 起）**：登录用户名和集群上的绝对路径写在 `config/cluster.local.yaml`（不入库，示例见
> `config/cluster.local.example.yaml`），入库的 `config/cluster.yaml` 中这些项是 TODO。`python -m scripts.make_cluster_jobs`
> 另外生成真实的作业脚本到 `cluster/local/`（不入库），`scripts/push_to_cluster.sh` 推送代码后把它们放到集群的 `cluster/` 下，
> 所以下文在集群上执行的命令不变。下文说"补全 cluster.yaml 的 paths.*"的地方，现在指补全 `cluster.local.yaml`。

## 0. 前置 [本地]

1. 连上 Chalmers VPN（或在 Chalmers 网络内）。
2. 在 `~/.ssh/config` 中配置别名（CID 换成自己的）：

   ```
   Host minerva
       HostName minerva.cse.chalmers.se
       User CID
   ```

3. 确认能连通：`ssh minerva true && echo OK`

## 1. 查询集群信息并补全 cluster.yaml

**[登录节点]**（都是轻量查询）：

```bash
sinfo -o "%P %l %D %c %m %G"            # 分区、时限、节点数、CPU、内存、GPU；带 * 的是默认分区
scontrol show partition | grep -E "PartitionName|MaxTime|DefaultTime"
sacctmgr -nP show assoc user=$USER format=account,partition,qos    # 账户与 QoS
python3 --version                        # 应为 3.12.x
quota -s 2>/dev/null; df -h $HOME        # home 配额
ls -d /scratch* /local* /data* 2>/dev/null; echo "TMPDIR=$TMPDIR"   # 是否有 scratch
# 下面两条在计算节点上执行（各约几秒）
srun --time=00:02:00 bash -c 'hostname; curl -sI --max-time 10 https://pypi.org | head -1 || echo "计算节点不能上网"'
srun --gres=gpu:L4:1 --time=00:02:00 nvidia-smi          # 右上角 CUDA Version = 驱动支持的最高 CUDA 版本
```

**[本地]** 把结果填进 `config/cluster.yaml`（全部 TODO 项的清单见 `docs/cluster_notes.md`），至少填好 `paths.*`，然后：

```bash
.venv/bin/python -m scripts.make_cluster_jobs      # 重新生成 cluster/*.sbatch 和 cluster_env.sh，并列出剩余 TODO
.venv/bin/python -m pytest -q
git add -A && git commit -m "C0: 补全 cluster.yaml"   # 正式运行前先提交，manifest 才不会记为 dirty
```

## 2. 推送代码 [本地]

```bash
scripts/push_to_cluster.sh --dry-run    # 先看会传什么
scripts/push_to_cluster.sh
```

只推送 git 跟踪的文件（不含 `data/raw/`、`artifacts/`、`.venv/`），并写入 `.git_commit`。连不上时会提示检查 VPN。

工作区有未提交的修改时脚本拒绝推送并列出这些文件：先提交再推。调试时可加 `--allow-dirty` 放行，
此时 `.git_commit` 和 `run_manifest.json` 标记 `dirty=true`；**正式运行的产物不得来自 dirty 的代码**。

## 3. 搭建环境 [登录节点]

```bash
cd <代码目录>
source cluster/cluster_env.sh            # 读入 CLUSTER_* 变量，下面的命令都用它们
python3 -m venv "$CLUSTER_VENV_DIR"
"$CLUSTER_VENV_DIR/bin/pip" install --upgrade pip
"$CLUSTER_VENV_DIR/bin/pip" install --no-cache-dir -r env/requirements-cluster.txt
mkdir -p "$CLUSTER_DATA_DIR" "$CLUSTER_ARTIFACT_DIR/slurm_logs"    # slurm 日志目录必须在提交前存在
```

torch 的 CUDA 构建与驱动不匹配时的处理见 `env/README.md`。

## 4. 准备数据

方式 A：登录节点能上网。**[登录节点]** 下载（网络 I/O），核对放到计算节点上：

```bash
export HAI_DATA_DIR="$CLUSTER_DATA_DIR"
"$CLUSTER_VENV_DIR/bin/python" -m scripts.download_hai            # 下载 8 个文件共 187 MB，并按官方仓库的大小与 sha 核对
```

方式 B：登录节点不能上网。**[本地]** 推送数据，**[登录节点]** 用 srun 离线核对：

```bash
scripts/push_to_cluster.sh --data        # [本地] rsync data/raw/hai-21.03/ 下的 *.csv.gz 和 download_manifest.json
```

```bash
# [登录节点]
HAI_DATA_DIR="$CLUSTER_DATA_DIR" srun --time=00:05:00 "$CLUSTER_VENV_DIR/bin/python" -m scripts.download_hai --offline
```

两种方式都应看到 8 行 `[一致]`。

## 5. 第一个作业：完整 pytest [登录节点]

全部通过后才允许提交训练作业。

```bash
cd "$CLUSTER_CODE_DIR" && source cluster/cluster_env.sh
HAI_DATA_DIR="$CLUSTER_DATA_DIR" ARTIFACT_DIR="$CLUSTER_ARTIFACT_DIR/pytest" \
    srun $CLUSTER_PYTEST_SRUN_ARGS "$CLUSTER_VENV_DIR/bin/python" -m pytest -q
```

- 预期：全部通过，没有失败。带真实数据时不应有因"没有完整的 HAI 数据"而跳过的测试。
- 默认分区时限不足 20 分钟时，加 `-p long`。

再确认 GPU 可用（在一个 L4 节点上重跑 pytest 并检查 torch 能用 GPU）：

```bash
sbatch cluster/pytest.sbatch
squeue -u $USER
tail -n 5 "$CLUSTER_ARTIFACT_DIR"/slurm_logs/hai-pytest_*.out     # 应有 "cuda True NVIDIA L4"
```

## 6. 冒烟测试 [登录节点，通过 srun]

1% 数据、1 个 epoch，产物在 `产物目录/smoke/`，数字不写进报告。

`srun` 必须带 `--ntasks=1`：Minerva 上申请 GPU 的 `srun` 不写任务数时会启动两个任务（2026-10-08 实测），
两个进程同时写同一个运行目录会互相破坏产物。`sbatch` 提交的作业脚本只执行一次，不受影响（脚本头里也写了 `--ntasks=1`）。

```bash
srun -p long --ntasks=1 --cpus-per-task=16 --mem=64G --time=00:15:00 bash cluster/baselines.sbatch --smoke
srun -p long --ntasks=1 --gres=gpu:L4:1 --cpus-per-task=4 --mem=32G --time=00:15:00 bash cluster/train_one.sbatch --smoke lstm_ae
srun -p long --ntasks=1 --gres=gpu:L4:1 --cpus-per-task=4 --mem=32G --time=00:15:00 bash cluster/ablation_array.sbatch --smoke 0
```

## 7. 计时作业（M2b 第一次运行）[登录节点]

只跑 1 个 epoch 的全量数据，用来估算完整训练和消融的时长：

```bash
sbatch --time=01:00:00 cluster/train_one.sbatch lstm_ae --run-name timing/lstm_ae \
    --stages prepare,train --set ml.train.max_epochs=1
```

跑完后看 `产物目录/timing/lstm_ae/run_manifest.json` 和 `train_log.csv` 中的 epoch 耗时。
`--time` = 实测耗时 × 最大 epoch 数 × 1.5，填进 `cluster.yaml` 的 `jobs.*.time`（重新生成并推送），或提交时用 `--time` 覆盖。
估算超过 1 天时按第 9 步串接。

## 8. 正式提交 [登录节点]

```bash
cd "$CLUSTER_CODE_DIR" && source cluster/cluster_env.sh

sbatch cluster/baselines.sbatch                      # PCA + IsolationForest（CPU）
sbatch cluster/train_one.sbatch lstm_ae              # 主模型
sbatch cluster/train_one.sbatch lstm_vae
sbatch cluster/train_one.sbatch deep_svdd
sbatch cluster/ablation_array.sbatch                 # 消融：--array=0-8%3（并发数来自 cluster.yaml）
```

- `jobs.*.time` 还没填时，每条都要加 `--time=HH:MM:SS`。
- **M2a 只提交第一条**（基线，先用 `--time=01:00:00`；实测耗时乘以 1.5 后填进 `jobs.baselines.time`）。
  作业先跑 PCA：`diagnose` 阶段只用 val_thr 计算各传感器对重构 MSE 的占比，写到 `产物目录/pca/val_sensor_contrib.csv`。
  单个传感器超过 30% 时作业在读取 test 之前停下（退出码 3，`run_manifest.json` 的 status 为 `stopped_for_review`，
  IsolationForest 也不会开始）。这时拉回产物交给 Claude 汇报，决定缩放方式之后再重新提交：
  `sbatch cluster/baselines.sbatch --set ml.diagnosis.allow_over_limit=true`（已训练的模型会直接读取，不重新训练）。
- 任务号与 (W, hidden) 的对应：`"$CLUSTER_VENV_DIR/bin/python" -m src.run_model --list-ablation`（只打印，不计算）。
- 只重跑某个消融任务：`sbatch --array=4 cluster/ablation_array.sbatch`。
- 临时改配置：在脚本后面加 `--set key=value`，会记录在 run_manifest.json 的 `overrides` 中。

查看状态和日志：

```bash
squeue -u $USER
sacct -j <作业号> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
tail -f "$CLUSTER_ARTIFACT_DIR"/slurm_logs/hai-train_<作业号>.out
tail -n 20 "$CLUSTER_ARTIFACT_DIR"/lstm_ae/run.log
scancel <作业号>
```

## 9. 被中断或超过 1 天的作业 [登录节点]

每个 epoch 保存 checkpoint，作业脚本总是带 `--resume`：有 checkpoint 就接着跑，没有就从头开始。

- 作业被中断（到时限、节点故障、scancel）：原样重新提交同一条命令即可。
- 预计超过 1 天：一次提交多段，后一段等前一段结束（无论成功与否）再开始：

  ```bash
  J1=$(sbatch --parsable --time=1-00:00:00 cluster/train_one.sbatch lstm_ae)
  J2=$(sbatch --parsable --time=1-00:00:00 --dependency=afterany:$J1 cluster/train_one.sbatch lstm_ae)
  J3=$(sbatch --parsable --time=1-00:00:00 --dependency=afterany:$J2 cluster/train_one.sbatch lstm_ae)
  echo "$J1 -> $J2 -> $J3"
  ```

  每一段都记录在 `run_manifest.json` 的 `segments` 里。前一段已经用相同的配置跑完时，后一段发现无事可做，
  几秒内结束（日志里有"无事可做"）。训练到一半的段从 `checkpoints/last.pt` 接着训练，结果与一次跑完相同。

## 10. 拉回产物 [本地]

```bash
scripts/pull_artifacts.sh --dry-run          # 先看会拉什么
scripts/pull_artifacts.sh                    # 全部运行目录
scripts/pull_artifacts.sh lstm_ae pca        # 只拉指定的运行目录
scripts/pull_artifacts.sh --with-checkpoints lstm_ae    # 连同 checkpoint 和模型权重
```

默认拉回 `scores_*.npy`、`per_sensor_err_test.npy`、`threshold.json`、`train_log.csv`、`run_manifest.json`、评估结果、
预处理摘要和日志；checkpoint 默认不拉。拉回后在本地生成报告素材（只接受非冒烟、`git_dirty=false`、已完成的运行）：

```bash
.venv/bin/python -m scripts.make_main_report     # reports/main/main_results.csv/.md、val_sensor_contrib.csv/.png
```

然后告诉 Claude，由它检查 `run_manifest.json` 和训练曲线并继续后续步骤。

## 11. 代码更新后

**[本地]** 提交 → `scripts/push_to_cluster.sh` → **[登录节点]** 重跑第 5 步的 pytest → 再提交作业。
依赖有变化时还要重新执行第 3 步的 pip 安装。
