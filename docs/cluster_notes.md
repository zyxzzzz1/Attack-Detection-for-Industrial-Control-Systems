# 集群说明：Minerva（Chalmers CSE）

机器可读的参数在 `config/cluster.yaml`；按顺序执行的命令在 `docs/cluster_runbook.md`。本文记录已知信息、
规则和尚未确认的事项。

## 已知信息

| 项目 | 内容 |
|---|---|
| 登录 | `ssh CID@minerva.cse.chalmers.se`，只能从 Chalmers 网络或 VPN 访问。CID 由使用者填写 |
| ssh 别名 | 本地 `~/.ssh/config` 中配置 `minerva`，同步脚本只用这个别名 |
| 调度器 | SLURM |
| 系统 | 所有节点 Ubuntu 24.04，系统 `python3` 为 3.12 |
| GPU | L4（24 GB，共 32 块）、L40s（48 GB，共 8 块） |
| 分区 | 默认分区时限短（具体值待查）；`-p long` 最长 1 天 |
| 环境 | 在项目目录下 `python3 -m venv .venv`；作业脚本里用绝对路径 source 激活脚本 |

## 硬性规则

**登录节点上禁止运行计算密集任务。**

- 必须通过 `srun` 或 `sbatch` 提交：pytest、冒烟测试、训练、打分、评估、数据核对。
- 登录节点上只做：创建 venv、pip 安装、rsync、git、提交和查看作业、查看日志。
- 作业脚本自带保护：不带 `--smoke` 且不在 SLURM 作业里运行时直接退出。

## 本项目的资源约定

| 作业 | 申请 | 脚本 |
|---|---|---|
| LSTM-AE、LSTM-VAE、Deep SVDD | `-p long --gres=gpu:L4:1 --cpus-per-task=4 --mem=32G` | `cluster/train_one.sbatch` |
| PCA、IsolationForest | `-p long --cpus-per-task=16 --mem=64G`，不申请 GPU | `cluster/baselines.sbatch` |
| LSTM-AE 消融（9 个组合） | 同 LSTM 类，`--array=0-8%3` | `cluster/ablation_array.sbatch` |
| pytest（第一个作业） | `srun --cpus-per-task=4 --mem=8G --time=00:20:00`，默认分区 | runbook 第 5 步 |

- 只申请一块 L4，代码不写多 GPU。
- 训练作业一律 `-p long` 并显式写 `--time`：按冒烟或计时作业的实测耗时乘以 1.5。
- 超过 1 天的作业：靠每个 epoch 的 checkpoint 和 `--resume`，用 `--dependency=afterany` 串接多段。
- DataLoader 的 `num_workers` 不超过 `--cpus-per-task`（代码按 `SLURM_CPUS_PER_TASK` 截断）。
- 所有资源参数来自 `config/cluster.yaml`，由 `python -m scripts.make_cluster_jobs` 写进作业脚本。
- 超参数（batch=256 等）在 CPU 和 GPU 上相同，不因换设备修改。

## 目录与环境变量

| 集群上的目录 | cluster.yaml | 作业中的环境变量 |
|---|---|---|
| 代码目录 | `paths.code_dir` | 作业开始时 `cd` 到这里 |
| 虚拟环境 | `paths.venv_dir`（约定为代码目录下的 `.venv`） | 用绝对路径 `source .../bin/activate` |
| HAI 数据 | `paths.data_dir` | `HAI_DATA_DIR` |
| 产物 | `paths.artifact_dir` | `ARTIFACT_DIR`；slurm 日志在其下的 `slurm_logs/` |

每个运行目录（`产物目录/<模型名>/` 或 `产物目录/ablation/lstm_ae_w<W>_h<hidden>/`）里有 `run_manifest.json`：
git 提交号、配置哈希、完整超参数、Python / torch / sklearn 版本、设备型号、主机名、起止时间、最终 epoch、早停 epoch，
以及 `SLURM_JOB_ID`、`SLURM_ARRAY_TASK_ID`、节点名、`nvidia-smi` 查询到的 GPU 型号。多段作业逐段记录在 `segments` 中。

## 结果的权威来源

- 主表、消融、归因的数字只来自集群上的完整运行。
- 冒烟测试（`--smoke`，1% 数据、1 个 epoch）的产物在 `smoke/` 子目录，数字不写进报告。
- `alerts.json` 只由集群上完整运行的 LSTM-AE 产物生成，在本地导出。

## 待确认（登录后按 runbook 第 1 步查询，填进 cluster.yaml）

| cluster.yaml 字段 | 内容 | 影响 |
|---|---|---|
| `cluster.cid` | Chalmers CID | 仅记录；登录用 ssh 别名 |
| `paths.code_dir`、`venv_dir`、`data_dir`、`artifact_dir` | 集群上的绝对路径 | **补全前作业脚本拒绝在集群上运行** |
| `partitions.default.name`、`max_time` | 默认分区名和时限 | pytest 用默认分区，时限不够时改用 `-p long` |
| `storage.home_quota`、`storage.scratch` | home 配额、是否有 scratch | 决定数据和产物放哪里 |
| `network.compute_nodes_internet` | 计算节点能否上网 | 数据核对用在线还是 `--offline` |
| `network.cuda_driver` | 驱动支持的 CUDA 版本 | 选择 torch 的 CUDA 构建（`env/README.md`） |
| `jobs.train.time`、`jobs.baselines.time`、`jobs.ablation.time` | 作业时长 | 实测后填；未填时提交必须带 `--time` |
