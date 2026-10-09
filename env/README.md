# 集群环境搭建（Minerva，Python 3.12）

代码兼容 Python 3.10–3.12：本地开发用 3.10，集群用系统自带的 3.12。

## 文件

- `requirements-cluster.txt`：锁定版本。由本地 `.venv` 的 `pip freeze` 生成，torch、scikit-learn、numpy、pandas 等
  版本与本地完全一致；唯一区别是 torch 在本地为 CPU 构建（`2.14.1+cpu`），在集群为同版本的 CUDA 构建。

## 安装（在登录节点上，创建 venv 和 pip 安装是允许的）

```bash
cd <代码目录>
python3 --version                      # 应为 3.12.x
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --no-cache-dir -r env/requirements-cluster.txt
```

### torch 的 CUDA 构建

- 默认（上面的命令）：PyPI 上的 `torch==2.14.1` Linux wheel 自带 CUDA 运行库（依赖 `nvidia-*` 和 `cuda-toolkit` 包，
  当前解析结果为 CUDA 13.0 系列），不需要集群另外加载 CUDA 模块，但要求显卡驱动足够新。
- 先查驱动支持的 CUDA 版本（runbook 第 1 步）：`srun --gres=gpu:L4:1 --time=00:02:00 nvidia-smi`，
  输出右上角的 `CUDA Version` 是驱动支持的最高版本。
- 如果驱动支持的版本低于默认 wheel 需要的版本，改从 PyTorch 官方 index 安装同一个 torch 版本的旧 CUDA 构建：

  ```bash
  # cuXXX 按驱动支持的版本在 https://pytorch.org/get-started/locally/ 上选择
  .venv/bin/pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cuXXX
  .venv/bin/pip install --no-cache-dir -r env/requirements-cluster.txt   # 其余包；torch 版本号相同，不会被替换
  ```

- 验证（必须通过 srun，不在登录节点上跑）：`sbatch cluster/pytest.sbatch`，日志末尾应显示 `cuda True` 和 GPU 型号。

## 已做的验证（本地，2026-10-07）

- 用 Python 3.12.14 新建虚拟环境，按 `requirements-cluster.txt` 的锁定版本安装（torch 用同版本的 CPU 构建），
  全部包都有 3.12 的 wheel。
- 在只含 git 跟踪文件的干净副本上（模拟 `push_to_cluster.sh` 推送的内容），用这个 3.12 环境跑完整 pytest：
  135 通过、3 跳过（跳过的是"本地执行作业脚本"的冒烟测试，因为那个副本里没有 `.venv`；在集群的代码目录里会运行）。
- PyPI 上 `torch==2.14.1` 的 CUDA 构建在 Python 3.12 上可解析（只做了依赖解析，没有下载）。
- 未验证：CUDA 构建在 Minerva 的驱动上能否运行。这要等登录后用 `cluster/pytest.sbatch` 确认。

## 更新锁定文件

本地依赖有变化时重新生成，并在 3.12 环境里重跑上面的验证：

```bash
{ sed -n '/^#/p' env/requirements-cluster.txt
  .venv/bin/pip freeze | sed -e 's/^torch==\(.*\)+cpu$/torch==\1/' \
      -e 's/^\(exceptiongroup\|tomli\)==\(.*\)$/\1==\2 ; python_version < "3.11"/'; } > /tmp/req && mv /tmp/req env/requirements-cluster.txt
```
