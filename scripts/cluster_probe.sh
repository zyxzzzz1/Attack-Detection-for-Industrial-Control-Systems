#!/usr/bin/env bash
# 查询集群信息（runbook 第 1 步），用于补全 config/cluster.yaml。
# 在本地执行：ssh minerva 'bash -l -s' < scripts/cluster_probe.sh > artifacts/cluster_probe.txt 2>&1
# 登录节点上只做轻量查询；最后两条通过 srun 在计算节点上执行，各约几秒。
set -u

echo "== 用户与目录 =="
whoami
echo "HOME=$HOME"
echo "TMPDIR=${TMPDIR:-}"

echo "== 分区 =="
sinfo -o "%P %l %D %c %m %G"
scontrol show partition | grep -E "PartitionName|MaxTime|DefaultTime"

echo "== 账户 =="
sacctmgr -nP show assoc user="$USER" format=account,partition,qos

echo "== python =="
python3 --version

echo "== 配额 =="
quota -s 2>/dev/null
df -h "$HOME"

echo "== scratch =="
ls -d /scratch* /local* /data* 2>/dev/null

echo "== 登录节点能否上网 =="
curl -sI --max-time 10 https://github.com | head -1
[[ "${PIPESTATUS[0]}" == "0" ]] || echo "登录节点不能上网"

# srun 加 </dev/null：否则它会把本脚本剩余部分当作标准输入读走
echo "== 计算节点能否上网 =="
srun --time=00:02:00 bash -c 'hostname; curl -sI --max-time 10 https://pypi.org | head -1; [[ "${PIPESTATUS[0]}" == "0" ]] || echo "计算节点不能上网"' </dev/null

echo "== GPU =="
srun --gres=gpu:L4:1 --time=00:02:00 nvidia-smi </dev/null

echo "== 结束 =="
