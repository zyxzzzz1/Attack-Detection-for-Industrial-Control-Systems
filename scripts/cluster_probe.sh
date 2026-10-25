#!/usr/bin/env bash
# Query cluster information (runbook step 1) to fill in config/cluster.yaml.
# Run locally: ssh minerva 'bash -l -s' < scripts/cluster_probe.sh > artifacts/cluster_probe.txt 2>&1
# Only lightweight queries on the login node; the last two run on a compute node via srun, a few seconds each.
set -u

echo "== User and directories =="
whoami
echo "HOME=$HOME"
echo "TMPDIR=${TMPDIR:-}"

echo "== Partitions =="
sinfo -o "%P %l %D %c %m %G"
scontrol show partition | grep -E "PartitionName|MaxTime|DefaultTime"

echo "== Accounts =="
sacctmgr -nP show assoc user="$USER" format=account,partition,qos

echo "== python =="
python3 --version

echo "== Quota =="
quota -s 2>/dev/null
df -h "$HOME"

echo "== scratch =="
ls -d /scratch* /local* /data* 2>/dev/null

echo "== Internet access from the login node =="
curl -sI --max-time 10 https://github.com | head -1
[[ "${PIPESTATUS[0]}" == "0" ]] || echo "login node has no internet access"

# srun gets </dev/null: otherwise it would consume the rest of this script as its stdin
echo "== Internet access from compute nodes =="
srun --time=00:02:00 bash -c 'hostname; curl -sI --max-time 10 https://pypi.org | head -1; [[ "${PIPESTATUS[0]}" == "0" ]] || echo "compute node has no internet access"' </dev/null

echo "== GPU =="
srun --gres=gpu:L4:1 --time=00:02:00 nvidia-smi </dev/null

echo "== Done =="
