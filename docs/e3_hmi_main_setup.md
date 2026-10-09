# E3：第二台靶机 hmi_main 搭建说明

> 资产映射与漏洞数据来自实验环境，仅用于演示。本靶机只接入 host-only 实验网 192.168.56.0/24。

## 目的

现在只有 ews_01（Metasploitable2）一台靶机。它的 R_asset = 1.0，于是 P1/P2 的 E_sub 都是 1、P3 是 0，
E_sub 只有 0/1 两个取值（风险饱和，见 PROGRESS.md）。
hmi_main 是 P1/P2/P3 共同的上游资产，给它一台**补丁较新、服务较少**的真实靶机后：
- P3 的 E_sub 从 0 变成 R_asset(hmi_main)，预期是较低的非零值；
- P1/P2 仍由 ews_01 决定（1.0），报告中可以如实讨论这一点。

## 规格

| 项目 | 值 |
|---|---|
| 角色 | hmi_main（HMI，Purdue level 2） |
| 系统 | Ubuntu Server 22.04 LTS（与扫描机同版本即可），装完后全部更新 |
| 资源 | 1 vCPU，1 GB 内存，10 GB 磁盘 |
| 网卡 | **只保留一块**，接到与 Metasploitable2 相同的 host-only 网络 |
| IP | 静态 192.168.56.20/24，**不设网关、不设 DNS** |
| 对外服务 | 22/tcp（SSH）、80/tcp 与 443/tcp（nginx，模拟 Web HMI）；其余端口不开放 |

## 步骤

### 1. 安装与更新（这一步需要临时联网）

1. 在虚拟化软件（和 Metasploitable2 同一个）中新建虚拟机，先给**一块 NAT 网卡**，用于下载更新。
2. 安装 Ubuntu Server 22.04，安装过程中勾选 OpenSSH server，其余 snap 都不选。
3. 登录后执行：
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo apt install -y nginx
   sudo reboot
   ```

### 2. 配置 Web HMI 页面与 HTTPS

```bash
# 静态"HMI"首页（只是占位页面，没有任何控制功能）
echo '<html><body><h1>HMI Main (lab)</h1><p>实验环境演示页面</p></body></html>' | sudo tee /var/www/html/index.html

# 自签名证书（有效期 1 年）
sudo openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
  -keyout /etc/ssl/private/hmi.key -out /etc/ssl/certs/hmi.crt -subj "/CN=hmi-main.lab"

# 在默认站点中加入 443 监听
sudo tee /etc/nginx/sites-available/default >/dev/null <<'EOF'
server {
    listen 80 default_server;
    listen 443 ssl default_server;
    ssl_certificate     /etc/ssl/certs/hmi.crt;
    ssl_certificate_key /etc/ssl/private/hmi.key;
    root /var/www/html;
    index index.html;
    location / { try_files $uri $uri/ =404; }
}
EOF
sudo nginx -t && sudo systemctl restart nginx
```

不要额外降低 TLS/SSH 配置，保持发行版默认值：这台机器代表"打过补丁的资产"，我们要看的是真实的扫描结果。

### 3. 切换到 host-only 网络

1. 关机。在虚拟机设置中**删除 NAT 网卡**，添加一块 host-only 网卡（与 Metasploitable2 相同的网络，例如 VMware 的 VMnet1）。
2. 开机后配置静态 IP（先用 `ip -br link` 确认网卡名，下面以 `ens33` 为例）：
   ```bash
   sudo tee /etc/netplan/50-lab.yaml >/dev/null <<'EOF'
   network:
     version: 2
     ethernets:
       ens33:
         dhcp4: false
         addresses: [192.168.56.20/24]
   EOF
   sudo chmod 600 /etc/netplan/50-lab.yaml
   sudo rm -f /etc/netplan/00-installer-config.yaml   # 删掉安装时生成的 DHCP 配置
   sudo netplan apply
   ip -br addr
   ```
3. 在靶机上确认监听端口只有 22、80、443：`sudo ss -tlnp`。

### 4. 在扫描机（192.168.56.10）上确认可达

```bash
ping -c 2 192.168.56.20
curl -s  http://192.168.56.20/ | head -1
curl -sk https://192.168.56.20/ | head -1
```

这几条通过后，给虚拟机拍一个快照（例如 `e3-baseline`），保证以后能复现扫描结果。

## 搭好之后

告诉我"hmi_main 已就绪"。我会先把 `config/asset_map.yaml` 中 hmi_main 改为
`ip: 192.168.56.20, scanned: true`（扫描脚本要求目标必须是 scanned: true 且位于实验网段内），然后你运行：

```bash
cd ~/桌面/openvas
# 1) 预检：只连接和校验，不创建任何对象
.venv/bin/python -m scripts.smoke_scan --asset hmi_main --output data/samples/hmi_main_report.xml
# 2) 确认预检输出里的 IP 是 192.168.56.20，再真正扫描（约 20~60 分钟）
.venv/bin/python -m scripts.smoke_scan --asset hmi_main --output data/samples/hmi_main_report.xml --execute
```

扫描完成后，脚本会校验导出条数（输出里应出现 `filtered == full`）。之后我负责：
parse（合并两份报告）→ intel → 重跑融合 → 用 `scripts/compare_fusion.py` 对比加靶机前后的 R_asset、E_sub 和排序。
