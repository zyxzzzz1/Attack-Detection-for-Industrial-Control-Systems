# Alert LSTMAE-01505 — priority rank 28 / 1703

> Disclaimer: Asset mapping and vulnerability data come from a lab environment and are for demonstration only.

## Summary

| Item | Value |
|---|---|
| Priority | **1.598** (rank 28) |
| Start ~ end | 2020-07-31 08:41:25 ~ 2020-07-31 08:42:04 |
| Duration | 40 s |
| Max anomaly score / threshold | 0.5164 / 0.2369 (2.18x) |
| Anomaly severity S | 1.000 |
| Weighted exposure risk E_sub | 0.788 (possibly underestimated) |

## Top-5 anomalous sensors

| Sensor | Share of total sensor error | Subsystem |
|---|---|---|
| `P1_PIT01` | 14.0% | P1 |
| `P1_FT01Z` | 12.5% | P1 |
| `P4_ST_PT01` | 10.6% | P4 |
| `P2_HILout` | 6.5% | P2 |
| `P2_VT01` | 6.4% | P2 |

## Inferred affected subsystems

- **P1 Boiler process**: weight 52.9%, business criticality 0.9, exposure risk E_sub = 1.000 (3 more upstream assets not assessed)
- **P2 Turbine process**: weight 25.9%, business criticality 1.0, exposure risk E_sub = 1.000 (3 more upstream assets not assessed)
- **P4 HIL simulation**: weight 21.2%, business criticality 0.6, exposure risk E_sub = 0.000 (no control chain: HIL simulation)
  - Note: P4 is the HIL simulation subsystem and has no control chain of its own. It is physically coupled with P1, P2 and P3 (the pump and valve of P3 are controlled by the pumped-storage hydropower generation model in the HIL); its anomalies may be caused by attacks on P1/P2/P3 propagating through physical coupling, so the assets related to P1/P2/P3 should be investigated as well.

## Possible intrusion paths

> **Note: This path contains assets that have not been assessed; the actual risk may be underestimated** (not assessed: historian, hmi_main, plc_p1, plc_p2)

- P1: ews_01 → plc_p1
- P1: historian → hmi_main → plc_p1
- P2: ews_01 → plc_p2
- P2: historian → hmi_main → plc_p2

Assets on the paths:

| Asset | Risk R_asset | Status | Vulnerabilities counted | KEV vulns needing review |
|---|---|---|---|---|
| ews_01 | 1.000 | Scanned | 66 | 7 |
| historian | Not assessed | Not scanned (not assessed) | 0 | 0 |
| hmi_main | Not assessed | Not scanned (not assessed) | 0 | 0 |
| plc_p1 | Not assessed | Not scanned (not assessed) | 0 | 0 |
| plc_p2 | Not assessed | Not scanned (not assessed) | 0 | 0 |

## High-risk vulnerabilities on the path assets (sorted by r_v, QoD ≥ 70)

| # | Asset | Port | Vulnerability | CVSS | EPSS | KEV | QoD | r_v |
|---|---|---|---|---|---|---|---|---|
| 1 | ews_01 | 8009/tcp | Apache Tomcat AJP RCE Vulnerability (Ghostcat) - Active Check | 9.8 | 0.993 | **KEV** CVE-2020-1938 | 99 | 1.000 |
| 2 | ews_01 | 80/tcp | PHP < 5.3.13, 5.4.x < 5.4.3 Multiple Vulnerabilities - Active Check | 7.5 | 1.000 | **KEV** CVE-2012-1823 | 95 | 1.000 |
| 3 | ews_01 | 80/tcp | TWiki < 4.2.4 Multiple XSS / Command Execution Vulnerabilities | 10.0 | 0.046 | - | 80 | 1.000 |
| 4 | ews_01 | 512/tcp | The rexec service is running | 10.0 | 0.019 | - | 80 | 1.000 |
| 5 | ews_01 | 1524/tcp | Possible Backdoor: Ingreslock | 10.0 | 0.000 | - | 99 | 1.000 |
| 6 | ews_01 | 513/tcp | rlogin Passwordless Login | 10.0 | 0.000 | - | 80 | 1.000 |
| 7 | ews_01 | 8787/tcp | Distributed Ruby (dRuby/DRb) Multiple RCE Vulnerabilities | 10.0 | 0.000 | - | 99 | 1.000 |
| 8 | ews_01 | general/tcp | Operating System (OS) End of Life (EOL) Detection | 10.0 | 0.000 | - | 80 | 1.000 |
| 9 | ews_01 | 21/tcp | vsftpd Compromised Source Packages Backdoor Vulnerability - Active Check | 9.8 | 0.962 | - | 99 | 1.000 |
| 10 | ews_01 | 6200/tcp | vsftpd Compromised Source Packages Backdoor Vulnerability - Active Check | 9.8 | 0.962 | - | 99 | 1.000 |

## Needs manual verification (in KEV but QoD < 70; not included in the risk calculation)

| Asset | Port | Vulnerability | KEV CVE | EPSS | QoD |
|---|---|---|---|---|---|
| ews_01 | 80/tcp | Apache HTTP Server < 2.4.49 Multiple Vulnerabilities - Linux | CVE-2021-40438 | 1.000 | 30 |
| ews_01 | 80/tcp | PHP < 8.1.29, 8.2.x < 8.2.20, 8.3.x < 8.3.8 Multiple Vulnerabilities - Linux | CVE-2024-4577 | 1.000 | 30 |
| ews_01 | 80/tcp | Apache HTTP Server < 2.4.60 Multiple Vulnerabilities - Linux | CVE-2024-38475 | 1.000 | 30 |
| ews_01 | 80/tcp | PHP < 7.1.33, 7.2.x < 7.2.24, 7.3.x < 7.3.11 RCE Vulnerability - Version Check | CVE-2019-11043 | 0.998 | 30 |
| ews_01 | 80/tcp | phpMyAdmin Code Injection and XSS Vulnerability | CVE-2009-1151 | 0.966 | 30 |
| ews_01 | 80/tcp | phpMyAdmin 2.11.x < 2.11.9.4 / 3.0.x < 3.1.3 Multiple Vulnerabilities | CVE-2009-1151 | 0.966 | 30 |
| ews_01 | 80/tcp | jQuery 1.0.3 < 3.5.0 XSS Vulnerability | CVE-2020-11023 | 0.849 | 30 |

## Recommended actions

1. Investigate these assets first: ews_01 (R_asset 1.000)
2. Manually verify whether the 7 vulnerabilities that are in KEV but have low QoD really exist
3. Assets on the path that have not been assessed (not scanned): historian, hmi_main, plc_p1, plc_p2 (PLCs are not actively scanned; check them using logs and configuration)
4. Follow the subsystem note and investigate the physically coupled upstream processes

OpenVAS remediation advice:

- **Apache Tomcat AJP RCE Vulnerability (Ghostcat) - Active Check** (ews_01 8009/tcp): - Update Apache Tomcat to version 7.0.100, 8.5.51, 9.0.31 or later - For other products using Tomcat please contact the vendor for more information on fixed versions
- **PHP < 5.3.13, 5.4.x < 5.4.3 Multiple Vulnerabilities - Active Check** (ews_01 80/tcp): PHP: Update to version 5.3.13, 5.4.3 or later - Other products / applications: Please contact the vendor for a solution
- **TWiki < 4.2.4 Multiple XSS / Command Execution Vulnerabilities** (ews_01 80/tcp): Update to version 4.2.4 or later.
- **The rexec service is running** (ews_01 512/tcp): Disable the rexec service and use alternatives like SSH instead.
- **Possible Backdoor: Ingreslock** (ews_01 1524/tcp): A whole cleanup of the infected system is recommended.

---
Asset mapping and vulnerability data come from a lab environment and are for demonstration only.
