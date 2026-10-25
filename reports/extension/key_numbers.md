# Extension key numbers

> Disclaimer: Asset mapping and vulnerability data come from a lab environment and are for demonstration only.

## Vulnerability collection (OpenVAS)

- 1 reports, hosts 192.168.56.11
- 630 results, 90 Log dropped, 0 duplicates dropped, 540 written
- Critical (CVSS ≥ 9.0): 101

## Threat intelligence

- KEV version 2026.09.29, EPSS date 2026-09-29
- 1086 unique CVEs, 5 without an EPSS record, 47 vulnerabilities without CVEs
- 9 in KEV (8 CVEs)
- EPSS distribution (per vulnerability, max over its CVEs): > 0.0: 493, > 0.01: 448, > 0.1: 208, > 0.5: 89, > 0.9: 33

## QoD filtering

- min_qod = 70: kept 66 / 540 (88% dropped), of which 2 in KEV; 7 need manual verification

| QoD | Count | Kept |
|---|---|---|
| 30 | 472 | no |
| 50 | 2 | no |
| 70 | 8 | yes |
| 80 | 31 | yes |
| 95 | 4 | yes |
| 98 | 5 | yes |
| 99 | 18 | yes |

## Asset risk and subsystem exposure risk

Parameters α=1.0, β=1.0, γ=1.5

| Asset | R_asset | Status | Vulns used | Needs review |
|---|---|---|---|---|
| ews_01 | 1.000 | scanned | 66 | 7 |
| ews_02 | Not assessed | not_scanned | 0 | 0 |
| hmi_main | Not assessed | not_scanned | 0 | 0 |
| historian | Not assessed | not_scanned | 0 | 0 |
| plc_p1 | Not assessed | not_scanned | 0 | 0 |
| plc_p2 | Not assessed | not_scanned | 0 | 0 |
| plc_p3 | Not assessed | not_scanned | 0 | 0 |

| Subsystem | E_sub | Source asset | Unassessed upstream assets |
|---|---|---|---|
| P1 | 1.000 | ews_01 | historian, hmi_main, plc_p1 |
| P2 | 1.000 | ews_01 | historian, hmi_main, plc_p2 |
| P3 | not assessed (counted as 0) | - | ews_02, historian, hmi_main, plc_p3 |
| P4 | 0 (no associated assets) | - | - |

## Risk saturation

- Of the 66 vulnerabilities used, 18 have r_v clipped to 1.0; R_asset and E_sub are therefore capped, and with only one target assessed, E_sub can only be 0 or 1.
- This is caused jointly by the min(1, ·) formula and the single target, and is written into the report as a limitation (the second target of E3 was skipped).
