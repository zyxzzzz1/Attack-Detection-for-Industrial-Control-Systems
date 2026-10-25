# Vulnerabilities in CISA KEV

> Disclaimer: Asset mapping and vulnerability data come from a lab environment and are for demonstration only.

KEV version 2026.09.29; 9 entries. Those with QoD < 70 are excluded from the risk computation and listed under "needs manual verification".

| KEV CVE | Vulnerability | Port | CVSS | EPSS | QoD | solution_type | r_v | Used in risk computation |
|---|---|---|---|---|---|---|---|---|
| CVE-2020-1938 | Apache Tomcat AJP RCE Vulnerability (Ghostcat) - Active Check | 8009/tcp | 9.8 | 0.993 | 99 | VendorFix | 1.000 | yes |
| CVE-2012-1823 | PHP < 5.3.13, 5.4.x < 5.4.3 Multiple Vulnerabilities - Active Check | 80/tcp | 7.5 | 1.000 | 95 | VendorFix | 1.000 | yes |
| CVE-2021-40438 | Apache HTTP Server < 2.4.49 Multiple Vulnerabilities - Linux | 80/tcp | 9.8 | 1.000 | 30 | VendorFix | 1.000 | no (needs manual verification) |
| CVE-2024-38475 | Apache HTTP Server < 2.4.60 Multiple Vulnerabilities - Linux | 80/tcp | 9.8 | 1.000 | 30 | VendorFix | 1.000 | no (needs manual verification) |
| CVE-2019-11043 | PHP < 7.1.33, 7.2.x < 7.2.24, 7.3.x < 7.3.11 RCE Vulnerability - Version Check | 80/tcp | 9.8 | 0.998 | 30 | VendorFix | 1.000 | no (needs manual verification) |
| CVE-2024-4577 | PHP < 8.1.29, 8.2.x < 8.2.20, 8.3.x < 8.3.8 Multiple Vulnerabilities - Linux | 80/tcp | 9.8 | 1.000 | 30 | VendorFix | 1.000 | no (needs manual verification) |
| CVE-2009-1151 | phpMyAdmin 2.11.x < 2.11.9.4 / 3.0.x < 3.1.3 Multiple Vulnerabilities | 80/tcp | 9.8 | 0.966 | 30 | VendorFix | 1.000 | no (needs manual verification) |
| CVE-2009-1151 | phpMyAdmin Code Injection and XSS Vulnerability | 80/tcp | 9.8 | 0.966 | 30 | VendorFix | 1.000 | no (needs manual verification) |
| CVE-2020-11023 | jQuery 1.0.3 < 3.5.0 XSS Vulnerability | 80/tcp | 6.1 | 0.849 | 30 | VendorFix | 1.000 | no (needs manual verification) |
