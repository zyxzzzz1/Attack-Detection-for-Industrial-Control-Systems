# Fusion ranking evaluation

72 of the 1703 alerts of lstm_ae are true alarms (base rate 0.0423, the expected precision@k of a random ranking). precision@k is the fraction of the top k alerts that are true alarms; ci_low / ci_high are 95% Wilson confidence intervals.

- raw_score: ranked by the maximum anomaly score within the alarm segment.
- alpha=0: fusion priority without the vulnerability exposure risk (only anomaly severity S and subsystem criticality).
- alpha=1: fusion priority with the vulnerability exposure risk E_sub (default parameters).

The anomaly severity S = min(1, max score ÷ threshold − 1) is capped once the score reaches twice the threshold: 61 alerts have S = 1. In the fusion ranking these alerts are distinguished only by subsystem weight, criticality and E_sub; n_tied_with_kth is the number of alerts with exactly the same priority as the k-th, and ties are broken by alert ID (time order). n_shared_with_raw_score is the number of alerts in the top k that are also in the top k of the raw-score ranking (equal precision does not mean the same alerts were selected).

**Limitation (constraint 7)**: the asset mapping is constructed by hand and the vulnerability data come from lab targets, with no causal link to the intrusions in HAI. This table only describes how this ranking formula behaves on this data; it does not show that vulnerability context improves detection, and the fusion part claims no improvement of detection metrics.

| ranking | k | n_true | precision | ci_low | ci_high | n_alerts | n_true_total | base_rate | n_tied_with_kth | n_shared_with_raw_score |
|---|---|---|---|---|---|---|---|---|---|---|
| raw_score | 10 | 10 | 1.0000 | 0.7225 | 1.0000 | 1703 | 72 | 0.0423 | 1 | 10 |
| raw_score | 20 | 20 | 1.0000 | 0.8389 | 1.0000 | 1703 | 72 | 0.0423 | 1 | 20 |
| raw_score | 50 | 45 | 0.9000 | 0.7864 | 0.9565 | 1703 | 72 | 0.0423 | 1 | 50 |
| alpha=0 | 10 | 10 | 1.0000 | 0.7225 | 1.0000 | 1703 | 72 | 0.0423 | 1 | 9 |
| alpha=0 | 20 | 20 | 1.0000 | 0.8389 | 1.0000 | 1703 | 72 | 0.0423 | 1 | 15 |
| alpha=0 | 50 | 45 | 0.9000 | 0.7864 | 0.9565 | 1703 | 72 | 0.0423 | 1 | 46 |
| alpha=1 | 10 | 10 | 1.0000 | 0.7225 | 1.0000 | 1703 | 72 | 0.0423 | 1 | 9 |
| alpha=1 | 20 | 20 | 1.0000 | 0.8389 | 1.0000 | 1703 | 72 | 0.0423 | 1 | 15 |
| alpha=1 | 50 | 45 | 0.9000 | 0.7864 | 0.9565 | 1703 | 72 | 0.0423 | 1 | 44 |
