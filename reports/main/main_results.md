# Main results

All numbers come from full runs on the cluster. Threshold = 99% quantile of the val_thr scores; point-wise metrics **do not use point-adjust**, so a low point-wise F1 is expected.

The first W−1 seconds of each test file have no score and are excluded from every metric; their number is in the n_unscored column. Detection delay counts only detected events; false-alarm rate = alarm segments overlapping no attack event ÷ normal duration (hours, counting only scored seconds without attack).

Metric conventions: alarm rule is score > threshold; with no alarms precision is shown as — (NaN), and if there are attacks within the evaluation range recall and F1 are 0; AUROC depends only on the scores, not on the threshold, and is undefined only when all labels within the evaluation range are of one class.

| model | precision | recall | f1 | auroc | event_recall | delay_median_s | delay_mean_s | false_alarms_per_hour | threshold | n_unscored |
|---|---|---|---|---|---|---|---|---|---|---|
| pca | 0.0946 | 0.8405 | 0.1700 | 0.9014 | 0.9600 | 6.0 | 18.2 | 7.1585 | 0.098545 | 295 |
| iforest | 0.4144 | 0.4828 | 0.4460 | 0.8347 | 0.7000 | 28.0 | 43.5 | 6.8377 | 0.524486 | 295 |
| lstm_ae | 0.2765 | 0.8040 | 0.4290 ± 0.0184 | 0.9335 ± 0.0016 | 0.9920 ± 0.0110 | 6.0 | 15.8 | 13.1878 ± 1.6764 | 0.236868 | 295 |
| lstm_vae | 0.0606 | 0.8111 | 0.1128 | 0.8800 | 0.9600 | 4.0 | 16.8 | 21.2189 | 0.279032 | 295 |
| deep_svdd | 0.1683 | 0.8287 | 0.2797 | 0.9055 | 0.9800 | 3.0 | 13.9 | 69.3304 | 0.000078 | 295 |

## Random seeds

In the lstm_ae row, f1, auroc, event_recall, false_alarms_per_hour are the mean ± sample std (ddof=1) over 5 random seeds (42, 43, 44, 45, 46); the other columns of that row come from the main-model run with seed 42 (that run's f1 0.4115, auroc 0.9339, event_recall 1.0000, false_alarms_per_hour 14.9495); alert export, attribution and figures also use only that run. The seed runs have identical configurations apart from the seed, and each threshold is the 99% quantile of its own val_thr. **All other models have results for a single seed only**; their random variation was not assessed.

| seed | run_name | precision | recall | f1 | auroc | event_recall | delay_median_s | delay_mean_s | false_alarms_per_hour | threshold | n_unscored |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | lstm_ae | 0.2765 | 0.8040 | 0.4115 | 0.9339 | 1.0000 | 6.0 | 15.8 | 14.9495 | 0.236868 | 295 |
| 43 | seeds/lstm_ae_s43 | 0.3047 | 0.7997 | 0.4413 | 0.9331 | 1.0000 | 7.0 | 17.3 | 10.6324 | 0.248462 | 295 |
| 44 | seeds/lstm_ae_s44 | 0.2847 | 0.7964 | 0.4194 | 0.9311 | 1.0000 | 5.0 | 16.5 | 14.1795 | 0.251917 | 295 |
| 45 | seeds/lstm_ae_s45 | 0.3162 | 0.8111 | 0.4550 | 0.9354 | 0.9800 | 4.0 | 13.7 | 13.6388 | 0.242055 | 295 |
| 46 | seeds/lstm_ae_s46 | 0.2824 | 0.8018 | 0.4177 | 0.9341 | 0.9800 | 5.0 | 15.8 | 12.5389 | 0.247456 | 295 |
| mean |  | 0.2929 | 0.8026 | 0.4290 | 0.9335 | 0.9920 | 5.4 | 15.8 | 13.1878 | 0.245352 | 295 |
| std |  | 0.0168 | 0.0055 | 0.0184 | 0.0016 | 0.0110 | 1.1 | 1.3 | 1.6764 | 0.005917 | 0 |
