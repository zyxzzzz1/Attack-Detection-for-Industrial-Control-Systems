# Attribution evaluation summary

For each alarm segment, take the share of each sensor's mean reconstruction error within the segment; a hit if the Top-1 sensor's subsystem is in the event's subsystem set. When an alarm segment overlaps several events (n_multi_event_segments of them), the primary definition uses the event with the longest overlap (n_evaluable, n_hit, subsystem_hit_rate); the union definition (Top-1 subsystem in the union of those events' subsystem sets, n_evaluable_union, n_hit_union, subsystem_hit_rate_union) is a supplement and can only have at least as many hits as the primary one. Only alarm segments overlapping an attack event whose event has subsystem labels count toward the hit rate; alarm segments overlapping no attack (n_non_overlap) and those overlapping only events without subsystem labels are listed separately. top1_p4_ratio is the fraction of attack-overlapping alarm segments whose Top-1 is in P4 (P4 has no subsystem label, so Top-1 in P4 is always a miss).

There are two per-event hit-rate definitions, both with the detected events that have subsystem labels as the denominator:

- **Primary, event_first_hit_rate**: take each event's first overlapping alarm segment (by segment start time) and check whether its Top-1 subsystem is in the event's subsystem set. This is the alert an operator sees first; the segment's error is computed over the whole segment (the alarm may start before the event does).
- **Supplementary, event_hit_rate**: merge the errors of all alarm seconds within the event interval and take the Top-1. It uses every alarm up to the end of the event, which is not available when the alarm first appears, so it is only a supplement.

ci_low and ci_high after each hit rate are 95% Wilson confidence intervals. The denominators are small and the intervals wide, so differences between models must be read together with the intervals.

Sensor-level Hit@k was not done: the HAI 21.03 data files have no per-sensor attack-target annotation.

| model | n_alarm_segments | n_overlap_attack | n_evaluable | n_hit | subsystem_hit_rate | subsystem_hit_rate_ci_low | subsystem_hit_rate_ci_high | n_multi_event_segments | n_evaluable_union | n_hit_union | subsystem_hit_rate_union | n_top1_p4 | top1_p4_ratio | n_overlap_no_subsystem_label | n_non_overlap | n_events | n_events_without_subsystem | n_events_attributed | n_events_first_evaluable | n_events_first_hit | event_first_hit_rate | event_first_hit_rate_ci_low | event_first_hit_rate_ci_high | n_events_merged_hit | event_hit_rate | event_hit_rate_ci_low | event_hit_rate_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_ae | 1703 | 72 | 72 | 54 | 0.7500 | 0.6391 | 0.8356 | 0 | 72 | 54 | 0.7500 | 6 | 0.0833 | 0 | 1631 | 50 | 0 | 50 | 50 | 43 | 0.8600 | 0.7381 | 0.9305 | 45 | 0.9000 | 0.7864 | 0.9565 |
| lstm_vae | 2372 | 57 | 57 | 52 | 0.9123 | 0.8106 | 0.9619 | 0 | 57 | 52 | 0.9123 | 4 | 0.0702 | 0 | 2315 | 50 | 0 | 48 | 48 | 47 | 0.9792 | 0.8910 | 0.9963 | 47 | 0.9792 | 0.8910 | 0.9963 |
| pca | 834 | 53 | 53 | 37 | 0.6981 | 0.5646 | 0.8048 | 0 | 53 | 37 | 0.6981 | 0 | 0.0000 | 0 | 781 | 50 | 0 | 48 | 48 | 36 | 0.7500 | 0.6122 | 0.8508 | 34 | 0.7083 | 0.5682 | 0.8176 |

## By alarm-segment duration

Only attack-overlapping, evaluable alarm segments. variant = main is the main result; valnorm is the exploratory analysis below.

| model | variant | duration_group | n | n_hit | rate | ci_low | ci_high |
|---|---|---|---|---|---|---|---|
| lstm_ae | main | <10 s | 20 | 7 | 0.3500 | 0.1812 | 0.5671 |
| lstm_ae | main | 10–60 s | 4 | 1 | 0.2500 | 0.0456 | 0.6994 |
| lstm_ae | main | >60 s | 48 | 46 | 0.9583 | 0.8602 | 0.9885 |
| lstm_ae | valnorm | <10 s | 20 | 13 | 0.6500 | 0.4329 | 0.8188 |
| lstm_ae | valnorm | 10–60 s | 4 | 4 | 1.0000 | 0.5101 | 1.0000 |
| lstm_ae | valnorm | >60 s | 48 | 44 | 0.9167 | 0.8045 | 0.9671 |
| lstm_vae | main | <10 s | 4 | 2 | 0.5000 | 0.1500 | 0.8500 |
| lstm_vae | main | 10–60 s | 3 | 0 | 0.0000 | 0.0000 | 0.5615 |
| lstm_vae | main | >60 s | 50 | 50 | 1.0000 | 0.9287 | 1.0000 |
| lstm_vae | valnorm | <10 s | 4 | 4 | 1.0000 | 0.5101 | 1.0000 |
| lstm_vae | valnorm | 10–60 s | 3 | 2 | 0.6667 | 0.2077 | 0.9385 |
| lstm_vae | valnorm | >60 s | 50 | 46 | 0.9200 | 0.8116 | 0.9685 |
| pca | main | <10 s | 5 | 2 | 0.4000 | 0.1176 | 0.7693 |
| pca | main | 10–60 s | 2 | 1 | 0.5000 | 0.0945 | 0.9055 |
| pca | main | >60 s | 46 | 34 | 0.7391 | 0.5974 | 0.8440 |
| pca | valnorm | <10 s | 5 | 2 | 0.4000 | 0.1176 | 0.7693 |
| pca | valnorm | 10–60 s | 2 | 2 | 1.0000 | 0.3424 | 1.0000 |
| pca | valnorm | >60 s | 46 | 40 | 0.8696 | 0.7433 | 0.9388 |

## Multiple random seeds

Attribution computed once for each seed run in ml.seed_study (identical configuration apart from the seed); the last two rows are the mean and sample std (ddof=1). The main result is still the main-model run (the seed in the first row).

| seed | run_name | n_alarm_segments | n_evaluable | n_hit | subsystem_hit_rate | n_events_first_evaluable | n_events_first_hit | event_first_hit_rate | n_events_merged_hit | event_hit_rate | n_top1_p4 | top1_p4_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | lstm_ae | 1703.0000 | 72.0000 | 54.0000 | 0.7500 | 50.0000 | 43.0000 | 0.8600 | 45.0000 | 0.9000 | 6.0000 | 0.0833 |
| 43 | seeds/lstm_ae_s43 | 1223.0000 | 63.0000 | 51.0000 | 0.8095 | 50.0000 | 44.0000 | 0.8800 | 46.0000 | 0.9200 | 7.0000 | 0.1111 |
| 44 | seeds/lstm_ae_s44 | 1611.0000 | 64.0000 | 48.0000 | 0.7500 | 50.0000 | 43.0000 | 0.8600 | 46.0000 | 0.9200 | 9.0000 | 0.1406 |
| 45 | seeds/lstm_ae_s45 | 1555.0000 | 67.0000 | 51.0000 | 0.7612 | 49.0000 | 42.0000 | 0.8571 | 45.0000 | 0.9184 | 10.0000 | 0.1493 |
| 46 | seeds/lstm_ae_s46 | 1428.0000 | 60.0000 | 49.0000 | 0.8167 | 49.0000 | 45.0000 | 0.9184 | 46.0000 | 0.9388 | 5.0000 | 0.0833 |
| mean |  | 1504.0000 | 65.2000 | 50.6000 | 0.7775 | 49.6000 | 43.4000 | 0.8751 | 45.6000 | 0.9194 | 7.4000 | 0.1135 |
| std |  | 186.0027 | 4.5497 | 2.3022 | 0.0329 | 0.5477 | 1.1402 | 0.0259 | 0.5477 | 0.0137 | 2.0736 | 0.0310 |

## Exploratory analysis: normalized by the val_thr mean error (post hoc, does not replace the main results)

**This analysis was done after seeing the main results.** Method: the mean error of each sensor within the alarm segment is first divided by that sensor's mean error on val_thr, then normalized and ranked for Top-k, i.e. what is compared is the "multiple of the usual error". The baseline comes only from val_thr and uses no test labels; but the decision to do this step was made after seeing the hit rates on the test set, so its hit rates are optimistic and must not be cited as the result of a method fixed in advance. The main results, alerts.json and the alert cards do not use this normalization.

| model | level | n | n_hit_main | rate_main | n_hit_valnorm | rate_valnorm | rate_valnorm_ci_low | rate_valnorm_ci_high | n_top1_p4_main | n_top1_p4_valnorm | n_segments_top1_changed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm_ae | alarm segment | 72 | 54 | 0.7500 | 61 | 0.8472 | 0.7468 | 0.9125 | 6 | 3 | 57 |
| lstm_ae | event (first alarm segment) | 50 | 43 | 0.8600 | 43 | 0.8600 | 0.7381 | 0.9305 | 6 | 3 | 57 |
| lstm_ae | event (merged) | 50 | 45 | 0.9000 | 46 | 0.9200 | 0.8116 | 0.9685 | 6 | 3 | 57 |
| lstm_vae | alarm segment | 57 | 52 | 0.9123 | 52 | 0.9123 | 0.8106 | 0.9619 | 4 | 2 | 33 |
| lstm_vae | event (first alarm segment) | 48 | 47 | 0.9792 | 44 | 0.9167 | 0.8045 | 0.9671 | 4 | 2 | 33 |
| lstm_vae | event (merged) | 48 | 47 | 0.9792 | 44 | 0.9167 | 0.8045 | 0.9671 | 4 | 2 | 33 |
| pca | alarm segment | 53 | 37 | 0.6981 | 44 | 0.8302 | 0.7077 | 0.9080 | 0 | 1 | 35 |
| pca | event (first alarm segment) | 48 | 36 | 0.7500 | 41 | 0.8542 | 0.7283 | 0.9275 | 0 | 1 | 35 |
| pca | event (merged) | 48 | 34 | 0.7083 | 42 | 0.8750 | 0.7530 | 0.9414 | 0 | 1 | 35 |
