# Design constraints and decisions

Documents in this repository refer to rules by number, for example "constraint 7" (约束 7) or "decision 6b" (决策 6b,
also written 主线补充决策 6b). The numbers come from the project's working instructions, a file named `CLAUDE.md` that
is not part of this repository. This page lists every numbered constraint and decision with its content, so that those
references can be resolved.

All of them were fixed before the results they govern were produced. Where a date is given, it is the date the
decision was made. Decision 6b in particular was fixed before the first evaluation on the test set.

## Hard constraints

Breaking any of these counts as a failed stage.

| No. | Constraint |
|---|---|
| 1 | Training uses train1–3 only (all normal data). After concatenation in time order, the last 20% is the validation set. No shuffling. |
| 2 | test1–5 (attacks and per-second labels) are read only for the final evaluation. They are not used for tuning, thresholding, model selection or early stopping. |
| 3 | Threshold = 99th percentile of the normal scores on `val_thr`, the threshold part of the validation set (see decision 1). |
| 4 | No point-adjust evaluation, because it inflates the metrics. |
| 5 | The scaler is fitted on the training set only. |
| 6 | All models use the same data split, window settings and evaluation code. |
| 7 | The asset mapping of the extension is constructed by hand, and the vulnerability data comes from scans of a self-built target. Neither has a causal relation to the attacks in HAI. The fusion part must not claim to improve detection metrics; it is a prototype demonstration and case study only. |
| 8 | OpenVAS scans only self-built lab virtual machines on an isolated network. |
| 9 | The configuration of the main model (LSTM-AE) is fixed in advance: W=60, hidden=64. Ablation results are for sensitivity analysis only and must not be used to change the main model. |
| 10 | Sliding windows must not cross any boundary: file boundaries, the train/validation split point, or the split point inside the validation set. |

Note on numbering: `reports/main/data_summary.md` cites "约束 8" for the rule that windows must not cross file
boundaries. That is an older number for what is constraint 10 in this list.

## Decisions

### 1. Validation split

The validation set (the last 20%) is split in half in time order. The first half, `val_es`, is used only for early
stopping. The second half, `val_thr`, is used only to compute the threshold. The order is `train | val_es | val_thr`,
and the split points are treated as boundaries. Reason: using the same data for early stopping and for the threshold
would make the threshold optimistic.

### 1b. Overlap between train and test

The last 43,201 seconds of the official train1 and the first 43,201 seconds of test3 are the same data (identical row
by row across the 79 sensors, no attack). See [`reports/main/data_overlap.md`](../reports/main/data_overlap.md).

- These 43,201 seconds are removed from the training set. The test files are kept as released. The switch is
  `ml.data.drop_train_overlap` (on by default); the removed rows are described in `ml.data.train_overlap_drops` as row
  counts at the head and tail of a file, and the training pipeline does not read test files.
- Principle: no test row appears in the training data, not even one. Row 1 of train1 equals the last row of test2 and
  is removed under the same switch.
- Constant columns are determined and the scaler is fitted after the removal.
- A test asserts that no training window contains a row of the first 43,201 seconds of test3, checked by a hash of the
  sensor values and not only by timestamp.
- The data part and the limitations part of the report both cite this finding.

### 2. Label isolation

- Label reading is concentrated in `src/eval/labels.py`.
- Only `evaluate.py` and the attribution hit-rate evaluation may import it.
- `src/data/` drops the `time` and `attack*` columns when reading data.
- A test scans every `.py` file under `src/` and confirms that only whitelisted files refer to the labels module or
  the string "attack".

### 3. Common model interface

Every model implements `fit(train_ds, val_es_ds)` and `score(ds) -> (scores, per_sensor_err | None)`.

- A score is a per-window value assigned to the last second of the window.
- The first W−1 seconds of each test file are NaN.
- All metrics exclude NaN points, and the report states the number of seconds excluded.

### 4. Training defaults

- Adam, lr=1e-3, batch=256, max_epochs=50, early stopping with patience=5 on the `val_es` loss.
- Training stride 10; validation and test stride 1.
- Windows are sliced on demand in the Dataset and not expanded in advance; features are float32.
- The device is detected automatically. Training hyperparameters such as batch=256 are the same on CPU and GPU.

### 5. Model details

- **LSTM-AE**: the encoder LSTM's last hidden state is repeated W times and fed to the decoder LSTM, followed by a
  Linear layer. The score is the window MSE. `per_sensor_err` is the squared error of each sensor at the last second of
  the window.
- **LSTM-VAE**: loss = reconstruction + β·KL, β = 1.0 by default. The score is the reconstruction error.
  `per_sensor_err` is output in the same way.
- **Deep SVDD**: the LSTM encoder has no bias terms and no normalisation layer with affine parameters. The centre c is
  the mean of the initial forward outputs, with components close to 0 set to ±eps, and is then fixed. The score is the
  distance to c. It outputs no `per_sensor_err` and takes no part in attribution.
- **PCA**: keeps 95% of the variance. The score is the reconstruction error; `per_sensor_err` is the per-second
  reconstruction error.
- **IsolationForest**: the input is per-window, per-sensor statistics (mean, std, min, max, last − first). The score
  is −score_samples.

### 6. Alarm segments and events

- `src/eval/segments.py` provides `to_segments(binary, merge_gap_s)`. The default `merge_gap_s=0` merges only
  consecutive alarms. The event-level statistics of `evaluate.py` and `export_alerts.py` call this same function.
- Attack event = a maximal run of consecutive seconds with attack=1.
- An event is detected if at least one second inside it raises an alarm.
- Detection delay = time of the first alarm − start of the event, over detected events only; the number of missed
  events is reported as well.
- False alarm rate = number of alarm segments overlapping no attack event ÷ normal time in hours.

### 6b. Metric conventions and scaling (2026-10-07, before the first evaluation on test; not changed afterwards)

- Alarm rule: score > threshold. Unscored seconds (NaN) raise no alarm and enter no metric.
- With no alarm at all, precision is NaN. If the evaluated range contains attacks, recall = 0 and F1 = 0.
- If the evaluated range contains no attack, recall and F1 are undefined and recorded as NaN.
- AUROC depends only on the scores. It is NaN only when the evaluated range has a single label class.
- Normal time counts only seconds that are scored and labelled 0.
- Heavy-tail diagnostic (PCA, `val_thr` only): if a single sensor's share exceeds 30%, the run stops before reading
  test.
- Scaling stays StandardScaler, fitted on the train part only, with no lower bound on std. Basis: in the PCA
  diagnostic the largest share on `val_thr` is P3_FIT01 at 9.0%, far below 30%.
- The scaling is locked by this decision. For LSTM-AE and LSTM-VAE the per-sensor share on `val_thr` is reported only
  and never stops a run.
- The Deep SVDD centre rule is not changed. `run_manifest.json` records the minimum, median and maximum of |c| before
  replacement and the number of replaced dimensions.
- Local diagnostic runs are marked `local_diagnostic=true` in the manifest and are rejected by the report scripts.
- Smoke-test numbers are never written into the report.

### 6c. Multiple random seeds (2026-10-09)

- Only for the main LSTM-AE configuration (W=60, hidden=64): 5 seeds, 42 (the main run itself) and 43–46.
- The LSTM-AE row of the main table gives mean ± sample standard deviation for F1, AUROC, event detection rate and
  false alarm rate. The other models keep a single seed, and the report says so.
- Purpose: to judge whether the F1 difference between LSTM-AE and IsolationForest is within random variation. The main
  model, threshold and evaluation method are not changed because of it.
- Alarm export, attribution and figures use only the seed-42 main run.

### 7. Attribution

- Within each alarm segment, the mean `per_sensor_err` of each sensor is normalised to a share, and the Top-k (k=3 and
  5) are taken.
- The subsystem hit rate is computed only for alarm segments overlapping an attack event: a hit means the subsystem of
  the Top-1 sensor is in the event's subsystem set.
- When a segment overlaps several events (2026-10-07): the main definition uses the event with the longest overlap
  (the earlier one on a tie). The union definition is output as a supplement.
- Subsystem set of an event = union of attack_P* over all seconds of the event. An event with an empty union is left
  out of the denominator and listed separately. In HAI 21.03, 179 seconds have attack=1 with all subsystem labels 0;
  all are inside events, and the union is non-empty for all 50 events.
- Detection metrics use only the `attack` column and are not affected by subsystem labels.
- The share of Top-1 sensors in P4 is reported separately. P4 has no subsystem label.
- Alarm segments not overlapping an attack are listed separately and do not count towards the hit rate.
- Sensor-level Hit@k is computed only if an official table of attack targets is found. Otherwise it is skipped and
  recorded as a limitation.
- Per-event hit rate (2026-10-09): the main definition takes the first alarm segment overlapping the event. The
  original definition (all alarm seconds inside the event merged) is kept as a supplement.
- All hit rates carry 95% Wilson intervals. Hit rates are also reported by segment length (<10 s, 10–60 s, >60 s).
  The 5 LSTM-AE seeds are each evaluated, and mean ± standard deviation is reported.
- Normalising by the mean error on `val_thr` before ranking is an exploratory post-hoc analysis only. It is output
  separately, labelled as such, and does not replace the main result.
- An alarm's share is relative to all sensors, so the Top-k shares sum to at most 1.

### 8. The alerts.json interface

- The field contract is taken from `data/mock/alerts.json` and the extension code that parses it.
- Real alarms have exactly the same fields as the mock ones (`is_mock=false`, with the model name in `model`).
- A pydantic contract test checks both the mock and the real output.
- The interface of the extension is not changed.

### 8b. Integration (2026-10-09)

- `alerts.json` contains all LSTM-AE alarm segments (1,703), not filtered by label. Cards and the Streamlit demo render
  only the top N by priority; N is a config parameter, 50 by default.
- Fusion ranking is evaluated by precision@k (k=10, 20, 50) for three rankings: raw score, α=0 and α=1. It is computed
  in `src/eval/fusion_ranking.py`; `export_alerts` does not touch labels.
- Three cases are chosen to reflect the results honestly: one correctly attributed real attack, one wrongly attributed
  real attack, and one high-priority false alarm.
- The anomaly severity S is capped at 2 times the threshold. This is recorded as a limitation and future work; the
  formula is not changed.

### 8c. Display language (2026-10-09)

- `ui.lang` (en / zh, default en); the environment variable `UI_LANG` overrides it.
- It covers the Streamlit app, alert cards, the index page, `cases.md`, disclaimers and all figure text.
- Only display text changes. No computation, field name or number changes, and tests check that the values are
  identical in both languages.
- The disclaimer is translated exactly: "Asset mapping and vulnerability data come from a lab environment and are for
  demonstration only."

### 9. Portability

- No absolute path appears in the code. The data and artifact directories come from the config and can be overridden
  by `HAI_DATA_DIR` and `ARTIFACT_DIR`.
- The device is detected automatically. With a GPU, `cudnn.deterministic=True` and `benchmark=False` are set, and all
  random seeds are fixed.
- Each run directory gets a `run_manifest.json` with the git commit hash, a hash of the config, the full
  hyperparameters, library versions, device, host name, start and end time, final epoch and early-stopping epoch.
- `alerts.json` is generated only from the artifacts of the full LSTM-AE cluster run.

### 10. Artifact locations

- Intermediate artifacts (models, scores, checkpoints) go under `artifacts/{model}/` and are not tracked.
- Report material goes under `reports/main/` and is tracked.
- The interface file is `data/processed/alerts.json` (real data, tracked).
