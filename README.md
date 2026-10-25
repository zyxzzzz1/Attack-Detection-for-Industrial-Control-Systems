# Process Anomaly Detection + Vulnerability Context: a Fused Alerting Prototype for Industrial Control Systems

**English** | [中文](README.zh.md)

**Status: code and experiments complete; written report in progress.**

> Asset mapping and vulnerability data come from a lab environment and are for demonstration only.

Unsupervised attack detection on the HAI 21.03 industrial control dataset with an LSTM autoencoder trained on normal
data only, compared against four other models, plus a prototype that attaches vulnerability context from OpenVAS scans
to the resulting alarms.

## Background

Labelled attack data from industrial control systems (ICS) is scarce, so detectors are usually trained on normal
operation only. This project has two parts:

- **Main line (quantitative).** Train an LSTM autoencoder (LSTM-AE) on the normal-only HAI 21.03 training files, score
  the test files second by second, set the threshold on held-out normal data, and evaluate against the attack labels.
  Four comparison models (PCA, IsolationForest, LSTM-VAE, Deep SVDD) use the same data split, windows and evaluation
  code. Alarms are attributed to sensors by per-sensor reconstruction error.
- **Extension (prototype and case study).** Map the top-k sensors of each alarm to a hand-built set of ICS assets,
  combine this with a real OpenVAS scan of a lab target and with CISA KEV and FIRST EPSS data, compute an alert
  priority, and render alert cards.

The asset mapping is constructed by hand and the vulnerabilities come from a self-built lab target. They have no causal
relation to the attacks in HAI. The fusion part does not claim to improve any detection metric.

## Main results

All numbers below are quoted from [`reports/main/key_numbers.md`](reports/main/key_numbers.md), which
lists every number with its definition and source file. They come from full runs on a university SLURM cluster
(NVIDIA L4). Local runs were used only for development and smoke tests, and no smoke-test number appears here.

### Detection

HAI 21.03 test1–5: 402,005 seconds, 8,947 attack seconds (2.23%), 50 attack events.

Conventions for every detection metric:

- **No point-adjust.** Point-wise F1 is therefore low, which is expected.
- Threshold = 99th percentile of the scores on `val_thr`, a held-out normal segment used only for thresholding. An
  alarm is raised when score > threshold.
- The first W−1 seconds of each test file have no score and are excluded: 295 seconds in total, none of them attack
  seconds.
- An event counts as detected if at least one second inside it raises an alarm. False alarms are alarm segments that
  overlap no attack event, per hour of normal time.

| Model | Precision | Recall | F1 | AUROC | Events detected | Median delay (s) | False alarms / hour |
|---|---|---|---|---|---|---|---|
| pca | 0.0946 | 0.8405 | 0.1700 | 0.9014 | 48/50 | 6 | 7.16 (781 segments) |
| iforest | 0.4144 | 0.4828 | 0.4460 | 0.8347 | 35/50 | 28 | 6.84 (746 segments) |
| lstm_ae, seed 42 | 0.2765 | 0.8040 | 0.4115 | 0.9339 | 50/50 | 6 | 14.95 (1631 segments) |
| **lstm_ae, 5 seeds** | 0.2929 ± 0.0168 | 0.8026 ± 0.0055 | **0.4290 ± 0.0184** | **0.9335 ± 0.0016** | 0.9920 ± 0.0110 (49.6/50 on average) | n/a | **13.19 ± 1.68** |
| lstm_vae | 0.0606 | 0.8111 | 0.1128 | 0.8800 | 48/50 | 4 | 21.22 (2315 segments) |
| deep_svdd | 0.1683 | 0.8287 | 0.2797 | 0.9055 | 49/50 | 3 | 69.33 (7564 segments) |

- The 5-seed row is the main LSTM-AE configuration (W=60, hidden=64) with seeds 42–46, reported as mean ± sample
  standard deviation (ddof=1). Every other row is a single seed (42).
- LSTM-AE has the highest AUROC and event detection. For the main configuration fixed in advance (W=60, hidden=64),
  its point-wise F1 cannot be distinguished from IsolationForest: 0.4290 vs 0.4460, a difference of 0.0170 = 0.93
  standard deviations, and 1 of the 5 seeds is above IsolationForest. The seed variation of IsolationForest itself was
  not measured. This statement applies to that configuration only. Other LSTM-AE configurations in the ablation give
  very different F1 values (see below), each from a single seed, and none was used to select a model.
- LSTM-AE raises more false-alarm segments than PCA and IsolationForest.
- Ablation over 9 combinations (W ∈ {30, 60, 120} × hidden ∈ {32, 64, 128}, single seed): F1 0.084–0.617, AUROC
  0.899–0.946, false alarms 0.33–74.66 per hour. The best is W=60, hidden=32 (F1 0.617) and the worst is W=60,
  hidden=128 (F1 0.084). The main configuration was fixed in advance, ranks 5th by F1, and was not changed.

### Attribution

A hit means the subsystem of the top-1 sensor is in the set of subsystems attacked in the event. Only alarm segments
that overlap an attack are counted. Brackets are 95% Wilson intervals. Single seed.

| Model | Per alarm segment | Per event (first overlapping segment) | Always guessing P1, same segments |
|---|---|---|---|
| lstm_ae | 54/72 = 0.750 [0.639, 0.836] | 43/50 = 0.860 [0.738, 0.930] | 0.681 |
| lstm_vae | 52/57 = 0.912 [0.811, 0.962] | 47/48 = 0.979 [0.891, 0.996] | 0.702 |
| pca | 37/53 = 0.698 [0.565, 0.805] | 36/48 = 0.750 [0.612, 0.851] | 0.679 |

- LSTM-AE over 5 seeds (mean ± sample standard deviation): 0.777 ± 0.033 per alarm segment, 0.875 ± 0.026 per event.
- LSTM-AE by segment length: under 10 s 7/20 = 0.350 [0.181, 0.567]; 10–60 s 1/4 = 0.250 [0.046, 0.699]; over 60 s
  46/48 = 0.958 [0.860, 0.988].
- Sensor-level Hit@k was not computed: the HAI 21.03 data files do not give per-sensor attack targets.

### Fusion ranking

Of the 1703 real LSTM-AE alarms, 72 overlap an attack event (base rate 0.042). Ranking by raw score, by fused priority
with α=0, and by fused priority with α=1 all give precision@10 = 10/10, precision@20 = 20/20 and precision@50 = 45/50
(95% interval for precision@50: [0.786, 0.957]).

## Main findings

1. **train1 and test3 overlap in the official release.** The last 43,201 seconds of train1 and the first 43,201
   seconds of test3 are the same data, row for row across all 79 sensors, with no attack in it. In addition, the first
   row of train1 equals the last row of test2. Both were removed from the training set and the test files were left as
   released. Leaving the overlap in could understate false alarms on test3 (not measured). See
   [`reports/main/data_overlap.md`](reports/main/data_overlap.md).
2. **Thresholds do not transfer from validation to test.** By construction 1% of `val_thr` seconds exceed the
   threshold. On test the share is 19.8% for pca, 2.6% for iforest, 6.5% for lstm_ae, 29.8% for lstm_vae and 11.0% for
   deep_svdd. For LSTM-AE it varies widely by file: test1 4.9%, test2 3.6%, test3 2.9%, test4 20.8%, test5 9.0%. The
   likely cause is drift in normal operating conditions between training and test data. This is a hypothesis: which
   sensors drift was not verified separately.
3. **Good detection does not mean accurate attribution.** LSTM-AE is the best detector here but attributes 54/72 alarm
   segments to the right subsystem. LSTM-VAE, with F1 0.1128, attributes 52/57. 34 of the 50 events involve P1, so
   always answering P1 already scores about 0.68; PCA is at that baseline. LSTM-AE misses are concentrated in short
   alarm segments.
4. **Fusion ranking gives no measurable improvement.** The three rankings have identical precision@k. The ranking
   quality comes from the anomaly score, not from the vulnerability context.
5. **Threshold-based metrics vary far more across the ablation than AUROC does.** Over the 9 LSTM-AE configurations,
   AUROC stays within 0.899–0.946, while F1 ranges over 0.084–0.617 and false alarms over 0.33–74.66 per hour. This
   suggests that the threshold-based metrics are driven mainly by threshold transfer and distribution drift (finding
   2), not by how well the model separates attack from normal data. Two caveats: each ablation configuration is a
   single seed, and this reading is a post-hoc interpretation, not a tested hypothesis.

## Repository structure

```
config/             config.yaml (paths and parameters), asset_map.yaml, fusion.yaml, cluster.yaml
src/data/           HAI loading, preprocessing, sliding windows (never reads labels)
src/models/         LSTM-AE, LSTM-VAE, Deep SVDD, PCA, IsolationForest
src/eval/           labels (the only place labels are read), segments, metrics, alarms, fusion_ranking
src/run_model.py    main entry point: prepare → train → diagnose → score → threshold → evaluate
src/attribution.py  attribution and hit-rate evaluation
src/vuln/           OpenVAS report parsing, KEV / EPSS enrichment
src/fusion/         asset graph, risk, prioritisation, card rendering
src/pipeline.py     one-command extension pipeline
app/                Streamlit demo
cluster/            SLURM job scripts, generated from config/cluster.yaml
scripts/            data download, report and figure generation, alert export, cluster sync
templates/          alert card and index templates
tests/              unit tests
data/samples/       OpenVAS report sample, HAI 21.03 column names
data/mock/          mock alarms
data/processed/     alerts.json (real LSTM-AE alarms), vulns.json, vulns_enriched.json
reports/main/       main-line tables, figures and run snapshots; index in reports/main/README.md
reports/extension/  extension figures and tables
docs/               design constraints and decisions, cluster notes, runbook and job queue
PROGRESS.md         development log
```

## Setup

Python 3.10–3.12. Local development uses CPU-only torch.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # all tests, no network access
```

The cluster environment is described in [`env/README.md`](env/README.md) and pinned in `env/requirements-cluster.txt`.

OpenVAS runs as Greenbone Community Containers (Docker Compose). Scans target only hosts inside the host-only lab
subnet set in `gvm.lab_subnet`. Anything that needs GVM credentials is run manually by the operator.

## Reproducing

### Local

```bash
# HAI 21.03: download into data/raw/hai-21.03 and verify size and sha against the official repository
.venv/bin/python -m scripts.download_hai

# Smoke test: 1% of the data, 1 epoch. Checks the pipeline only; its numbers are not results.
bash cluster/train_one.sbatch --smoke lstm_ae

# Extension pipeline: parse → intel → prioritize → render (--offline uses cached KEV / EPSS; --mock uses mock alarms)
.venv/bin/python -m src.pipeline --offline
.venv/bin/python -m scripts.make_extension_report
.venv/bin/streamlit run app/streamlit_app.py
```

### Cluster (SLURM)

Full training, scoring, thresholding, evaluation and ablation run on a cluster. The tracked `config/cluster.yaml` has
placeholders for the user name and paths. Copy `config/cluster.local.example.yaml` to `config/cluster.local.yaml`, fill
it in, and run `python -m scripts.make_cluster_jobs` to generate the real job scripts. The full procedure is in
[`docs/cluster_runbook.md`](docs/cluster_runbook.md) and the job order in
[`docs/cluster_queue.md`](docs/cluster_queue.md) (both in Chinese).

```bash
scripts/push_to_cluster.sh                       # local: rsync tracked files; refuses a dirty working tree

# On the login node, submit only. Compute goes through srun or sbatch.
sbatch cluster/pytest.sbatch                     # the full test suite must pass before any training job
sbatch cluster/baselines.sbatch                  # PCA + IsolationForest (CPU)
sbatch cluster/train_one.sbatch lstm_ae          # likewise lstm_vae, deep_svdd
sbatch cluster/ablation_array.sbatch             # 9 (W, hidden) combinations
```

If `jobs.*.time` is not filled in `cluster.yaml`, pass `--time` explicitly to each `sbatch`. The four extra LSTM-AE
seeds are submitted as in `docs/cluster_queue.md`.

### Tables and figures from cluster artifacts (local)

```bash
scripts/pull_artifacts.sh
.venv/bin/python -m scripts.make_main_report && .venv/bin/python -m scripts.make_figures
.venv/bin/python -m scripts.make_ablation_report && .venv/bin/python -m src.attribution
.venv/bin/python -m scripts.export_alerts && .venv/bin/python -m src.eval.fusion_ranking
.venv/bin/python -m src.pipeline --offline && .venv/bin/python -m scripts.make_cases
.venv/bin/python -m scripts.make_key_numbers
```

### Evaluation rules

- Training uses train1–3 only. The last 20% in time order is validation, split in half into `val_es` (early stopping)
  and `val_thr` (threshold).
- test1–5 are read only for the final evaluation, never for tuning, thresholding, model selection or early stopping.
- The scaler is fitted on the training part only. Windows never cross a file boundary or a split point.
- Labels are read only in `src/eval/labels.py`, `src/evaluate.py` and `src/attribution.py`; an automated test checks
  this.

The full list of numbered constraints and decisions is in [`docs/design_decisions.md`](docs/design_decisions.md).
Wherever a document in this repository cites "constraint N" or "decision N" (约束 N, 决策 N), the number refers to that
list.

## Data

This repository does not distribute the HAI raw data. Download HAI 21.03 (train1–3, test1–5) from the official
repository, <https://github.com/icsdataset/hai>, either with `python -m scripts.download_hai` or by hand into
`data/raw/hai-21.03/` (or the directory given by `HAI_DATA_DIR`). Use of the data is subject to the terms stated in
that repository.

What is tracked here:

- `data/processed/alerts.json`: all 1703 alarm segments of the main LSTM-AE run (seed 42) on the test set, not
  filtered by label (`"is_mock": false`). Most of them are false alarms.
- `data/processed/vulns.json`, `vulns_enriched.json`: a real OpenVAS scan of one lab target (Metasploitable2), with
  KEV / EPSS as of 2026-09-29.
- `data/mock/alerts.json`: mock alarms (`"is_mock": true`), selectable with `--mock`.
- `reports/main/runs/`: small files from each cluster run (metrics, manifest, threshold, preprocessing summary,
  training log), with host names replaced by placeholders.

## Limitations

- **Threshold transfer and false alarms.** The 99th-percentile validation threshold produces far more than 1% alarms
  on test. LSTM-AE averages 13.19 ± 1.68 false-alarm segments per hour, and only 72 of its 1703 alarms overlap an
  attack. Neither the threshold nor the evaluation method was adjusted on the test set.
- **One dataset, mostly one seed.** Evaluation is on HAI 21.03 only. Only the main LSTM-AE configuration was run with
  5 seeds. That configuration was fixed in advance and is not the best in the ablation. LSTM-AE training had not
  triggered early stopping after 50 epochs (best epoch 48).
- **Coarse attribution evaluation.** Hit rates are at subsystem level only, with small denominators and wide
  intervals. P4 has no subsystem label, so a top-1 sensor in P4 always counts as a miss. There is no sensor-level
  Hit@k.
- **Explanations are hypotheses.** The causes offered for the observations (drift in normal operation and others) were
  not verified separately, except that short alarm segments are attributed worse, which the grouped numbers show.
- **Severity saturates.** The anomaly severity S = min(1, max score ÷ threshold − 1) reaches 1 at twice the threshold.
  61 alarms are capped and are no longer separated by score. The formula was left unchanged; improving it is future
  work.
- **One scanned target, saturated risk.** Only one lab host (ews_01) was scanned, so the subsystem exposure E_sub takes
  only the values 0 or 1 and cannot separate P1 from P2. Other upstream assets are unassessed, so risk may be
  understated.
- **QoD.** With min_qod = 70, 66 of 540 vulnerabilities are kept. The rest are mostly remote version-banner matches,
  which can be false positives on systems with backported patches.
- **No causal link.** The asset mapping is constructed by hand and the vulnerabilities are unrelated to the HAI
  attacks. Fusion output is a demonstration of alert enrichment and ranking, not a detection result.

## Disclaimer

Asset mapping and vulnerability data come from a lab environment and are for demonstration only.

OpenVAS was run only against self-built virtual machines on an isolated host-only network.

## Notes

- **Commit hashes.** The commit hashes in `run_manifest.json` files and in `PROGRESS.md` refer to a private
  development repository. They do not exist in this public repository's history.
- **Language.** The code, comments, generated reports, Streamlit app, alert cards and figures are in English.
  Some internal documents (`PROGRESS.md`, most of `docs/`, the `*.zh.md` files) are in Chinese.
- **AI-assisted development.** This project was developed with AI assistance (Claude Code). `docs/cluster_queue.md`
  and `docs/cluster_runbook.md` refer to this workflow. Research design, evaluation methodology and conclusions are my
  own.

## Citation

This project uses the HAI 21.03 dataset. The [official HAI repository](https://github.com/icsdataset/hai) asks that
work using HAI 21.03 cite the following (BibTeX as given in its README, with the abstract field omitted):

```bibtex
@inproceedings{10.1145/3474718.3474719,
    author = {Shin,  Hyeok-Ki and Lee, Woomyo and Yun, Jeong-Han and Min, Byung-Gi},
    title = {Two ICS Security Datasets and Anomaly Detection Contest on the HIL-Based Augmented ICS Testbed},
    year = {2021},
    isbn = {9781450390651},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3474718.3474719},
    doi = {10.1145/3474718.3474719},
    booktitle = {Cyber Security Experimentation and Test Workshop},
    pages = {36–40},
    numpages = {5},
    keywords = {security dataset, testbed, artificial intelligence, hardware-in-the-loop, industrial control system, anomaly detection},
    location = {Virtual, CA, USA},
    series = {CSET '21}
}
```
