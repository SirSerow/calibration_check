# Detector confidence and calibration

A reproducible COCO benchmark of YOLO26n, YOLO11n, YOLOX-S, RF-DETR Nano, and original YOLOv3 weights. PicoDet was excluded after repeated GPU smoke-test failures. RTMDet-tiny and SSD300 were excluded at the user’s request due to slow inference. Their configurations and exclusion audits are retained. Results concern these checkpoints and native inference settings, not a controlled effect of architecture age.

Measured tables and figures: [results report](results/report.md). Practical explanation: [article draft](article/article.md).

## Checkpoints and score semantics

| Checkpoint | Input size | Emitted score | Native suppression |
|---|---:|---|---|
| YOLO26n | 640 | Class sigmoid score | End-to-end, no NMS |
| YOLO11n | 640 | Class sigmoid score | Classwise NMS, IoU 0.7 |
| RF-DETR Nano | 384 | Query/class sigmoid score | Query top-k, no NMS |
| YOLOX-S | 640 | Objectness × class probability | COCO evaluator classwise NMS, IoU 0.65 |
| Original YOLOv3 | 416 | Objectness × class probability | Native Darknet letterboxing and classwise NMS, IoU 0.45 |

FP32 math is enforced after model construction, because importing a model library can change PyTorch’s global matmul defaults. NVIDIA_TF32_OVERRIDE=0 also applies before CUDA initialization.

The scores are measured after native postprocessing. Calibration against correct class and IoU ≥0.50 is a shared evaluation target; it is not necessarily the target each original score was trained to estimate. The registry records family introduction dates independently of checkpoint dates.

## Setup and stages

Python 3.10 or 3.11. Analysis runs locally; incompatible inference packages run in separate environments.

```bash
python3 -m venv .venv
.venv/bin/pip install -r configs/requirements-analysis.lock.txt
export PYTHONPATH=src
.venv/bin/python -m model_calibration download-coco
.venv/bin/python -m model_calibration fetch-models
.venv/bin/python -m model_calibration prepare
```

`download-coco --with-images` downloads all validation images as well. COCO uses its public S3 HTTPS endpoint because the vanity image host has a certificate mismatch. Downloads, model repositories, weights, environments, credentials, prediction caches, and logs are ignored by Git. SHA-256 hashes and exact upstream repository revisions are retained in `configs/provenance.lock.json`. The initial family dates are recorded separately from checkpoint dates; unknown checkpoint dates stay null.

Preparation saves exact partitions in `results/runs/coco-v1/manifest.json`. Sorted image IDs are shuffled using NumPy's default generator with seed 42: 1,000 selection, 1,500 calibration, 2,500 final images. Existing runs reject changed settings or annotations. Use a new `--run` directory for a changed protocol.

```bash
# Within each model's isolated inference environment:
python -m model_calibration infer yolo26n --limit 50
python -m model_calibration infer yolo26n

# After every model completes, locally:
.venv/bin/python -m model_calibration evaluate
.venv/bin/python -m model_calibration select
.venv/bin/python -m model_calibration calibrate
.venv/bin/python -m model_calibration report
# Or run the four analysis stages:
.venv/bin/python -m model_calibration analyze
```

Global options (`--run`, `--annotations`, `--images`, `--config`, `--registry`) precede the stage name. `run --interpreters configs/interpreters.json` orchestrates all stages with isolated model executables. Paid remote inference should use the budget-aware worker below. `status` shows each model's explicit pending, smoke, complete, or failed state. Malformed native boxes retain their original coordinates and scores in the cache, with an explicit flag. COCO evaluation uses zero area for an inverted dimension, so these predictions remain false positives rather than being silently removed.

Inference writes one atomic shard per image, even when detections are empty, and resumes by verifying cache identity. Aggregation also resumes completed models after verifying prediction, evaluation-code, package, and match-file hashes. Selection freezes inference; calibration saves its chosen method before reading final matched predictions.

## Matching and metrics

RF-DETR’s sparse COCO head can emit unused class slots at very low scores. Those have no COCO class name and are saved separately as `excluded_non_coco_class_slots`, rather than assigned an invented category. Only assigned COCO categories enter shared evaluation.

Adapters emit original-image xyxy boxes, sparse COCO category IDs, native scores, and stable `model:image:index` detection IDs. FP32 disables autocast, half precision, and PyTorch TF32. Native preprocessing, suppression, and proposal limits are retained. Inference keeps scores ≥0.001 and the top 100 detections per image. PicoDet uses its no-postprocess export and explicitly lowers the Python postprocessor's hidden 0.4 score threshold to 0.001. Its exported FP32 graph runs with Paddle IR optimization disabled because the cuDNN fused activation kernel is unsupported on this runtime; native preprocessing and NMS are retained.

The shared evaluator calls the official `pycocotools.COCOeval` implementation. Correctness means correct class and IoU ≥0.50, with greedy score-ordered matching, duplicate handling, and reusable crowd matches. Ignored detections do not enter calibration statistics. Matching at 0.75 provides sensitivity analysis. COCO mAP (0.50:0.95), AP50, AP75, and AR100 use all cached scores ≥0.001. Calibration metrics use the separate original-score ≥0.25 population.

Precision ECE is `sum_bin abs(sum_confidence - sum_correct) / N`, with 15 equal-width bins. Intervals are left-inclusive, right-exclusive except that confidence 1 belongs to the last bin. Binary NLL clips probabilities to [1e-12, 1−1e-12]; Brier uses original probabilities. Empty populations yield null metrics. Mean confidence, observed precision, counts, and bin tables accompany ECE.

Each 95% interval uses 1,000 image-level bootstrap resamples with replacement. Empty images remain in the sampling universe. Sufficient statistics preserve the same detection-weighted estimand while avoiding repeated materialization of detections. Degenerate all-empty resamples are skipped and their count is reported. Sensitivity checks vary original thresholds (0.05, 0.10, 0.50), IoU (0.75), or bin counts (10, 20), one factor at a time. Selection interval overlap is reported as uncertainty; it is not proof of equivalence.

The highest selection ECE among YOLO26n, YOLO11n, and RF-DETR Nano determines the checkpoint. Exact ECE ties use higher NLL, then ascending model ID. Temperature correction is `sigmoid(logit(score)/T)`, T>0; Platt correction is `sigmoid(a*logit(score)+b)`, a>0. Both fit binary NLL. Five calibration folds group by image; pooled held-out detection NLL chooses the method, preferring temperature within 1e-9. Both methods are refitted on the full calibration partition and reported on final data.

This is an emitted-score adaptation of classifier temperature scaling. The same original-score ≥0.25 detection IDs, boxes, categories, matching labels, and rankings are retained after correction. No new threshold is applied. Positive slopes preserve rankings mathematically. Saved checks verify fixed precision and logit ranking; mAP is unchanged by retaining original detection order. Sigmoid floating point saturation can create equal probabilities, so downstream ranking must retain original order rather than re-sort rounded corrected probabilities. Paired image resamples estimate after-minus-before metric intervals without refitting calibrators.

## RunPod execution and costs

Load `RUNPOD_API_KEY` and `HASHNODE_PAT` from the execution environment or an ignored project `.env`. Never paste credentials into a command or a document. The remote project upload excludes `.env`, credential directories, and local environments.

```bash
.venv/bin/python infra/runpod/control.py quotes
.venv/bin/python infra/runpod/control.py provision --public-key /path/to/project-key.pub
.venv/bin/python infra/runpod/control.py upload --key /path/to/project-key
.venv/bin/python infra/runpod/control.py status
```

The controller prefers one on-demand RTX 4090, then RTX 3090 or A5000. Live compute and running storage prices must total ≤$1/hour. It tracks all project pod sessions against $10 and reserves $1 for retrieval and cleanup. The container image digest is pinned; the 80 GB temporary disk has no stopped-volume charge. A server-side termination deadline limits unattended spending. Network volumes are not created. Credentials are not sent to the GPU pod.

On the pod, in `/workspace/model-calibration`:

```bash
export PYTHONPATH=src
python infra/runpod/setup.py
python -m model_calibration download-coco --with-images
# The uploaded checkpoints and reference repositories are already locked.
python infra/runpod/worker.py --hourly-usd LIVE_TOTAL_RATE --remaining-usd LIVE_REMAINING_BUDGET
```

Setup saves complete resolved package locks and builds native Darknet with CUDA and FP32. The worker runs smoke inference for all five models before full inference, measures elapsed throughput, and adds a 50% runtime margin. It refuses a full run that exceeds the remaining allocation. Setup time is included in the controller's spending, and the worker's remaining-budget argument must come from a fresh controller status after setup. Logs and model failures remain explicit.

Run `control.py monitor --key /path/to/project-key` locally during setup and inference to retrieve completed shards every 30 seconds. After completion, sync again and verify the artifact inventory locally before deleting the recorded project pod:

```bash
.venv/bin/python infra/runpod/control.py sync --key /path/to/project-key
.venv/bin/python infra/runpod/worker.py --verify-local --hourly-usd LIVE_TOTAL_RATE --remaining-usd LIVE_REMAINING_BUDGET
.venv/bin/python infra/runpod/control.py terminate
.venv/bin/python -m model_calibration analyze
```

The watchdog terminates at the retrieval reserve if inference fails to finish. The server deadline remains a second safeguard if the monitor disconnects. Only recorded project pod IDs may be terminated. RunPod stopped pods can retain billable storage; terminate the project pod after verifying retrieval. Actual charges can be reconciled with the account billing view; the saved spend ledger is an elapsed-time estimate.

## Artifacts and article

`summary.json`, `metrics.csv`, `bins.csv`, `selection.json`, `calibrators.json`, `correction_results.json`, and `report.md` retain the numerical evidence. `results/figures` contains PNG/SVG comparisons, all-model reliability curves and histograms, detection-quality versus calibration charts, correction reliability diagrams, and paired metric-change charts.

`article/article.template.md` is a source-linked article template whose result placeholders are filled only from validated metrics. The final `article/article.md` must be 800–1,100 words with three principal figures. Article hashes bind it to summary and correction results. Production reporting refuses incomplete five-model coverage. Synthetic verification stays explicitly labeled and cannot be uploaded as benchmark evidence.

The target is `https://olegserov.hashnode.dev/`. Draft upload follows the project [gql-api skill](.agents/skills/gql-api/SKILL.md), uses the documented beta GraphQL endpoint, uploads and confirms PNG images, then creates or updates an unpublished draft and reads back its exact Markdown. It never calls a publish mutation. A documented Pro-gate response is retained as an explicit blocker, not retried. Domain setup does not require publishing.

```bash
.venv/bin/python -m model_calibration draft --publication https://olegserov.hashnode.dev/ --methodology-url https://PUBLIC_METHODOLOGY_URL
```

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Tests cover hand-checkable calibrated/overconfident/underconfident scores, bin endpoints, NLL/Brier, duplicate/wrong-class/crowd matching, empty predictions, class mapping, grouped validation, positive corrections, disjoint splits, frozen decisions, incomplete model coverage, resumability, deterministic aggregation, and PNG/SVG/article generation. The end-to-end test uses synthetic detector records and establishes software behavior, not empirical checkpoint quality. GPU smoke diagnostics are saved in the run. `infra/runpod/reference_check.py` compares cached predictions for two images with the native package APIs using independent class mapping and serialization; Darknet is checked against its CLI. Darknet’s JSON exporter has a hardcoded 0.005 score floor, so its reference comparison uses 0.005001 while inference still retains 0.001. Per-model reports save the maximum score and box differences. `worker.py --verify-local` verifies retrieval hashes and exact image coverage before cleanup.
