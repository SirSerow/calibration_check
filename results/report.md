# Detector calibration results

COCO val2017 benchmark of specific checkpoints.

Excluded checkpoints: picodet_s: User requested exclusion after repeated GPU smoke failures, before evaluation or selection; rtmdet_tiny: User requested exclusion due to slow inference, before selection and calibration; ssd300: User requested exclusion due to slow inference, before selection and calibration

Primary metrics: original scores ≥0.25, correct class, COCO IoU ≥0.50, 15 equal-width bins.
COCO mAP and AR100 use the full cached ≥0.001 prediction population. All values are fractions.
Intervals are 95% percentile intervals from 1,000 image-level bootstrap samples (seed 42).
Ignored crowd matches are excluded. Empty images stay in the bootstrap universe.

| Checkpoint | ECE [95% CI] | NLL | Brier | Confidence | Precision | Count | mAP | AR100 |
|---|---|---|---|---|---|---|---|---|
| yolo26n | 0.1664 [0.1578, 0.1748] | 0.4936 | 0.1653 | 0.6181 | 0.7845 | 10893 | 0.4037 | 0.5920 |
| yolo11n | 0.1369 [0.1281, 0.1456] | 0.5083 | 0.1699 | 0.5965 | 0.7334 | 12642 | 0.3910 | 0.5457 |
| rfdetr_nano | 0.0864 [0.0807, 0.0923] | 0.4683 | 0.1520 | 0.5489 | 0.5750 | 22432 | 0.4870 | 0.6373 |
| yolox_s | 0.0471 [0.0425, 0.0542] | 0.4739 | 0.1552 | 0.6247 | 0.6617 | 16772 | 0.4065 | 0.5613 |
| yolov3 | 0.0255 [0.0186, 0.0332] | 0.3943 | 0.1316 | 0.7165 | 0.7420 | 15505 | 0.3823 | 0.4935 |

Adapter diagnostics on final data (retained malformed boxes and excluded unused COCO slots):

- yolo26n: {'invalid_boxes': 0, 'invalid_boxes_above_primary': 0, 'excluded_non_coco_slots': 0, 'excluded_non_coco_slots_above_primary': 0}
- yolo11n: {'invalid_boxes': 0, 'invalid_boxes_above_primary': 0, 'excluded_non_coco_slots': 0, 'excluded_non_coco_slots_above_primary': 0}
- rfdetr_nano: {'invalid_boxes': 0, 'invalid_boxes_above_primary': 0, 'excluded_non_coco_slots': 3, 'excluded_non_coco_slots_above_primary': 0}
- yolox_s: {'invalid_boxes': 0, 'invalid_boxes_above_primary': 0, 'excluded_non_coco_slots': 0, 'excluded_non_coco_slots_above_primary': 0}
- yolov3: {'invalid_boxes': 0, 'invalid_boxes_above_primary': 0, 'excluded_non_coco_slots': 0, 'excluded_non_coco_slots_above_primary': 0}

Selected using selection data: **yolo26n**.
Overlapping selection ECE intervals: [].
Chosen using grouped calibration CV: **platt**.

| Method | ECE change [paired 95% CI] | NLL change | Brier change |
|---|---|---|---|
| temperature | -0.0136 [-0.0146, -0.0125] | -0.0070 | 0.0002 |
| platt | -0.1533 [-0.1620, -0.1377] | -0.0832 | -0.0309 |

Correction preserves the original population, classes, boxes, matching, and rankings. No corrected-score threshold is applied.
Precision and COCO AP are checked numerically. AP is re-evaluated using corrected logits for sorting on all originally retained detections, avoiding sigmoid saturation ties.
Sensitivity metrics and intervals are in summary.json. Failed models cannot silently disappear.
These measurements do not establish a causal effect of architecture age or measure confidence for missed objects.

Article: article/article.md
Full provenance, partitions, native suppression settings, and timing: manifest.json.
