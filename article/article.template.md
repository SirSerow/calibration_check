# Can you trust an object detector’s confidence?

An object detector draws a box, assigns a class, and gives you a score. A score of 0.90 looks reassuring. But does it mean that roughly nine out of ten comparable detections are correct? That is a question we can measure. It matters when scores guide human review, automatic alerts, or decisions about which predictions to keep.

Accuracy and confidence answer different questions. A detector can locate many objects successfully while overstating how often its predictions are right. It can also be accurate and too cautious. Calibration asks whether the scores agree with observed correctness across many predictions. It does not ask whether one particular box is guaranteed to be correct.

[Guo and colleagues](https://arxiv.org/abs/1706.04599) studied this mismatch in neural network classifiers and showed that temperature scaling often helped. Their findings motivate this experiment, but do not establish that recent object detectors are worse calibrated than older ones. Detection introduces additional choices: a prediction can have the right class and still put its box in the wrong place.

For this benchmark, correctness means the class matches a COCO annotation and the box has intersection over union of at least 0.50. The official COCO matching procedure handles competing detections and duplicates. Detections that COCO ignores, including qualifying crowd matches, do not enter calibration statistics. Changing the overlap criterion changes what the score is being asked to predict.

## Five checkpoints, one explicit protocol

We compare YOLO26n, YOLO11n, YOLOX-S, RF-DETR Nano, and the original YOLOv3 COCO weights. PicoDet was excluded after repeated GPU smoke-test failures. RTMDet-tiny and SSD300 were excluded at the user’s request because inference was too slow, before selection or calibration. These omissions limit the comparison. Each evaluated checkpoint retains its documented input size and native suppression behavior. Inference uses FP32, a minimum score of 0.001, and at most 100 detections per image. This comparison concerns these checkpoints and settings; model size, training, preprocessing, and suppression are not controlled experiments on architecture age.

The 5,000 COCO val2017 images are shuffled reproducibly by image ID with seed 42. We reserve 1,000 images for selecting the checkpoint to correct, 1,500 for fitting and validating the correction, and 2,500 for final evaluation. The exact IDs, checkpoint hashes, package versions, and inference settings are saved alongside the results. Caching predictions also lets us reproduce the analysis without repeating GPU inference. RF-DETR’s unused COCO class slots are excluded and audited separately.

The primary analysis uses original scores of at least 0.25. We compute expected calibration error, or ECE, using 15 equal-width confidence bins. In each occupied bin, we compare mean confidence with observed precision. ECE averages the absolute gaps, weighted by the number of detections. Smaller is better, but a single aggregate number can conceal differences between classes or object sizes.

We therefore also report binary negative log likelihood and Brier score, plus mean confidence, observed precision, and detection counts. COCO mAP and recall describe detection quality using the fuller prediction cache. Confidence thresholds of 0.05, 0.10, and 0.50, overlap of 0.75, and 10 or 20 bins provide sensitivity checks. Uncertainty comes from 1,000 bootstrap samples drawn by image, keeping detections from an image together.

![Calibration comparison across the five checkpoints, with image-level confidence intervals.]({{figure_prefix}}/calibration_comparison.png)

On the final partition, **{{best}}** has the lowest primary ECE, **{{best_ece}} percentage points**. {{age_result}} The interval bars matter when comparing nearby values. A point estimate alone should not turn a small difference into a strong ranking claim. The fuller tables and reliability diagrams are linked in the [reproduction methodology]({{methodology}}).

## Correcting the score without moving the box

Among YOLO26n, YOLO11n, and RF-DETR Nano, **{{selected}}** has the highest primary ECE on the selection partition. That decision was saved before fitting. {{selection_uncertainty}} Selecting a checkpoint on one partition and describing its final results on another helps avoid rewarding a correction chosen after seeing the answer.

We fit two small transformations. Temperature correction computes `sigmoid(logit(score) / T)`, where `T` is positive. Platt correction computes `sigmoid(a * logit(score) + b)`, with positive `a`. The temperature approach is an adaptation applied to emitted detector probabilities; it is not a claim that these scores are the multiclass logits used in Guo’s original classifier experiments.

Both transformations minimize binary negative log likelihood on calibration data. Five-fold validation keeps every image’s detections in the same fold. The validation losses choose **{{chosen}}**, with temperature preferred on a numerical tie. Both methods are then refitted on the entire calibration partition. The chosen correction has parameters **{{parameters}}**. Neither method is chosen using final evaluation results. For example, the chosen mapping turns 0.90 into {{example_090}}.

![Selected checkpoint before correction and after each fitted transformation.]({{figure_prefix}}/correction.png)

On final images held out from correction fitting and validation, original mean confidence is {{mean_confidence}}%, versus {{precision}}% observed precision. The chosen method changes ECE from **{{before_ece}}** to **{{after_ece}} percentage points**. The change is **{{delta_ece}}**, with a paired 95% interval of **[{{delta_lo}}, {{delta_hi}}]** percentage points. Negative log likelihood changes from **{{before_nll}}** to **{{after_nll}}**. {{conclusion}}

The evaluation keeps the same **{{count}} detections**, selected by their original scores. We do not apply the 0.25 threshold again after correction. Fixed-population precision remains **{{precision}}%**. Boxes, classes, and matching labels stay fixed, and the positive transformations preserve score ordering. Retaining that ordering also preserves mAP. Calibration changes how a score is interpreted; it does not recover an object the detector missed.

![Detection quality plotted against calibration error.]({{figure_prefix}}/quality_vs_calibration.png)

## What to take into production

A useful correction must be fitted on representative data and checked on data it has never seen. Keep the operating population explicit: lowering a threshold after fitting changes which predictions you evaluate. Watch negative log likelihood and Brier score alongside ECE, because optimizing one summary does not guarantee improvement in all the others.

This study measures calibration of emitted detections. [Detection calibration research](https://arxiv.org/abs/2004.13546) also considers location and scale, which a single global score transformation does not capture. Missed objects, shifts in camera conditions, and performance within individual classes need separate checks. The practical lesson is to measure confidence against the correctness rule your application needs, then validate any correction before relying on its scores.
