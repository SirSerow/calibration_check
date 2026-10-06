A confidence score of 0.90 looks reassuring. But does an object detector actually get about nine out of ten similar predictions right?

That question motivated my first article.

Guo and colleagues showed that newer neural networks could make better predictions while giving less reliable confidence scores. Their paper, “On Calibration of Modern Neural Networks,” inspired a test of whether a similar pattern appears in object detectors:
https://arxiv.org/abs/1706.04599

I compared five pretrained detectors on 5,000 COCO images, using separate images to choose a model, fit a correction, and test the result.

YOLOv3 had the lowest calibration error. YOLO26n, YOLO11n, and RF-DETR Nano all had higher error, resembling the original paper’s pattern. This comparison does not prove that model age caused the difference.

YOLO26n was too cautious: average confidence was 61.81%, while 78.45% of its detections were correct. A simple Platt correction reduced its expected calibration error from 16.64 to 1.31 percentage points on the final test images. The boxes and mAP stayed unchanged.

Better detection and more reliable confidence are separate things. A small score correction can help with the second.

This is my first article. The full comparison, charts, and explanation are on my Hashnode blog:
https://blog.olegserov.com
