# Experiments

One row per run, including failed ones. Scores come from the official `evaluar.py` on the test split.

| # | Model | Params | GFLOPs | imgsz | Epochs | Machine | Train time | Local score | mAP50-95 | F1macro | Best thr | AP per class (sb/cs/fv/ws/wv) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sanity | yolo26n | - | - | 640 | 1 | Mac M4 (mps) | 2.0 min total | 0.0227 | 0.0125 | 0.0466 | 0.05 | 0.000/0.057/0.005/0.000/0.000 | Pipeline check only: 5% of train, 1 epoch. Proves train, evaluar, ZIP and verificar work end to end. |
| exp01 | yolo26s (COCO) | 9.47M | 20.8 @640, ~53 @1024 | 1024 | 34 | AWS g5.xlarge (A10G) | 57 min | 0.4654 | 0.4112 | 0.5921 | 0.36 | 0.364/0.433/0.127/0.742/0.390 | Main model per onboard budget, official baseline script with s, batch 16, --horas 0.95. Best epoch 29. Bottleneck: fishing_vessel AP 0.127 (recall 0.27). Sources: fair1m 0.416, shiprsimagenet 0.584, vhrships 0.729, xview 0.410. |
| exp01-cal | exp01 + per-class thresholds | 9.47M | same as exp01 | 1024 | - | Mac | 1 min | 0.4691 | 0.4112 | 0.6044 | 0.50 (sb 0.35, cs 0.47, fv 0.15, ws 0.59, wv 0.39) | same as exp01 | Free F1 gain with no extra compute. mAP unchanged as required. |
| exp01-tta55 | exp01 + 4-flip TTA, WBF iou 0.55 | 9.47M | same as exp01 | 1024 | - | Mac (mps) | 2 min test | 0.4784 | 0.4248 | 0.6037 | 0.25 | 0.372/0.461/0.147/0.693/0.450 | Warship dropped 0.742 to 0.693: WBF merged neighbors in warship rows. |
| exp01-tta70 | exp01 + 4-flip TTA, WBF iou 0.70 | 9.47M | same as exp01 | 1024 | - | Mac (mps) | 2 min test | 0.4919 | 0.4432 | 0.6057 | 0.28 | 0.383/0.467/0.156/0.765/0.445 | Higher WBF iou keeps adjacent ships apart; warship recovered to 0.765, all classes above exp01. |
| exp03 | yolo26m (COCO) | ~20M | ~3x exp01 per image | 1280 | 11 | AWS g5.xlarge (A10G) | 51 min | 0.4737 | 0.4136 | 0.6137 | 0.22 | 0.311/0.440/0.237/0.699/0.381 | Bigger model at native resolution, scripts/train.py: warmup 1, close_mosaic 2, batch 6, --hours 0.85. Beats exp01 on mAP50 (+0.037) and F1 (+0.022) after only 11 epochs; fishing_vessel almost doubled (0.127 to 0.237); small_boat and warship below exp01, so complementary for an ensemble. |
| exp03-cal | exp03 + per-class thresholds | ~20M | same as exp03 | 1280 | - | Mac | 1 min | 0.4789 | 0.4136 | 0.6313 | 0.50 (sb 0.22, cs 0.22, fv 0.20, ws 0.45, wv 0.36) | same as exp03 | Not uploaded: below exp01-tta70-cal (0.4956). |
| exp01-tta70-cal | exp01-tta70 + per-class thresholds | 9.47M | same as exp01 | 1024 | - | Mac | 1 min | 0.4956 | 0.4432 | 0.6180 | 0.50 (sb 0.32, cs 0.44, fv 0.14, ws 0.52, wv 0.29) | same as exp01-tta70 | Best so far. |
