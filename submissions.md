# Submissions

One row per uploaded ZIP. Local score is the official `evaluar.py` result on the test split for the same pipeline.

| Time | ZIP | Local test score | Public score | Git commit | Threshold |
| --- | --- | --- | --- | --- | --- |
| about 15:10 | envio_yolo26s_1024_calibrado.zip (exp01-cal) | 0.4691 | 0.4912 | e17937d4 | 0.50 (per-class calibrated) |
| about 15:28 | envio_yolo26s_1024_tta_calibrado.zip (exp01-tta70-cal) | 0.4956 | 0.5127 | e17937d4 | 0.50 (per-class calibrated) |
| about 15:55 | envio_yolo26m_1280_calibrado.zip (exp03-cal) | 0.4789 | 0.4868 | e17937d4 | 0.50 (per-class calibrated) |
| (team run) | envio_mejor (team yolo26s, 2.25 h, own TTA pipeline) | ~0.528 proxy | 0.5263 | - | 0.50 |
| about 16:55 | envio_final_ens3_calibrado.zip (exp05 m-ft + exp01 s + team s, flips4, WBF 0.70 avg) | 0.5410 | 0.5476 | 249c2ae4 | 0.50 (per-class calibrated) |
| about 17:16 | envio_final_d8_calibrado.zip (same 3 models, 8 views, weights 1.5/0.5/1.5, WBF 0.70 avg) | 0.5635 | 0.5658 | b66fc978 | 0.50 (per-class calibrated) |
