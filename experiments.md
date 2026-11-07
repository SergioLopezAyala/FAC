# Experiments

One row per run, including failed ones. Scores come from the official `evaluar.py` on the test split.

| # | Model | Params | GFLOPs | imgsz | Epochs | Machine | Train time | Local score | mAP50-95 | F1macro | Best thr | AP per class (sb/cs/fv/ws/wv) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sanity | yolo26n | - | - | 640 | 1 | Mac M4 (mps) | 2.0 min total | 0.0227 | 0.0125 | 0.0466 | 0.05 | 0.000/0.057/0.005/0.000/0.000 | Pipeline check only: 5% of train, 1 epoch. Proves train, evaluar, ZIP and verificar work end to end. |
