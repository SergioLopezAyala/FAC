# Vessel Detection and Classification in Satellite Imagery

[Leer en espanol](README.es.md)

A detector that finds every vessel in very-high-resolution optical satellite images, draws a horizontal box around it, and classifies it into one of five types: small boat, cargo ship, fishing vessel, warship or work vessel. It combines three YOLO26 detectors, eight-view test-time augmentation, Weighted Boxes Fusion and per-class confidence calibration, and every result in this repository was measured with the official challenge evaluator.

## Achievement

> **3rd place** at the SpaceLAB Hackathon FAC + PUJ (GeoVision Colombia), Pontificia Universidad Javeriana, Bogota, 7 November 2026.
>
> - Team: **DualCore** (Sergio Lopez and Santiago Rodriguez).
> - Best public leaderboard score: **0.5658** (Score = 0.7 x mAP@0.5:0.95 + 0.3 x F1macro@0.5).
> - Final ranking on the private split: **3rd place**.

## The challenge

The hackathon was organized by the Faculty of Engineering of the Pontificia Universidad Javeriana (PUJ) and the Colombian Aerospace Force (Fuerza Aeroespacial Colombiana, FAC). The work is linked to an Earth-observation satellite that the FAC plans to launch in 2027 for defense and maritime surveillance.

Colombia has coastlines on two oceans and a large maritime territory to monitor. Knowing which vessels are present, where they are and what type they are matters for security and defense, for fighting illegal fishing and smuggling, for search and rescue, and for port management. Very-high-resolution satellite images cover large areas of sea and coast, but reviewing them by hand is slow and expensive. The challenge asked for a model that does this automatically.

**Task.** For each image, output a list of horizontal boxes, each with a class and a confidence between 0 and 1.

| ID | Class | Includes |
| --- | --- | --- |
| 0 | small_boat | Launches, small boats, yachts, recreational craft |
| 1 | cargo_ship | Container ships, bulk carriers, tankers, cargo barges |
| 2 | fishing_vessel | Fishing boats |
| 3 | warship | Frigates, destroyers, aircraft carriers, naval support ships |
| 4 | work_vessel | Tugs, dredgers, floating cranes, work ships |

**Data.** The organizers combined four public remote sensing collections (FAIR1M, ShipRSImageNet, VHRShips and xView) into the five classes, in YOLO label format. The data is not included in this repository.

| Split | Images | Objects | Use |
| --- | --- | --- | --- |
| train | 5,926 | 22,155 | Training, with labels |
| test | 801 | 3,964 | Local validation, with labels |
| hidden | 1,743 | 6,811 | Official evaluation, no labels |

Key properties, from the challenge document and our own audit of every image and label:

- Objects are tiny: a vessel covers about 0.3% of the image on average, and many cover less than 0.05%.
- Scenes can be dense, with more than 200 vessels in one image (ports, shipyards, naval bases).
- Images are JPEG RGB, 600 to 1,500 pixels per side, from different sensors, resolutions and annotation styles.
- The view is top-down, so vessels appear in any orientation.
- work_vessel is the minority class in training (1,712 boxes, 7.7% of the training boxes).

**Evaluation.** Score = 0.7 x mAP@0.5:0.95 + 0.3 x F1macro@0.5.

- mAP is COCO-style average precision (101-point interpolation) over IoU thresholds 0.50 to 0.95, averaged over the five classes. It uses every submitted box, up to 300 per image.
- F1 counts only boxes with confidence at or above the team threshold (`umbral.txt` in the submission), at IoU 0.5, averaged over the five classes.
- The leaderboard showed the score on a random 30% of the hidden set (523 images). The final ranking took each team's best public submission and scored it on the other 70% (1,220 images).
- Up to 15 submissions, at least 20 minutes apart. External labeled data was not allowed, in particular the four source collections, because they contain the hidden images.

## Why it matters

Analysts cannot inspect every square kilometer of coast by eye. A detector that marks and classifies hundreds of vessels in seconds lets them focus on the detections that matter, and the per-class thresholds we calibrate decide which detections are raised as alerts.

The solution was built with satellite deployment in mind:

- Every detector is a single-stage YOLO26 model from the compact end of the family. The YOLO26s model has about 9.5 million parameters and 20.8 GFLOPs at 640 pixels.
- A single model with one forward pass per image is a complete configuration on its own. The accuracy configuration adds models and views, and both can be adjusted to the compute available on board.
- Per-class calibration adds no compute: it only rescales confidences.
- The output is simple and standard: horizontal boxes, a class and a confidence per detection.
- The models were trained only on the challenge data, starting from public COCO weights.

## Our solution

The final submission is produced in five steps:

1. **Three detectors.** YOLO26m at 1280 pixels (trained, then trained again from its best weights), a YOLO26s at 1024 pixels trained on AWS, and a YOLO26s at 1024 pixels trained by Santiago Rodriguez on Kaggle.
2. **Eight-view test-time augmentation.** Each image is predicted in the eight orientations of a square: identity, horizontal flip, vertical flip, 180-degree rotation, 90 and 270-degree rotations, transpose and anti-transpose. The boxes are mapped back to the original image.
3. **Weighted Boxes Fusion (WBF).** The 24 prediction sets (3 models x 8 views) are fused per class with an IoU threshold of 0.70, averaged confidences and model weights of 1.5 for YOLO26m, 0.5 for the AWS YOLO26s and 1.5 for the Kaggle YOLO26s. The top 300 boxes per image are kept.
4. **Per-class calibration.** The F1-optimal threshold of each class is found on the test split, and each class's confidences are rescaled with a monotonic piecewise-linear map so that this threshold lands on one global threshold of 0.50. Ranking within each class is preserved, so mAP does not change while F1 improves.
5. **Official formatting and checks.** Files are written with the organizers' `formato_envio.py` and checked with its `verificar` command.

Why this design works for this data:

- The models are complementary. YOLO26m at 1280 found many more fishing vessels; the YOLO26s models fitted tighter boxes on small boats and warships. Fusing them put every class above every single model.
- Satellite images have no natural up direction, but the training pipeline never showed the models 90-degree rotations. Adding them at inference was the largest single gain after ensembling.
- WBF averages box coordinates from several predictions, which tightens boxes. That matters because 7 of the 10 IoU thresholds in the metric are 0.65 or stricter.

## How we trained it

**Data use.** All three models were trained on the same train split. The labeled test split served as validation during training and as the local yardstick for every decision. Because it was used to pick checkpoints and thresholds, local scores are slightly optimistic. The hidden split was only used to produce submissions.

**Audit and preprocessing.** We scanned all 8,470 images and 26,119 boxes before training. All images were valid JPEG RGB with no corrupt files, no empty images and no invalid labels, so no image was dropped. Hazy or bright images were kept. Box sizes decided the resolution: when the long side is resized to 640 pixels, half of the test small boats are under 16 pixels; at 1024 it is 10.2%, and at 1280 it is 2.9%. No offline preprocessing was applied; Ultralytics resizes and pads each image to the training size.

**Augmentation.** Horizontal and vertical flips (probability 0.5 each), mosaic (probability 1.0, switched off for the last epochs), HSV color jitter (0.015 / 0.7 / 0.4), translation 0.1 and scale 0.5. No rotation, shear or mixup.

**Optimization.** All four checkpoints record the AdamW optimizer with learning rate 0.001111, final learning rate factor 0.01, momentum 0.937, weight decay 0.0005 and a cosine learning rate schedule; for the AWS runs this was Ultralytics' automatic optimizer choice. All runs used mixed precision and seed 0, starting from public COCO weights or from our own checkpoints.

| Model | Start | Image size | Batch | Epochs | Training time | Other settings | Script |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26s (AWS) | yolo26s.pt (COCO) | 1024 | 16 | 34 | 57 min | Time limit 0.95 h, warmup 3 epochs, patience 15, mosaic off for the last 5 epochs | Organizers' `entrenar_baseline.py` |
| YOLO26m | yolo26m.pt (COCO) | 1280 | 6 | 11 | 51 min | Time limit 0.85 h, warmup 1 epoch, mosaic off for the last 2 epochs | `scripts/train.py` |
| YOLO26m, continued | Best weights of the run above | 1280 | 6 | 9 | 45 min | Time limit 0.75 h, same settings | `scripts/train.py` |
| YOLO26s (Kaggle, Santiago Rodriguez) | yolo26s.pt (COCO) | 1024 | 8 | Not recorded (30 configured, stopped by the time limit) | 2.25 h | Time limit 2.25 h, warmup 3 epochs, patience 15, mosaic off for the last 5 epochs, Ultralytics 8.4.205 | Kaggle notebook, not included in this repository |

**Hardware.** AWS EC2 g5.xlarge (one NVIDIA A10G with 24 GB, 4 vCPU), Ultralytics 8.4.203, PyTorch 2.14.1 (CUDA 12.6 build), Python 3.11. During training the GPU stayed near full utilization (84% to 100% in our samples). The Kaggle model was trained on Kaggle's hardware.

**Evaluation and selection.** Every number comes from the organizers' `evaluar.py` on the test split, including its threshold sweep (`--barrer-umbral`). Before each upload we also simulated the public leaderboard: 300 random 30% subsets of the test split, scored with the official evaluator functions, and we only uploaded a candidate that beat our current best in at least 95% of them. The simulation script is not included in this repository.

**Reproducing training.** Run from the repository root with the data in place (see Setup). With a time limit, the number of epochs depends on the hardware.

```bash
.venv/bin/python Entrega_Context/kit_participantes/entrenar_baseline.py --datos Entrega_Context/spacelab_dataset --modelo weights/yolo26s.pt --imgsz 1024 --horas 0.95 --batch 16 --workers 4 --proyecto runs/exp01 --nombre yolo26s_1024
.venv/bin/python scripts/train.py --model weights/yolo26m.pt --imgsz 1280 --hours 0.85 --batch 6 --workers 4 --project runs/exp03 --name yolo26m_1280
.venv/bin/python scripts/train.py --model runs/exp03/yolo26m_1280/weights/best.pt --imgsz 1280 --hours 0.75 --batch 6 --workers 4 --project runs/exp05 --name yolo26m_1280_ft
```

The Kaggle model's settings match the organizers' baseline script with `--modelo yolo26s.pt --imgsz 1024 --horas 2.25 --batch 8`.

## Journey of decisions

The steps below are in the order we took them. Scores are local test scores from `evaluar.py` unless marked as public.

1. **Pipeline first.** We ran the organizers' baseline on 5% of the training data for one epoch on a laptop. The score (0.0227) was meaningless by design: the point was to prove that training, evaluation, packaging and the official checker worked end to end. It also showed the laptop was far too slow for real training (our estimate: about 12 minutes per epoch at 640 pixels), so real training moved to a cloud GPU.
2. **Audit before training.** The box-size audit ruled out 640 pixels and set training at 1024 and 1280. It also showed that tiling was unnecessary (every image fits in one 1280 input) and that no image needed to be removed.
3. **YOLO26s at 1024 instead of the unchanged baseline.** With time for one real training run, we spent it on YOLO26s rather than the official YOLO26n baseline. Result: 0.4654. Warships were strong (AP 0.742) and fishing vessels were the bottleneck (AP 0.127, recall 0.27).
4. **Per-class calibration.** Each class gets its own F1-optimal threshold, folded into the single official threshold. Result: 0.4691, with mAP unchanged. Reviewing the code, we found an error in our own F1 formula before it touched a real result, and pinned the correct formula with a unit test. First upload: **0.4912 public**.
5. **Flip test-time augmentation, and a dead end.** Four flips fused with WBF at IoU 0.55 gave 0.4784, but warship AP fell from 0.742 to 0.693: the fusion merged neighboring warships moored in rows. Raising the fusion IoU to 0.70 fixed it (warship 0.765, score 0.4919); 0.80 gave nothing more (mAP 0.4427 against 0.4432). With calibration: 0.4956, and **0.5127 public**.
6. **A bigger model, and a lesson about noise.** YOLO26m at 1280, with a shorter warmup and fewer final epochs without mosaic to suit a short run, reached 0.4737 after 11 epochs (0.4789 calibrated). It found many more fishing vessels (AP 0.237 against 0.127) but was weaker on shiprsimagenet and vhrships images. Its upload scored **0.4868 public**, below the calibrated YOLO26s (0.4912) even though it was about 0.01 higher locally. This is when we built the leaderboard simulation; it gave that upload only a 76% chance of beating YOLO26s, so a local gain of 0.01 was not a reliable signal.
7. **Two-model ensemble, held back.** YOLO26m plus the AWS YOLO26s, each with flip TTA, scored 0.5321. Santiago's separate submission with his Kaggle model had reached **0.5263 public**, and against a local reproduction of his pipeline the ensemble won only 73% of the simulated draws, so we did not upload it.
8. **Continuing YOLO26m.** We trained YOLO26m again from its best weights. The restart first dropped validation mAP50-95 to 0.324 (a fresh learning-rate schedule with mosaic back on), then it recovered to 0.459, our best single model.
9. **Three-model ensemble.** We added the Kaggle YOLO26s, trained for 2.25 hours. On test, our fusion settings (four flips, averaged confidence) scored 0.5395, while the settings of Santiago's pipeline (rotation views, maximum confidence) scored 0.5315, because taking the maximum confidence across three models let more false alarms through. Calibrated: 0.5410. It won 97% of the simulated draws and scored **0.5476 public**.
10. **Eight views and model weights.** We cached the raw predictions of all three models in all eight views once, which made each fusion setting take about 2 seconds to score instead of a full prediction run. Eight views scored 0.5581 against 0.5395 for four flips. A grid of 35 settings showed that weighting the two stronger models (1.5 / 0.5 / 1.5) added about 0.003; we picked a setting in the middle of the flat region rather than the single top result. Calibrated: 0.5635. It won 100% of the simulated draws and scored **0.5658 public**, our best.

We also evaluated a public single-class vessel detector trained on 10-meter Sentinel-2 imagery. We did not use it, because it detects one class and was trained at a far lower resolution than this data.

**Decisions with the biggest impact**, measured on the local test score:

| Decision | Before | After | Gain |
| --- | --- | --- | --- |
| Ensembling YOLO26m with YOLO26s (both with flip TTA) | 0.4919 | 0.5276 | +0.036 |
| Flip TTA with WBF at IoU 0.70 (YOLO26s) | 0.4654 | 0.4919 | +0.027 |
| Eight views instead of four flips (three models) | 0.5395 | 0.5581 | +0.019 |
| Fusion IoU 0.70 instead of 0.55 (YOLO26s with TTA) | 0.4784 | 0.4919 | +0.014 |
| Third model and continued YOLO26m training | 0.5276 | 0.5395 | +0.012 |
| Per-class calibration (final pipeline) | 0.5615 | 0.5635 | +0.002 |

**Experiments and scores:**

| Experiment | Local score | mAP50-95 | F1 | Public score |
| --- | --- | --- | --- | --- |
| YOLO26s (AWS), 1024 | 0.4654 | 0.4112 | 0.5921 | - |
| YOLO26s (AWS), calibrated | 0.4691 | 0.4112 | 0.6044 | 0.4912 |
| YOLO26s (AWS), flip TTA, WBF 0.55 | 0.4784 | 0.4248 | 0.6037 | - |
| YOLO26s (AWS), flip TTA, WBF 0.70 | 0.4919 | 0.4432 | 0.6057 | - |
| YOLO26s (AWS), flip TTA, calibrated | 0.4956 | 0.4432 | 0.6180 | 0.5127 |
| YOLO26m, 1280 | 0.4737 | 0.4136 | 0.6137 | - |
| YOLO26m, calibrated | 0.4789 | 0.4136 | 0.6313 | 0.4868 |
| YOLO26m + YOLO26s (AWS), flip TTA, calibrated | 0.5321 | 0.4810 | 0.6509 | not uploaded |
| YOLO26s (Kaggle), Santiago's pipeline | about 0.528 (local reproduction) | - | - | 0.5263 |
| Three models, four flips, calibrated | 0.5410 | 0.4928 | 0.6536 | 0.5476 |
| Three models, eight views, weighted, calibrated | **0.5635** | **0.5125** | **0.6825** | **0.5658** |

Full details, including per-class results and the reason for each run, are in `experiments.md` and `submissions.md`.

## Results

Best submission: three models, eight views, weighted WBF and per-class calibration.

- Public leaderboard: **0.5658**. Final ranking on the private split: **3rd place**.
- Local test: score 0.5635, mAP@0.5:0.95 0.5125, mAP@0.5 0.6994, F1macro 0.6825 at threshold 0.50.

| Class | AP50-95 | AP50 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| small_boat | 0.406 | 0.687 | 0.663 | 0.709 | 0.685 |
| cargo_ship | 0.558 | 0.704 | 0.668 | 0.650 | 0.659 |
| fishing_vessel | 0.314 | 0.489 | 0.502 | 0.574 | 0.535 |
| warship | 0.776 | 0.904 | 0.902 | 0.835 | 0.867 |
| work_vessel | 0.508 | 0.713 | 0.658 | 0.674 | 0.666 |

| Source | Score | mAP50-95 | F1 |
| --- | --- | --- | --- |
| fair1m | 0.5248 | 0.4703 | 0.6521 |
| shiprsimagenet | 0.6079 | 0.5852 | 0.6609 |
| vhrships | 0.7651 | 0.7148 | 0.8824 |
| xview | 0.5158 | 0.4392 | 0.6945 |

**Against earlier attempts.** The public score rose from 0.4912 (first upload) to 0.5658 (+0.075). Locally, the final pipeline is +0.098 above our first trained model (0.4654).

**Baseline.** We did not run the official baseline unchanged (YOLO26n, 30 epochs). The closest reference is the same baseline script with YOLO26s at 1024, which scored 0.4654 locally. The single calibrated YOLO26s (0.4912 public) is also the one-model, one-pass configuration.

## Repository structure

```text
scripts/
  train.py               Time-limited Ultralytics training with short-run settings
  cache_predict.py       Predict every TTA view once per model and cache the raw boxes
  fuse_cached.py         Fuse cached predictions with WBF and score fusion settings
  tta_wbf.py             One-step TTA and multi-model WBF prediction
  calibrar_umbrales.py   Per-class threshold calibration and submission ZIP
  smoke_test_sahi.py     Environment smoke test (PyTorch, Ultralytics, SAHI, rasterio, GPU)
tests/                   Unit tests for box transforms and calibration
experiments.md           One row per experiment, with the reason for each
submissions.md           One row per leaderboard upload
refs/README.md           Index of the pinned upstream source used during development
requirements.txt         Pinned Python packages
pyproject.toml           Lint and test settings
```

Not in the repository: the dataset and the organizers' kit (expected in `Entrega_Context/`), model weights (`weights/`, `runs/`) and submissions.

## Setup

Requirements: Python 3.11 and `uv`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

On Linux with an NVIDIA GPU, install PyTorch from the CUDA 12.6 index first, because the default Linux build of torch 2.14.1 targets CUDA 13 and needs driver 580 or newer:

```bash
uv pip install --python .venv/bin/python torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r requirements.txt
```

Do not install `opencv-python-headless` next to `opencv-python`. On a minimal Ubuntu server, OpenCV also needs the system package `libgl1`.

Place the organizers' files as follows:

```text
Entrega_Context/
  kit_participantes/   evaluar.py, formato_envio.py, entrenar_baseline.py
  spacelab_dataset/    data.yaml, images/{train,test,oculto}, labels/{train,test}
```

COCO weights (`yolo26s.pt`, `yolo26m.pt`) download automatically on first use. The Kaggle YOLO26s weights are expected at `weights/team_s1024.pt`; they are not included in this repository and are available from the team on request.

Check the installation:

```bash
.venv/bin/python scripts/smoke_test_sahi.py
.venv/bin/pytest -q
```

## Usage

**Score any prediction folder or ZIP on the test split:**

```bash
.venv/bin/python Entrega_Context/kit_participantes/evaluar.py --gt Entrega_Context/spacelab_dataset/labels/test --envio <folder-or-zip> --barrer-umbral
```

**Single model, one pass per image** (predicts test and hidden, scores test and builds the submission ZIP), then per-class calibration:

```bash
.venv/bin/python Entrega_Context/kit_participantes/entrenar_baseline.py --datos Entrega_Context/spacelab_dataset --pesos runs/exp01/yolo26s_1024/weights/best.pt --imgsz 1024 --proyecto runs/exp01 --nombre yolo26s_1024
.venv/bin/python scripts/calibrar_umbrales.py --test-preds runs/exp01/yolo26s_1024/pred_test --oculto-preds runs/exp01/yolo26s_1024/envio --out runs/exp01/calib
```

**Reproduce the best submission:**

```bash
for split in test oculto; do
  src=Entrega_Context/spacelab_dataset/images/$split
  .venv/bin/python scripts/cache_predict.py --weights runs/exp05/yolo26m_1280_ft/weights/best.pt --imgsz 1280 --source $src --out runs/cache/m_$split.pkl
  .venv/bin/python scripts/cache_predict.py --weights runs/exp01/yolo26s_1024/weights/best.pt --imgsz 1024 --source $src --out runs/cache/s_$split.pkl
  .venv/bin/python scripts/cache_predict.py --weights weights/team_s1024.pt --imgsz 1024 --source $src --out runs/cache/k_$split.pkl
done
.venv/bin/python scripts/fuse_cached.py --test-caches runs/cache/m_test.pkl runs/cache/s_test.pkl runs/cache/k_test.pkl --oculto-caches runs/cache/m_oculto.pkl runs/cache/s_oculto.pkl runs/cache/k_oculto.pkl --configs d8:1.5,0.5,1.5:0.70:avg --write runs/fusion_best
.venv/bin/python scripts/calibrar_umbrales.py --test-preds runs/fusion_best/test --oculto-preds runs/fusion_best/oculto --out runs/fusion_best/calib
.venv/bin/python Entrega_Context/kit_participantes/formato_envio.py verificar --envio runs/fusion_best/calib/envio_calibrado.zip --imagenes Entrega_Context/spacelab_dataset/images/oculto
```

A fusion setting is written as `views:weights:iou:confidence`. View sets are `none`, `h`, `flips4`, `team4` and `d8`, the weights follow the order of the caches, and the confidence mode is `avg`, `max` or `box_and_model_avg`. Several settings can be passed to `--configs` to compare them on the test split; `--write` uses the first one.

## Team and credits

### Team DualCore

- Sergio Lopez: designed and directed the whole workflow.
- Santiago Rodriguez: trained the Kaggle YOLO26s model and built its inference pipeline, whose rotation views led us to add them to the final ensemble.
- Both members worked together on most of the project.

### Credits

- Organizers: Faculty of Engineering, Pontificia Universidad Javeriana, and the Colombian Aerospace Force (FAC), GeoVision Colombia.
- Data: FAIR1M, ShipRSImageNet, VHRShips and xView, compiled and relabeled by the organizers.
- Software: Ultralytics YOLO26, PyTorch, ensemble-boxes (Weighted Boxes Fusion) and SAHI (environment smoke test only).
- Compute: AWS EC2 g5.xlarge and Kaggle.
