# Reference source code

Read-only upstream source, pinned to the versions in `requirements.txt`. Read and grep it to confirm argument names and defaults. Never edit it or import from it.

The clones are not tracked by git. Recreate them from the repo root:

```bash
git -c advice.detachedHead=false clone --depth 1 --branch 0.14.0 https://github.com/obss/sahi.git refs/sahi
git -c advice.detachedHead=false clone --depth 1 --branch v8.4.203 https://github.com/ultralytics/ultralytics.git refs/ultralytics
```

## SAHI 0.14.0

| Path | Use it for |
| --- | --- |
| `sahi/sahi/predict.py` | `get_prediction`, `get_sliced_prediction` |
| `sahi/sahi/slicing.py` | `slice_image`, `slice_coco` |
| `sahi/sahi/models/ultralytics.py` | Ultralytics wrapper, OBB polygons in `prediction.mask.segmentation` |
| `sahi/sahi/postprocess/` | NMS and NMM merging of sliced predictions |
| `sahi/sahi/utils/coco.py` | COCO dataset utilities and format conversion |
| `sahi/docs/cli.md` | Command line usage |
| `sahi/demo/inference_for_ultralytics.ipynb` | Worked Ultralytics example |

## Ultralytics 8.4.203

| Path | Use it for |
| --- | --- |
| `ultralytics/ultralytics/cfg/default.yaml` | Training and prediction arguments and defaults |
| `ultralytics/docs/en/tasks/obb.md` | Oriented bounding box task |
| `ultralytics/docs/en/datasets/obb/` | OBB dataset formats |
| `ultralytics/ultralytics/cfg/datasets/DOTAv1.yaml` | DOTA class names (class 1 is `ship`) |
| `ultralytics/docs/en/guides/sahi-tiled-inference.md` | SAHI tiled inference guide |
| `ultralytics/docs/en/guides/model-training-tips.md` | Training tips |
| `ultralytics/docs/en/guides/yolo-performance-metrics.md` | Metric definitions |
| `ultralytics/docs/en/guides/streamlit-live-inference.md` | Streamlit inference guide |
