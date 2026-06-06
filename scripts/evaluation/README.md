# Evaluation scripts

Qualitative evaluation tooling for trained student models. Complements the
`torchmetrics`-based mAP that `scripts/training/yolov5s/evaluation.py` reports
during training by giving an interactive view of predictions vs. ground truth
in FiftyOne.

## Files

| File | Purpose |
|---|---|
| `run_inference.py` | Sample N images from `data/real/annotations_test.json`, run YOLOv5s inference, write a COCO-format GT subset + a COCO-results-format predictions JSON. |
| `visualize_fiftyone.py` | Load both JSONs into a FiftyOne dataset and launch the app. |
| `outputs/` | Run outputs (gitignored). |

## Typical workflow

```bash
# 1. Run inference on 100 deterministic test images (seed pinned in source).
python -m scripts.evaluation.run_inference

# 2. Launch FiftyOne to inspect predictions vs ground truth.
python -m scripts.evaluation.visualize_fiftyone
# local:     http://localhost:5155
# tailscale: http://gpu-server.taile550ef.ts.net:5155
```

Outputs land in `scripts/evaluation/outputs/`:
- `annotations_subset.json` — COCO JSON with the 100 sampled images + their GT annotations + the full 225-class `categories` list.
- `predictions.json` — flat COCO results array (`image_id`, `category_id`, `bbox=[x,y,w,h]`, `score`).

## CLI

`run_inference.py` defaults all paths and thresholds to the values in
`scripts/training/yolov5s/constants.py`. Override per-invocation:

```bash
python -m scripts.evaluation.run_inference \
    --weights scripts/training/yolov5s/model_exports/best.pt \
    --num-images 100 \
    --conf-thres 0.001 \
    --device cuda
```

`visualize_fiftyone.py` reads the two JSONs the inference script wrote.
Override to view other runs:

```bash
python -m scripts.evaluation.visualize_fiftyone \
    --annotations scripts/evaluation/outputs/annotations_subset.json \
    --predictions scripts/evaluation/outputs/predictions.json \
    --port 5155
```

## Notes

- Confidence threshold defaults to `EVAL_CONF_THRES = 0.001` so the predictions JSON is lossless for mAP-style analysis. Use the FiftyOne sidebar filter to hide low-confidence boxes interactively.
- The 100-image sample is deterministic (`SEED = 42`, pinned at module scope in `run_inference.py`) so re-running against a new checkpoint produces a directly comparable view.
