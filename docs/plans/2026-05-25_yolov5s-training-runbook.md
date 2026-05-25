# YOLOv5s Training Runbook

**Date:** 2026-05-25  
**Model:** YOLOv5s (7.2M params) — direct fine-tuning baseline  
**Dataset:** Wildlife 225-class, 154k train / 15k val / 75k test  
**Repo:** `/home/debian/Master-Thesis`

---

## Prerequisites

- Docker with NVIDIA runtime installed
- GPU available (tested on A40 48GB)
- Dataset splits complete (`data/real/annotations_*.json` and `data/synthetic/annotations_*.json` all exist)

---

## Step 1 — Clone and pin YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git /opt/yolov5
cd /opt/yolov5
git checkout 5cdad89
```

> Commits after `5cdad89` require an additional commercial license. Always use this exact commit.

---

## Step 2 — Patch `train.py` for AdamW

Open `/opt/yolov5/train.py` and find the optimizer block (search for `torch.optim`). Replace it with:

```python
optimizer = torch.optim.AdamW(pg0, lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
optimizer.add_param_group({'params': pg2})
```

If the commit already supports `--optimizer AdamW` natively (check `train.py --help`), skip the patch and add `--optimizer AdamW` to the training command in the Makefile instead.

---

## Step 3 — Build the Docker image

```bash
cd /home/debian/Master-Thesis
make docker-yolov5-build
```

This builds `wildlife-yolov5` from `Dockerfile.yolov5` (PyTorch 2.0.1 + CUDA 11.8).

---

## Step 4 — Prepare the dataset

Run once before any training:

```bash
make yolov5-prepare
```

This runs `scripts/training/1-prepare_yolov5_dataset.py`, which:
- Merges `data/real/annotations_train.json` + `data/synthetic/annotations_train.json`
- Converts COCO `[x, y, w, h]` bboxes to YOLO normalised `[cx, cy, w, h]`
- Writes label `.txt` files and image symlinks to `data/training/yolov5/`
- Emits `data/training/wildlife225_yolov5.yaml`

Takes ~5 minutes (mostly symlink I/O for 154k images).

**Verify the output:**

```bash
ls data/training/yolov5/images/train/ | wc -l   # expect ~154234
ls data/training/yolov5/labels/train/ | wc -l   # expect same count
cat data/training/wildlife225_yolov5.yaml        # check nc: 225, paths look correct
```

---

## Step 5 — Start training (seed 42)

```bash
make yolov5-train
# equivalent to: make yolov5-train YV5_SEED=42
```

Training runs detached in a Docker container. Follow progress:

```bash
docker logs -f wildlife-yolov5-run_seed42
```

Or watch the results CSV directly (updates each epoch):

```bash
tail -f output/yolov5_wildlife/yolov5s_wildlife225_seed42/results.csv
```

---

## Step 6 — Check loss balance at epoch 5

After epoch 5, open `output/yolov5_wildlife/yolov5s_wildlife225_seed42/results.csv` and compare the loss columns.

**If `cls_loss > 2 × (box_loss + obj_loss)`:** the classification head is dominating. Stop training, reduce `cls` from `0.5` to `0.3` in `scripts/training/configs/hyp.finetune-wildlife.yaml`, then restart.

This is caused by YOLOv5 internally scaling `cls` by `nc/80 = 225/80 ≈ 2.8`. The current value of `0.5` gives an effective weight of ~1.41, which is usually fine, but check anyway.

---

## Step 7 — Replicate with seeds 1 and 7

Once seed 42 looks healthy, start the other two seeds (sequentially or in parallel on separate GPUs):

```bash
make yolov5-train YV5_SEED=1
make yolov5-train YV5_SEED=7
```

---

## Step 8 — Evaluate on the test split

Run after each training completes:

```bash
make yolov5-eval YV5_SEED=42
make yolov5-eval YV5_SEED=1
make yolov5-eval YV5_SEED=7
```

Results are written to `output/yolov5_wildlife/yolov5s_wildlife225_seed<N>_eval/`.

---

## Step 9 — Per-band analysis

```bash
python3 scripts/training/eval_per_band.py \
  --predictions output/yolov5_wildlife/yolov5s_wildlife225_seed42_eval/predictions.json \
  --split test \
  --out reports/yolov5s_per_band_seed42.json
```

Repeat for seeds 1 and 7. The output table shows mAP@0.5 and mAP@0.5:0.95 broken down by band:

| Band | Classes | Data source | What it measures |
|------|---------|-------------|------------------|
| A | 51 | Synthetic train only | Synthetic-to-real transfer gap |
| B | 26 | Mixed (real + synthetic) | Mixed-data performance |
| C | 26 | Real train only (sparse) | Moderate-data baseline |
| D | 122 | Real train (capped at 1500/class) | Data-rich baseline |

---

## Expected timeline (A40 GPU)

| Step | Time |
|------|------|
| Dataset prep (Step 4) | ~5 min |
| Training per seed (Steps 5–7) | ~3–5h (150 epochs, early stop ~ep 80–120) |
| Three seeds total | ~10–15h sequential |
| Evaluation per seed (Step 8) | ~15 min |

---

## Troubleshooting

**`CUDA out of memory` during training:**  
Halve the batch size and double the accumulation steps:
```bash
# In docker-yolov5-shell: python train.py ... --batch-size 32 --accumulate 2
```

**`No such file or directory` for image symlinks:**  
A source image is missing. Check `data/training/yolov5/prep_stats.json` for `missing_source` count. Re-run `make yolov5-prepare` after fixing the missing files.

**Early stopping triggers before epoch 50:**  
Patience is set to 50 epochs. If it triggers earlier, the model converged quickly — check that val mAP is reasonable before concluding.

**`model.add_callback` AttributeError in MLflow callback:**  
The callback API may not exist at commit `5cdad89`. Comment out the `register_mlflow_callbacks` call and use YOLOv5's built-in CSV logger (`results.csv`) for now; import metrics into MLflow manually after training.
