# Wildlife 225-Class Detection — YOLOv5s Training

Fine-tuning YOLOv5s on the 225-class wildlife dataset (non-bird mammals). This covers the direct training path without Docker.

See `thesis/` for the Master's thesis manuscript and writing-reference materials (including an analysis of the author's bachelor thesis structure and style, used as a template).

## Prerequisites

- Python 3.10
- YOLOv5 source pinned at commit `5cdad89` in `/opt/yolov5`
- CUDA-capable GPU
Manual steps to run to start training:
# On host — clone YOLOv5 (once)
git clone https://github.com/ultralytics/yolov5.git /opt/yolov5
cd /opt/yolov5 && git checkout 5cdad89

# Build the image and open a shell
make build TARGET=yolov5 IMAGE=wildlife-yolov5
make run IMAGE=wildlife-yolov5

# Inside the container — check if --optimizer flag exists
python /opt/yolov5/train.py --help | grep optimizer
# If yes: add --optimizer AdamW to scripts/training/Makefile's yolov5-train target
# If no: patch train.py per the runbook (Step 2 in docs/plans/2026-05-25_yolov5s-training-runbook.md)

# Prepare dataset (once), then train
make -f /app/scripts/training/Makefile yolov5-prepare
make -f /app/scripts/training/Makefile yolov5-train


**Set the repo root once:**
```bash
export REPO=/home/ubuntu/Master-Thesis
```

**Install YOLOv5 dependencies** (run once):
```bash
pip3 install \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
pip3 install matplotlib numpy "opencv-python-headless>=4.1.2" Pillow \
  PyYAML requests scipy tqdm seaborn pandas thop mlflow
```

---

## Step 1 — Patch train.py for AdamW

Check whether `--optimizer` is available at this commit:
```bash
python3 /opt/yolov5/train.py --help | grep optimizer
```

If the flag is **not listed**, open `/opt/yolov5/train.py`, find the optimizer block (search for `torch.optim`), and replace it with:
```python
optimizer = torch.optim.AdamW(pg0, lr=hyp['lr0'], weight_decay=hyp['weight_decay'])
optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
optimizer.add_param_group({'params': pg2})
```

If the flag **is listed**, you can skip the patch and add `--optimizer AdamW` to the training command instead.

---

## Step 2 — Prepare the dataset

Run once before any training. Converts COCO annotations to YOLO TXT format and creates image symlinks.

```bash
python3 $REPO/scripts/training/1-prepare_yolov5_dataset.py
```

This reads `data/real/annotations_{train,val,test}.json` (and `data/synthetic/` if present), writes label `.txt` files and image symlinks to `data/training/yolov5/`, and emits `data/training/wildlife225_yolov5.yaml`.

**Verify the output:**
```bash
ls $REPO/data/training/yolov5/images/train/ | wc -l   # expect ~145k
cat $REPO/data/training/wildlife225_yolov5.yaml        # check nc: 225
```

---

## Step 3 — Train (seed 42)

```bash
mkdir -p $REPO/output

cd /opt/yolov5 && nohup python3 train.py \
  --weights yolov5s.pt \
  --cfg models/yolov5s.yaml \
  --data $REPO/data/training/wildlife225_yolov5.yaml \
  --hyp $REPO/scripts/training/configs/hyp.finetune-wildlife.yaml \
  --epochs 150 --batch-size 64 --imgsz 640 \
  --patience 50 --cos-lr --multi-scale --label-smoothing 0.1 \
  --project $REPO/output/yolov5_wildlife \
  --name yolov5s_wildlife225_seed42 \
  --workers 8 --seed 42 \
  > $REPO/output/yolov5_seed42.log 2>&1 &

tail -f $REPO/output/yolov5_seed42.log
```

Or watch the per-epoch results CSV directly:
```bash
tail -f $REPO/output/yolov5_wildlife/yolov5s_wildlife225_seed42/results.csv
```

**Check loss balance after epoch 5.** Open `results.csv` and compare columns. If `cls_loss > 2 × (box_loss + obj_loss)`, reduce `cls` from `0.5` to `0.3` in `scripts/training/configs/hyp.finetune-wildlife.yaml` and restart.

---

## Step 4 — Replicate with seeds 1 and 7

```bash
cd /opt/yolov5 && nohup python3 train.py \
  --weights yolov5s.pt --cfg models/yolov5s.yaml \
  --data $REPO/data/training/wildlife225_yolov5.yaml \
  --hyp $REPO/scripts/training/configs/hyp.finetune-wildlife.yaml \
  --epochs 150 --batch-size 64 --imgsz 640 \
  --patience 50 --cos-lr --multi-scale --label-smoothing 0.1 \
  --project $REPO/output/yolov5_wildlife \
  --name yolov5s_wildlife225_seed1 \
  --workers 8 --seed 1 \
  > $REPO/output/yolov5_seed1.log 2>&1 &
```

Replace `seed1` / `--seed 1` with `seed7` / `--seed 7` for the third run.

---

## Step 5 — Evaluate on the test split

```bash
cd /opt/yolov5 && python3 val.py \
  --weights $REPO/output/yolov5_wildlife/yolov5s_wildlife225_seed42/weights/best.pt \
  --data $REPO/data/training/wildlife225_yolov5.yaml \
  --imgsz 640 --batch-size 32 --task test --verbose --save-json \
  --project $REPO/output/yolov5_wildlife \
  --name yolov5s_wildlife225_seed42_eval
```

Results are written to `output/yolov5_wildlife/yolov5s_wildlife225_seed42_eval/`.

---

## Step 6 — Per-band analysis

```bash
python3 $REPO/scripts/training/eval_per_band.py \
  --predictions $REPO/output/yolov5_wildlife/yolov5s_wildlife225_seed42_eval/predictions.json \
  --split test \
  --out $REPO/reports/yolov5s_per_band_seed42.json
```

| Band | Classes | Source | Measures |
|------|---------|--------|---------|
| A | 51 | Synthetic train only | Synthetic-to-real transfer gap |
| B | 26 | Mixed (real + synthetic) | Mixed-data performance |
| C | 26 | Real only (sparse) | Moderate-data baseline |
| D | 122 | Real (capped at 1500/class) | Data-rich baseline |

---

## Troubleshooting

**CUDA out of memory:** halve the batch size:
```bash
--batch-size 32
```

**Missing image symlinks:** check `data/training/yolov5/prep_stats.json` for `missing_source` count, fix the missing files, then re-run the prepare step.

**Early stopping before epoch 50:** patience is 50 epochs — if it triggers, check that val mAP is reasonable before treating it as a problem.

**`model.add_callback` AttributeError (MLflow):** comment out the `register_mlflow_callbacks` call; use `results.csv` for metrics instead.
