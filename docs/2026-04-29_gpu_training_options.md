# GPU Options for Training: 48 GB vs 2 × 12 GB

Analysis of the two available NVIDIA A40 configurations for the KD training pipeline.

---

## Training Workload Summary

The thesis requires the following training runs:

**Teacher-level runs (1–3 total):**
- YOLOv8s fine-tune on Tier A+B+C data (primary teacher)
- RT-DETR-L fine-tune (optional higher-accuracy teacher)
- Teacher-assistant fine-tune via KD (capacity-gap mitigation; YOLOv12-m scale)

**Student-level runs (~24–28 total):**
- Three student architectures: YOLO11n / YOLO12n, NanoDet-Plus-m, PicoDet-S
- Three tiers × four conditions (Baseline-Direct, Synth-Direct, KD-Real, KD-Synth), with some conditions not applicable to all tiers
- 300 epochs per run, validated every 10 epochs

**Inference-only (not trained):**
- SpeciesNet (MegaDetector v5a + EfficientNetV2-M): runs once to generate soft-label JSONL for the entire training set; no gradient computation

The student runs are the bottleneck. With 24–28 runs at ~300 epochs each, total training time at ~2–4 hours/run amounts to roughly **2–5 days of continuous compute** if run sequentially.

---

## VRAM Requirements per Model

| Model | Role | Config | VRAM (est.) | Fits 12 GB? |
|---|---|---|---|---|
| NanoDet-Plus-m | Student | bs=64, 416×416 | 6–7 GB | Yes |
| PicoDet-S | Student | bs=96, 320×320 | 7–8 GB | Yes |
| YOLO11n / YOLO12n | Student | bs=32, 640×640 | 7–8 GB | Yes |
| YOLOv8s | Teacher fine-tune | bs=16, 640×640 | 9–10 GB | Yes |
| SpeciesNet inference | Soft-label gen | batch=8 | 4–6 GB | Yes |
| RT-DETR-L | Optional teacher | bs=8, 640×640 | 12–15 GB | Borderline (bs=4–6) |
| YOLOv12-m | Teacher-assistant | bs=8, 640×640 | 12–14 GB | Borderline (bs=4) |

> Memory estimates include model weights + gradients + Adam optimizer states (2× param bytes) + activations. Actual usage varies with PyTorch version and cudnn workspace allocation.

---

## A40 Hardware Context

The NVIDIA A40 is a datacenter GPU based on the Ampere architecture:

| Spec | Value |
|---|---|
| Total VRAM | 48 GB GDDR6 |
| Memory bandwidth | 696 GB/s |
| FP32 compute | ~37.4 TFLOPS |
| CUDA cores | 10,752 |
| TDP | 300 W |

The virtual 4-way split divides the 48 GB into four 12 GB slices. Compute is **shared dynamically**: each instance gets a guaranteed minimum of roughly ¼ of total throughput (~9.4 TFLOPS), but can scale toward 100% when other instances are idle. This makes peak training speed unpredictable — it depends on the load from co-tenants.

---

## Option A: Single 48 GB Instance

### Pros

- **All models fit without batch size compromise.** RT-DETR-L and YOLOv12-m can train at bs=16 or higher, improving gradient quality and convergence stability.
- **Full or near-full compute allocation.** As the sole user of the instance, throughput is dedicated, making per-run training time more predictable.
- **Simpler scheduling.** Only one SSH session, one Docker container, one job queue to manage.
- **Future-proofing.** If the experimental design is later extended to include ViT-based or 100M+ param teachers, they will fit without revisiting the infrastructure.

### Cons

- **Sequential student training.** 24–28 runs must be queued one after another. Total wall-clock time is the sum of all individual run times.
- **36 GB wasted for most runs.** Student models use only 7–10 GB; the remaining 38–41 GB sits idle.
- **Harder to justify.** The 48 GB tier typically requires a stronger business case and may face longer approval time.
- **Single point of failure.** An instance crash or resource preemption pauses the entire training campaign.

---

## Option B: Two 12 GB Instances

### Pros

- **Parallel student training.** Two runs execute simultaneously, roughly halving wall-clock time for the student phase (24–28 runs become 12–14 parallel pairs).
- **Natural experiment isolation.** Each instance runs a self-contained Docker container with its own dataset mount and log directory; no risk of jobs interfering with each other.
- **Easier approval path.** Two standard-tier instances are a routine request.
- **Resilience.** A crash on one instance does not interrupt the other.
- **Sufficient for all student and primary teacher runs.** YOLOv8s, all YOLO-nano variants, NanoDet, and PicoDet all fit within 12 GB at their configured batch sizes.

### Cons

- **Variable compute throughput.** Per-instance speed fluctuates between ¼ and full A40 performance depending on co-tenant activity. Training time per run is less predictable.
- **RT-DETR-L and YOLOv12-m are borderline.** These models require reducing batch size to bs=4–6 to fit in 12 GB, which may affect convergence speed. The teacher-assistant cascade should be tested at bs=4 before committing to 300 epochs.
- **Two environments to maintain.** SSH keys, Docker images, dataset syncing, and monitoring must be set up for two machines.
- **No headroom for unexpected memory growth.** Framework updates, larger augmentation pipelines, or switching to bf16 training can push usage above 12 GB without warning.

---

## Recommendation: Two 12 GB Instances

**The bottleneck is the number of student training runs, not the size of any individual model.**

All 24–28 student runs fit comfortably in 12 GB. Running two jobs in parallel cuts total wall-clock time from roughly 4–5 days to 2–2.5 days — a significant gain for a thesis timeline. The primary teacher (YOLOv8s at 11M params) also fits within 12 GB at the configured batch size.

The only borderline cases are RT-DETR-L and the teacher-assistant (YOLOv12-m). Both can be accommodated:

- **RT-DETR-L:** Reduce batch size to bs=4 (640×640). This is viable for fine-tuning a pretrained model where gradient variance matters less than for training from scratch. If 12 GB proves insufficient even at bs=4, fall back to YOLOv8s as the sole teacher — the experimental design supports this.
- **YOLOv12-m teacher-assistant:** The cascade is a mitigation strategy, not a core requirement. It can be deferred until teacher and direct-student runs are complete, then scheduled as a single dedicated run on one instance.

The 48 GB instance is over-specified for this workload. The extra memory would remain idle for ~95% of total training time. The approval overhead and sequential training constraint are not worth the headroom when all core runs fit in 12 GB.

---

## Practical Notes

### Suggested Division of Labor

| Instance | Assigned work |
|---|---|
| GPU-A | Teacher fine-tuning (YOLOv8s, then optional RT-DETR-L); teacher-assistant KD run |
| GPU-B | Student runs (start immediately once teacher soft-labels are ready for first tier) |

Soft-label generation via SpeciesNet can run on GPU-B before student training starts, since it requires only ~5 GB.

### Batch Size Fallbacks for Borderline Models

If a 12 GB run fails with CUDA OOM:

```
# NanoDet: reduce from bs=64 → bs=32
# PicoDet: reduce from bs=96 → bs=64
# RT-DETR-L: reduce from bs=8 → bs=4, increase grad_accumulate=2
# YOLOv12-m: bs=4, gradient_checkpointing=True (if supported)
```

Gradient accumulation can recover effective batch size at the cost of slightly higher per-step overhead.

### Docker Memory Flags

Each container should be started with:
```bash
--shm-size=8g        # prevents DataLoader shared-memory exhaustion
--gpus device=<id>   # pin to one GPU; avoids accidental cross-instance allocation
```

### Monitoring

Run `nvidia-smi dmon -s u` on each instance to track live VRAM and utilisation. For long runs (300 epochs), use `wandb` or TensorBoard with a remote log sync — do not rely on stdout-only logging when you cannot keep a terminal open.
