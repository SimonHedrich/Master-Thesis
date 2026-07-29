"""Regenerate the 225-class teacher soft-label cache from a fine-tuned
SpeciesNet checkpoint (Goal C output).

Run once, offline, inside `Dockerfile.speciesnet` (`make speciesnet-shell` /
`speciesnet-start`) after `run_finetune.py` has produced a `best.pt`. Output
feeds Goal B's KD training (`scripts/training/yolo26n --kd`) as a precomputed,
frozen signal — the ~54M-param teacher classifier never runs inside the
student's training loop. See
docs/plans/2026-06-30_yolo26-kd-and-teacher-finetune-implementation-plan.md §3.2.

Reuses Goal C's real components directly (no reimplementation):
  - `teacher_model.speciesnet_model()` to build the classifier
  - `dataset.SpeciesNetCropDataset` (annotation-indexed — one row per
    detection, matching this cache's per-annotation `detection_idx` schema)
  - `taxonomy.projection_tables()` + `taxonomy._load_script7().compute_probs_225()`
    for the 225-class projection, identical to `evaluate.py`'s call pattern.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import scripts.training.teacher_finetune.constants as constants
from scripts.training.teacher_finetune.dataset import SpeciesNetCropDataset
from scripts.training.teacher_finetune.taxonomy import _load_script7, projection_tables
from scripts.training.teacher_finetune.teacher_model import _check_environment, speciesnet_model

logger = logging.getLogger(__name__)

_SPLIT_ANNOTATIONS = {
    "train": constants.ANNOTATIONS_TRAIN,
    "val": constants.ANNOTATIONS_VAL,
    "test": constants.ANNOTATIONS_TEST,
}


class _FileKeyedDataset(Dataset):
    """Composition wrapper: pairs `SpeciesNetCropDataset`'s preprocessed crop
    with its source `file_name`, without touching the base `__getitem__`/
    `collate_fn` contract other call sites (training_pipeline.py, evaluate.py)
    rely on.
    """

    def __init__(self, base: SpeciesNetCropDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        crop, _idx_225 = self.base[idx]
        file_name = self.base.samples[idx][0]
        return crop, file_name


def _collate(batch: list):
    crops, file_names = zip(*batch)
    return torch.stack(crops), list(file_names)


def cache_soft_labels(
    split: str,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = constants.BATCH_SIZE,
    num_workers: int = constants.NUM_WORKERS,
) -> None:
    _check_environment()
    model, preprocess_fn, _labels = speciesnet_model(device, freeze_fraction=0.0)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(
        "loaded fine-tuned teacher checkpoint: %s (epoch=%s)", checkpoint_path, ckpt.get("epoch")
    )

    base_ds = SpeciesNetCropDataset(_SPLIT_ANNOTATIONS[split], constants.IMAGE_ROOT, preprocess_fn)
    ds = _FileKeyedDataset(base_ds)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate
    )

    s7 = _load_script7()
    proj_tables = projection_tables()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    detection_counters: dict[str, int] = {}
    n_written = 0
    with output_path.open("w") as f, torch.no_grad():
        pbar = tqdm(loader, desc=f"cache teacher soft labels ({split})", unit="batch")
        for crops, file_names in pbar:
            crops = crops.to(device)
            logits = model(crops)
            probs = torch.softmax(logits, dim=-1).cpu()
            for i, file_name in enumerate(file_names):
                probs_225, prob_sum = s7.compute_probs_225(probs[i].tolist(), *proj_tables)
                det_idx = detection_counters.get(file_name, 0)
                detection_counters[file_name] = det_idx + 1
                f.write(
                    json.dumps(
                        {
                            "filepath": file_name,
                            "detection_idx": det_idx,
                            "probs_225": probs_225,
                            "prob_225_sum": prob_sum,
                        }
                    )
                    + "\n"
                )
                n_written += 1
    logger.info(
        "wrote %d soft-label records (%d unique images) to %s",
        n_written,
        len(detection_counters),
        output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to teacher_finetune's latest_run_dir()/best.pt",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=constants.NUM_WORKERS)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        run_dir = constants.latest_run_dir()
        if run_dir is None:
            raise SystemExit(
                "no teacher_finetune run found under "
                f"{constants.OUTPUT_DIR} — run run_finetune.py first, or pass --checkpoint"
            )
        checkpoint = run_dir / "best.pt"

    output = args.output or (constants.DATA_ROOT / "real" / f"teacher_soft_labels_{args.split}.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_soft_labels(args.split, checkpoint, output, device, args.batch_size, args.num_workers)
