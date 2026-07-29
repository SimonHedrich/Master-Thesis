"""Evaluation loop and MLflow logging helper for the SpeciesNet teacher fine-tune.

Projects the classifier's native 2,498-way softmax onto the project's
225-class vector via ``compute_probs_225`` (reused from
``scripts/dataset_quality/7-filter_speciesnet.py`` — same function the dataset
pipeline already uses), then scores top-1 accuracy and macro/micro F1 against
ground truth. Per-source breakdown is computed directly via a groupby on each
sample's ``source`` field (already present in ``data/real/annotations_*.json``'s
``images`` list) — **not** script 8's dataset-composition-tiering machinery,
which serves a different purpose (estimating trusted-label pool sizes for
dataset-build decisions, not scoring model predictions against ground truth).

**Circularity caveat** (parent strategy doc §1.1): part of the test set's
OpenImages/ImagesCV portion was originally filtered using the
*pre-fine-tuning* SpeciesNet, so that portion's contribution to any reported
accuracy improvement is optimistic by an unknown (likely small) amount. This
is disclosed via the per-source breakdown reported here, not engineered around
by carving out a bespoke test slice — every model in the comparison matrix is
scored on the same fixed test set, per the strategy doc's single-fixed-
instrument principle.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm

import scripts.training.teacher_finetune.constants as constants

logger = logging.getLogger(__name__)

_SCRIPT7_PATH = (
    Path(__file__).resolve().parents[2] / "dataset_quality" / "7-filter_speciesnet.py"
)


def _load_script7():
    """Same importlib pattern as `taxonomy.py` / `8-class_distribution_report.py`."""
    spec = importlib.util.spec_from_file_location("filter_speciesnet", _SCRIPT7_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["filter_speciesnet"] = mod
    spec.loader.exec_module(mod)
    return mod


def _per_source_breakdown(
    preds: list[int], targets: list[int], sources: list[str]
) -> dict[str, dict]:
    by_source: dict[str, list[bool]] = defaultdict(list)
    for p, t, s in zip(preds, targets, sources):
        by_source[s].append(p == t)
    return {
        source: {"n": len(correct), "accuracy_top1": sum(correct) / len(correct)}
        for source, correct in sorted(by_source.items())
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    idx_to_label: dict[int, str],
    genus_species_to_225: dict[str, int],
    genus_to_225: dict[str, int],
    family_to_225: dict[str, int],
) -> dict:
    model.eval()
    s7 = _load_script7()

    dataset = data_loader.dataset
    sources = getattr(dataset, "sources", None)

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_sources: list[str] = []

    offset = 0
    logger.info("evaluating on %d batches", len(data_loader))
    pbar = tqdm(data_loader, desc="eval", unit="batch", leave=False, dynamic_ncols=True)
    for arrs, labels in pbar:
        arrs = arrs.to(device)
        logits = model(arrs)
        probs = torch.softmax(logits, dim=-1).cpu()

        for i in range(probs.shape[0]):
            probs_225, _prob_sum = s7.compute_probs_225(
                probs[i].tolist(),
                idx_to_label,
                genus_species_to_225,
                genus_to_225,
                family_to_225,
            )
            pred_225 = max(range(constants.NUM_CLASSES_225), key=lambda k: probs_225[k])
            all_preds.append(pred_225)
            all_targets.append(int(labels[i].item()))
            if sources is not None and offset + i < len(sources):
                all_sources.append(sources[offset + i])
            else:
                all_sources.append("unknown")

        offset += probs.shape[0]

    n = max(len(all_targets), 1)
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    accuracy_top1 = correct / n

    preds_t = torch.tensor(all_preds, dtype=torch.long)
    targets_t = torch.tensor(all_targets, dtype=torch.long)
    f1_macro = MulticlassF1Score(num_classes=constants.NUM_CLASSES_225, average="macro")(
        preds_t, targets_t
    ).item()
    f1_micro = MulticlassF1Score(num_classes=constants.NUM_CLASSES_225, average="micro")(
        preds_t, targets_t
    ).item()

    per_source = _per_source_breakdown(all_preds, all_targets, all_sources)

    return {
        "accuracy_top1": accuracy_top1,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "per_source": per_source,
        "n_samples": len(all_targets),
    }


def eval_log_mlflow(result: dict, prefix: str, step: int | None = None) -> None:
    mlflow.log_metric(f"{prefix}/accuracy_top1", result["accuracy_top1"], step=step)
    mlflow.log_metric(f"{prefix}/f1_macro", result["f1_macro"], step=step)
    mlflow.log_metric(f"{prefix}/f1_micro", result["f1_micro"], step=step)
    logger.info(
        "mlflow: logged %s/accuracy_top1=%.4f %s/f1_macro=%.4f %s/f1_micro=%.4f (step=%s, n=%d)",
        prefix,
        result["accuracy_top1"],
        prefix,
        result["f1_macro"],
        prefix,
        result["f1_micro"],
        step,
        result["n_samples"],
    )

    per_source = result.get("per_source")
    if per_source:
        rows = [
            {"source": s, "n": v["n"], "accuracy_top1": v["accuracy_top1"]}
            for s, v in per_source.items()
        ]
        # Deliberately log_artifact, NOT mlflow.log_table: log_table appends an
        # entry per file to the run's `mlflow.loggedArtifacts` tag, which is
        # capped at 8000 chars server-side — one table per epoch overflows and
        # silently corrupts the tag after ~137 epochs, killing the run (see
        # docs/progress_notes/2026-07-13_mlflow-log-table-crash-and-resume.md).
        # Best-effort: a logging failure must never abort a multi-day training run.
        try:
            with tempfile.TemporaryDirectory() as tmp:
                name = f"{prefix}_per_source_step{step if step is not None else 'final'}.json"
                table_path = Path(tmp) / name
                table_path.write_text(json.dumps(rows, indent=1))
                mlflow.log_artifact(str(table_path), artifact_path="per_source")
            logger.info("mlflow: logged %s per-source accuracy table (%d sources)", prefix, len(rows))
        except Exception:
            logger.exception("mlflow: failed to log %s per-source accuracy table — continuing", prefix)
