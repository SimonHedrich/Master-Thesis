"""Smoke test for the teacher_finetune group-table + grouped-CE loss.

Risk-mitigation deliverable (implementation plan §2.2's grouped/marginal
cross-entropy design), analogous to yolo26n's `smoke_test_loss_and_decode.py`.
Must pass before any real training run, same gate discipline as Goal A.

Checks:
  1. Group-table self-consistency (requires `speciesnet` — SKIPped, not
     failed, if unavailable, so this test stays runnable outside the
     `Dockerfile.speciesnet` container): no species-level `idx_225` maps to
     **more than 1** leaf index (that would be a real ambiguity bug — verified
     clean: 0 occurrences). Empty groups (0 leaf matches) are reported but not
     treated as a failure — running this against the real classifier found 11
     project classes with **no** corresponding SpeciesNet leaf class at all
     (blackbuck, eared seals, elephant seal, japanese macaque, kob, pinniped
     clade, ring-tailed lemur, saiga, sea otter, walrus, yak) — a genuine,
     pre-existing taxonomy-coverage gap (also affects the existing production
     `compute_probs_225`), not a bug in this package's grouping logic. See
     `loss.py`'s docstring and README.md's "Limitations" section.
  2. `GroupedCrossEntropyLoss` reduces to `F.cross_entropy` exactly when a
     group has 1 member (pure Python/CPU — no `speciesnet` needed).
  3. Numerical stability (finite loss + finite gradients) on multi-member
     groups with exaggerated-magnitude synthetic logits.
  4. Empty-group handling: a batch containing a label whose group is empty
     does not crash or produce NaN — that sample is excluded from the loss
     (ignore_index-style) and the batch's loss is still finite.
  5. `compute_probs_225` round-trip sanity (requires `speciesnet`): a
     one-hot leaf softmax on a species-level class's sole leaf member
     projects back to that same `idx_225` with `prob_225_sum == 1.0`.

Run from the repo root:
    python -m scripts.training.teacher_finetune.smoke_test_taxonomy_and_loss
(checks 1 & 5 require the SpeciesNet package; checks 2-4 run anywhere.)
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.teacher_finetune.loss import GroupedCrossEntropyLoss

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def check(condition: bool, label: str, detail: str = "") -> bool:
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return bool(condition)


def _speciesnet_available() -> bool:
    try:
        import speciesnet  # noqa: F401

        return True
    except ImportError:
        return False


def run_group_table_check() -> bool:
    print("\n1. Group-table self-consistency (requires speciesnet)")
    if not _speciesnet_available():
        print(f"  [{SKIP}] speciesnet not installed — run inside Dockerfile.speciesnet to exercise this check")
        return True

    from scripts.training.teacher_finetune.taxonomy import build_group_table

    groups, levels = build_group_table()

    all_pass = True
    n_species = sum(1 for lvl in levels.values() if lvl == "species")
    n_genus = sum(1 for lvl in levels.values() if lvl == "genus")
    n_family = sum(1 for lvl in levels.values() if lvl == "family")
    all_pass &= check(
        n_species + n_genus + n_family == len(levels) == 225,
        "225 classes partition into species/genus/family levels",
        detail=f"species={n_species} genus={n_genus} family={n_family}",
    )

    empty_groups = sorted(idx for idx, g in groups.items() if len(g) == 0)
    print(
        f"  [info] {len(empty_groups)} idx_225 classes have an empty leaf group "
        f"(no matching SpeciesNet leaf — genuine taxonomy gap, not a bug): {empty_groups}"
    )

    ambiguous_species = [
        idx for idx, g in groups.items() if levels.get(idx) == "species" and len(g) > 1
    ]
    all_pass &= check(
        not ambiguous_species,
        "no species-level idx_225 maps to MORE THAN 1 leaf index (would be a real ambiguity bug)",
        detail=f"{len(ambiguous_species)} anomalies: {ambiguous_species}",
    )

    return all_pass


def run_loss_math_checks() -> bool:
    all_pass = True
    torch.manual_seed(0)
    num_leaf = 2498

    print("\n2. GroupedCrossEntropyLoss reduces to F.cross_entropy for single-member groups")
    group_table_single = {i: [i * 3 % num_leaf] for i in range(225)}
    loss_fn = GroupedCrossEntropyLoss(group_table_single)
    logits = torch.randn(8, num_leaf)
    targets_225 = torch.randint(0, 225, (8,))

    grouped_loss = loss_fn(logits, targets_225)
    leaf_targets = torch.tensor([group_table_single[int(t)][0] for t in targets_225])
    plain_loss = F.cross_entropy(logits, leaf_targets)
    all_pass &= check(
        torch.allclose(grouped_loss, plain_loss, atol=1e-5),
        "grouped CE == plain CE when |group|==1",
        detail=f"{grouped_loss.item():.6f} vs {plain_loss.item():.6f}",
    )

    print("\n3. Numerical stability on multi-member groups (forward + backward)")
    group_table_multi = {i: list(range(i * 5, i * 5 + 5)) for i in range(225)}
    loss_fn_multi = GroupedCrossEntropyLoss(group_table_multi)
    logits2 = (torch.randn(16, num_leaf) * 10).requires_grad_()  # exaggerated magnitude
    targets2 = torch.randint(0, 225, (16,))
    loss2 = loss_fn_multi(logits2, targets2)
    all_pass &= check(torch.isfinite(loss2).item(), "multi-member grouped CE is finite", detail=f"{loss2.item():.4f}")
    all_pass &= check((loss2 >= 0).item(), "grouped CE is non-negative", detail=f"{loss2.item():.4f}")

    loss2.backward()
    grad_finite = logits2.grad is not None and torch.isfinite(logits2.grad).all().item()
    all_pass &= check(grad_finite, "backward() populates finite gradients on logits")

    return all_pass


def run_empty_group_handling_check() -> bool:
    print("\n4. Empty-group handling (a batch containing an unmappable label doesn't crash)")
    all_pass = True

    # idx_225=0 has an empty group; idx_225=1 has a normal single-member group.
    group_table = {0: []}
    group_table.update({i: [i * 3 % 2498] for i in range(1, 225)})
    loss_fn = GroupedCrossEntropyLoss(group_table)

    logits = torch.randn(4, 2498)
    targets = torch.tensor([0, 1, 0, 1])  # half the batch is unmappable
    try:
        loss = loss_fn(logits, targets)
        ran_ok = True
    except Exception:
        traceback.print_exc()
        ran_ok = False
        loss = None
    all_pass &= check(ran_ok, "loss_fn(logits, targets) does not raise on a mixed batch")
    if ran_ok:
        all_pass &= check(torch.isfinite(loss).item(), "loss is finite (not NaN)", detail=f"{loss.item():.4f}")

    # Whole batch unmappable — must still not crash (degenerate all-ignored case).
    all_empty_targets = torch.tensor([0, 0, 0, 0])
    try:
        loss_all_empty = loss_fn(logits, all_empty_targets)
        ran_ok2 = True
    except Exception:
        traceback.print_exc()
        ran_ok2 = False
        loss_all_empty = None
    all_pass &= check(ran_ok2, "loss_fn does not raise when the ENTIRE batch is unmappable")
    if ran_ok2:
        all_pass &= check(
            torch.isfinite(loss_all_empty).item(),
            "whole-batch-unmappable loss is finite",
            detail=f"{loss_all_empty.item():.4f}",
        )

    return all_pass


def run_probs_225_roundtrip_check() -> bool:
    print("\n5. compute_probs_225 round-trip sanity (requires speciesnet)")
    if not _speciesnet_available():
        print(f"  [{SKIP}] speciesnet not installed — run inside Dockerfile.speciesnet to exercise this check")
        return True

    from scripts.training.teacher_finetune.taxonomy import (
        _load_script7,
        build_group_table,
        projection_tables,
    )

    groups, levels = build_group_table()
    idx_to_label, genus_species_to_225, genus_to_225, family_to_225 = projection_tables()
    s7 = _load_script7()

    species_idx = next(
        idx for idx, lvl in levels.items() if lvl == "species" and len(groups[idx]) == 1
    )
    leaf_idx = groups[species_idx][0]
    scores = [0.0] * 2498
    scores[leaf_idx] = 1.0

    probs_225, prob_sum = s7.compute_probs_225(
        scores, idx_to_label, genus_species_to_225, genus_to_225, family_to_225
    )
    all_pass = True
    all_pass &= check(
        abs(prob_sum - 1.0) < 1e-6,
        "prob_225_sum == 1.0 for a one-hot leaf on a 225-covered class",
        detail=f"{prob_sum}",
    )
    argmax_idx = max(range(225), key=lambda k: probs_225[k])
    all_pass &= check(
        argmax_idx == species_idx,
        "one-hot leaf softmax projects back to the same idx_225",
        detail=f"expected {species_idx}, got {argmax_idx}",
    )
    return all_pass


if __name__ == "__main__":
    all_ok = True
    try:
        all_ok &= run_group_table_check()
        all_ok &= run_loss_math_checks()
        all_ok &= run_empty_group_handling_check()
        all_ok &= run_probs_225_roundtrip_check()
    except Exception:
        traceback.print_exc()
        all_ok = False

    print(f"\n{'=' * 60}")
    print(f"OVERALL: {PASS if all_ok else FAIL}")
    sys.exit(0 if all_ok else 1)
