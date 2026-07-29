"""Grouped/marginal cross-entropy loss for genus/family-level rollup labels.

Implementation plan §2.2: 178 of the 225 project classes are species-level (a
1:1 match to exactly one of SpeciesNet's 2,498 leaf classes) but 35 are
genus-level and 12 family-level rollups with no single correct leaf index — a
genus-level label like "weasel species" maps to every ``mustela *`` leaf class
at once. Plain single-label cross-entropy is undefined for these 47 classes.

For a group of leaf indices ``G``, the target is "any leaf class in G":

    loss = -log( sum_{i in G} softmax(logits)_i )
         = logsumexp(logits) - logsumexp(logits[G])   (log-sum-exp over the group)

which reduces exactly to standard cross-entropy when ``|G| == 1`` (verified by
`smoke_test_taxonomy_and_loss.py`), and is numerically stable since
`torch.logsumexp` applies the max-subtraction trick internally.

**Empty groups — a real, discovered data characteristic, not a code bug.**
`smoke_test_taxonomy_and_loss.py` run against the actual SpeciesNet classifier
found **11 of 225 project classes have NO corresponding leaf class anywhere in
SpeciesNet's native 2,498-class taxonomy** (verified: 0 false positives — every
species-level class either matches exactly 1 leaf or 0, never >1, so this is
not a group-table bug): blackbuck, eared seals, elephant seal, japanese
macaque, kob, pinniped clade, ring-tailed lemur, saiga, sea otter, walrus, yak
— species SpeciesNet's own training distribution apparently never covered.
This also means the *existing* production `compute_probs_225` projection
(`7-filter_speciesnet.py`, used unmodified by this package) has always given
these 11 classes exactly zero probability mass — a pre-existing, structural
ceiling on SpeciesNet's achievable recall for them, not something this
package introduces. Samples labeled with one of these classes are **excluded
from the loss** (`ignore`d, matching the standard CE `ignore_index`
convention) rather than crashing on an empty-tensor `logsumexp` — see
README.md's "Limitations" section for the full class list and framing.

No person-class downweighting here (unlike the detector pipelines' 0.3× loss
weight on MegaDetector's `person` class) — this classifier only ever sees
animal-class MegaDetector crops (`data/real/annotations_*.json` contains no
person-class annotations), so there is no person-crop training signal to
downweight. See README.md's "Deviations" section.

Batch loop, not a vectorized masked-scatter: group sizes vary per sample
(1 to dozens of leaf indices), and batch sizes here are small (tens, not
thousands) relative to the GPU forward/backward pass this loss sits behind, so
the python-level loop is not the bottleneck. Kept simple and directly
auditable against the formula above rather than prematurely optimized.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GroupedCrossEntropyLoss(nn.Module):
    def __init__(self, group_table: dict[int, list[int]]) -> None:
        super().__init__()
        self.group_table = group_table

        n_single = sum(1 for g in group_table.values() if len(g) == 1)
        n_multi = sum(1 for g in group_table.values() if len(g) > 1)
        self.empty_idx_225 = {idx for idx, g in group_table.items() if len(g) == 0}
        if self.empty_idx_225:
            logger.warning(
                "%d idx_225 classes have an EMPTY leaf group (no matching SpeciesNet "
                "leaf class) — samples with these labels are excluded from the loss "
                "(ignore_index-style), not treated as NaN. idx_225 set: %s. "
                "See this module's docstring for the full class list/framing.",
                len(self.empty_idx_225),
                sorted(self.empty_idx_225),
            )
        logger.info(
            "GroupedCrossEntropyLoss: %d single-member groups (plain CE fast path), "
            "%d multi-member groups (grouped CE), %d empty (ignored)",
            n_single,
            n_multi,
            len(self.empty_idx_225),
        )

    def forward(self, logits: torch.Tensor, targets_225: torch.Tensor) -> torch.Tensor:
        """logits: [B, NUM_CLASSES_LEAF] raw (pre-softmax) model outputs.
        targets_225: [B] long tensor of idx_225 values (0..224).

        Samples whose label has an empty leaf group are excluded from the mean
        (matching `F.cross_entropy`'s `ignore_index` convention) rather than
        propagating a NaN from `logsumexp` over an empty index tensor.
        """
        losses = []
        for i in range(logits.shape[0]):
            idx_225 = int(targets_225[i].item())
            group = self.group_table[idx_225]

            if len(group) == 0:
                continue
            elif len(group) == 1:
                target = torch.tensor([group[0]], device=logits.device, dtype=torch.long)
                losses.append(F.cross_entropy(logits[i : i + 1], target))
            else:
                group_idx = torch.tensor(group, device=logits.device, dtype=torch.long)
                logsumexp_all = torch.logsumexp(logits[i], dim=-1)
                logsumexp_group = torch.logsumexp(logits[i, group_idx], dim=-1)
                losses.append(logsumexp_all - logsumexp_group)

        if not losses:
            # Whole batch happened to be unmappable labels (very unlikely at
            # 11/225 classes, but must not crash training). Zero loss that
            # still participates in autograd so the training loop's
            # loss.backward() call stays valid.
            return logits.sum() * 0.0

        return torch.stack(losses).mean()
