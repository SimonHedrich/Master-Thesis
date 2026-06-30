# Look-Alike Grouping Review & Refinement

**Date:** 2026-06-11
**Status:** Decision record — refines the frozen coarse-grouping table before evaluation
**Implements / refines:** `docs/plans/2026-06-10_model-evaluation-strategy.md` §5
(open question #1), reviewed per the manual step in
`docs/progress_notes/2026-06-11_evaluation-suite-implementation.md`.
**Inputs:** `reports/lookalike_groups.csv` (v1, genus backbone + 3 overrides),
`reports/classes_225.csv`
**Output:** `reports/lookalike_groups_v2.csv` (the table evaluation will use),
produced by `scripts/dataset_quality/16-build_lookalike_groups.py`.

---

## 1. What this table is for (the criterion for any change)

`coarse` granularity exists to **decompose detector error**, not as a rival
accuracy number. The strategy doc (§4) defines the gap ladder:

```
mAP_detect  ≥  mAP_coarse  ≥  mAP_fine
              └─ Δ_coarse: cost of naming the *group*   (cross-group confusion = a REAL failure)
                 Δ_fine:   cost of naming the *species* within a group (forgivable look-alike slip)
```

So the only correct test for whether two classes belong in the same coarse
group is **visual confusability in a single still frame**:

- **Merge** two classes ⇔ confusing them is a *forgivable* mistake we want
  charged to Δ_fine, not Δ_coarse.
- **Keep apart** ⇔ confusing them is a *genuine* failure we want to surface in
  Δ_coarse.

Taxonomy (genus) is a cheap, reproducible, leakage-free **proxy** for visual
similarity, and it is the right backbone. But it is a *leaky* proxy in both
directions, and the strategy doc explicitly sanctions correcting it with a
"small, frozen, documented override list" — including **keeping
visually-distinct same-genus classes split** (§5). The v1 builder applied only
*merge* overrides and noted "same-genus splits are not currently needed." This
review exercises the doc's "if warranted" clause: two genera clearly need
splitting, and one cross-genus look-alike pair clearly needs merging.

## 2. Review of the v1 table

v1 = 173 groups (28 multi-member) from genus rollup + 3 overrides (`elephant`,
`lynx_caracal_cluster`, `hyaena`). I checked every multi-member group against
the still-frame criterion.

**Genus groups that are good look-alike clusters — kept unchanged (24):**
`ursus` (3 bears), `capra` (ibex/goat), `marmota` (3 marmots), `ovis` (3 sheep),
`connochaetes` (2 wildebeest), `bison` (2 bison), `cervus` (elk/red/sika —
reddish deer), `sciurus` (4 tree squirrels), `lepus` (hares), `macaca`
(macaques), `tapirus` (3 tapirs — identical body plan), `odocoileus` (mule/
white-tailed deer), `felis` (domestic/wild cat), `hippotragus` (roan/sable),
`kobus` (kob/waterbuck), `bos` (cattle/yak), `sus` (pig/boar), `nasua` (coatis),
`leopardus` (ocelot + sp.), `macropus` (kangaroo/wallaby), `sylvilagus`
(cottontails), `tragelaphus` (5 spiral-horned antelopes), and the 3 overrides.
These all describe animals genuinely hard to tell apart in a still frame.

**`canis` (jackals/coyote/dingo/dog/wolf) — kept, with a noted caveat.** All six
are dog-shaped canids that are routinely confused in camera-trap imagery, so the
group is correct. The one wildcard is `domestic dog` (breed morphology is
enormous); this is recorded as a caveat for the writeup but does not justify a
split — wolf/coyote/jackal/dingo/dog confusion is exactly the forgivable error
Δ_fine should absorb.

**Two clear over-merges and one omission — changed (see §3).**

## 3. Changes (each individually justified)

### 3.1 Split `panthera` — STRONG (the clearest over-merge)

v1 merges **lion, tiger, leopard, jaguar, snow leopard** into one genus group.
These are not mutually confusable:

| Class | Distinguishing visual signature |
|-------|---------------------------------|
| lion | uniform tawny coat, mane (males), no pattern |
| tiger | bold orange-and-black vertical stripes |
| leopard / jaguar / snow leopard | **rosette-patterned** coat, similar felid build |

Lion and tiger are *unmistakable* — a model that calls a tiger a lion has made a
real classification failure, and merging them lets that error hide in Δ_fine.
The genuine fine-grained confusion is **leopard ↔ jaguar** (the textbook
rosette-cat pair), with snow leopard sharing the rosette pattern (it differs
mainly in coat *colour*, which degrades in poor light).

**Decision:** split `panthera` into three groups —
`lion` (singleton), `tiger` (singleton), and `panthera_rosette` =
{leopard, jaguar, snow leopard}.
This is also taxonomically clean: lion and tiger are precisely the two
non-rosette outliers in the genus. *(Snow leopard is the weakest member of the
rosette group — pale and thick-furred; it is the first candidate to demote to a
singleton if a reviewer disagrees.)*

### 3.2 Split `equus` — STRONG (doc-supported)

v1 merges all six equids — **grevy's / mountain / plains zebra** +
**asiatic wild ass / domestic donkey / domestic horse** — into one genus group.
The strategy doc §1 itself enumerates *"the three zebra species"* and *"the
Equus asses"* as **two distinct** look-alike clusters. Stripes are an unmissable
feature: a zebra is never confused with an unstriped horse or ass in a clear
frame, so that error belongs in Δ_coarse, not Δ_fine.

**Decision:** split `equus` into
`zebra` = {grevy's zebra, mountain zebra, plains zebra} and
`equine_unstriped` = {asiatic wild ass, domestic donkey, domestic horse}.
Within-group confusion (zebra-species ↔ zebra-species; ass ↔ donkey ↔ horse) is
where the genuine fine-grained difficulty lives and is correctly charged to
Δ_fine.

### 3.3 Merge true gazelles — MODERATE (reverses a v1 decision)

v1 leaves the gazelles as separate singletons (`nanger`, `eudorcas`,
`antidorcas`), with a builder comment that they are "sufficiently distinct." I
disagree for the core pair: **Grant's gazelle (Nanger granti)** and **Thomson's
gazelle (Eudorcas thomsonii)** are nearly identical (tan coat, white belly, dark
flank band, lyre horns), co-occur in East Africa, and are confused even by
experienced observers — a textbook look-alike pair. **Springbok (Antidorcas
marsupialis)** shares the same tan/white/flank-band template. Doc §1 explicitly
lists "multiple gazelles and antelopes" as a look-alike concern, so this merge
is consistent with the strategy's intent.

**Decision:** add cross-genus override
`gazelle` = {grant's gazelle, thomson's gazelle, springbok}.

*Deliberately excluded* (distinct enough to keep separate, charged to Δ_coarse
if confused): **gerenuk** (absurdly long neck/legs), **blackbuck** (males black-
and-white, spiral horns), **impala** (reddish, vertical black rump stripes,
different build — commonly *called* a gazelle but visually separable). This is
the most debatable change and is isolated in its own override so it can be
reverted without touching anything else.

## 4. Considered but NOT adopted (kept conservative)

To keep the table defensible and minimal, the following plausible cross-genus
merges were evaluated and **rejected** for v2 (recorded so the reasoning is
auditable; any can be promoted later by adding one override line):

- **River otters** (`lutra` Eurasian + `lontra` N-American river otter): very
  similar, but geographically disjoint and the genus backbone already isolates
  them; sea/giant otter differ by size. Borderline — left split.
- **Small spotted cats** (ocelot/`leopardus`, leopard cat/`prionailurus`,
  serval/`leptailurus`): serval is distinctive (long legs/large ears) and the
  others are on different continents; too speculative to merge a-priori.
- **Cheetah with the rosette cats**: solid round spots + slender build +
  tear-marks make it separable; kept singleton.
- **Clouded leopard**: distinctive large cloud blotches; kept singleton.
- **Foxes** (red/grey/bat-eared): grey and bat-eared foxes are visually
  distinct; no merge.

This mirrors the doc's conservatism: when uncertain, do **not** merge — an
unjustified merge silently forgives a real error.

## 5. Net effect (v1 → v2)

| | v1 | v2 |
|---|----|----|
| Total groups | 173 | 174 |
| Multi-member groups | 28 | 30 |
| Override groups | 3 | 8 |

Changes: `panthera` (1 group → 3: +`lion`, +`tiger`, `panthera_rosette`);
`equus` (1 group → 2: `zebra`, `equine_unstriped`); gazelles (3 singletons → 1
`gazelle` group). All changes are expressed as **frozen curated overrides** in
`16-build_lookalike_groups.py`; the genus backbone is otherwise untouched.

## 6. Reproducibility & freezing

- v1 (`reports/lookalike_groups.csv`) is **preserved unchanged** as an audit
  artifact (the genus-only baseline).
- v2 (`reports/lookalike_groups_v2.csv`) is the **active** table; the eval suite
  (`eval_suite/grouping.py`) now defaults to it.
- Rebuild: `python scripts/dataset_quality/16-build_lookalike_groups.py`.
- **v2 is frozen as of this document.** Any later change invalidates previously
  computed coarse-granularity mAP numbers and must be recorded as a new version.

## 7. Empirical cross-check (after the fact, never to define groups)

Per doc §5, option (b) (confusion-driven grouping) is **circular** and must not
define the table. After the first evaluation, compare the within-group confusion
matrix (§4.2) against these a-priori groups *descriptively only* — e.g. confirm
that leopard↔jaguar confusion is high (validating `panthera_rosette`) and that
lion↔tiger confusion is low (validating their separation). If the model's
empirical confusions diverge sharply from this table, that is a finding to
report, not a reason to silently re-cut the groups mid-study.
