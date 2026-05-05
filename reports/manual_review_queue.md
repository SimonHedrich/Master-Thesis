# Manual Review Queue

**Date:** 2026-05-05  
**Input:** `reports/class_distribution.csv`  
**Mode:** STATS ONLY — no data files modified

---

## Anomalies

Entries that are not valid canonical wildlife classes. Exclude from all
downstream training and evaluation pipelines.

### Ghost classes (6)

Class names where directory names on disk dropped apostrophes that appear
in the canonical 225-class list. Images belong to their canonical counterpart.
**Fix:** restore apostrophes in Script 8 `process_source` (line 140) so that
`Path(...).parent.name` is mapped through the canonical common-name lookup
before being stored as a class key.

| Ghost name (in CSV) | Canonical name | Trusted images |
|---|---|---:|
| `bairds tapir` | `baird's tapir` | 9 |
| `grants gazelle` | `grant's gazelle` | 7 |
| `grevys zebra` | `grevy's zebra` | 16 |
| `hoffmanns two-toed sloth` | `hoffmann's two-toed sloth` | 7 |
| `kirks dik-dik` | `kirk's dik-dik` | 11 |
| `thomsons gazelle` | `thomson's gazelle` | 9 |

### Pseudo-classes (1)

| Name | Effective pool |
|---|---:|
| `unmatched` | 7,715 |

### Zero-pool canonical classes (0)

None — every canonical class has at least one image.


---

## Workload Summary

All `trusted_quality_pass` images (SN-pass and SN-fail) are reviewed for
each queued class. Work through one class completely before starting the next.

| | Classes | Images to review |
|---|---:|---:|
| Tier 1 | 30 | 1,417 |
| Tier 2 | 63 | 16,289 |
| **Total** | **93** | **17,706** |

By review priority (informational only — priority signals label-error risk,
not whether images are skipped):

| Priority | Classes | Images |
|---|---:|---:|
| P1 HIGH | 14 | 2,160 |
| P2 MED | 53 | 10,662 |
| P3 LOW | 26 | 4,884 |

---

## Review Queue (Tier 1 + Tier 2)

Sorted by `trusted_quality_pass` ascending — smallest class first.
All `trusted_quality_pass` images must be reviewed regardless of priority.

| Class | Tier | eff_pool | tq_pass | tsn_fail_reason | Priority |
|---|---:|---:|---:|---|---|
| human | 1 | 5 | 5 | match_level_no_match | P2 MED |
| mouflon | 1 | 6 | 6 | family_mismatch_high_confidence | P1 HIGH |
| giant armadillo | 1 | 11 | 11 | match_level_no_match | P2 MED |
| domestic pig | 1 | 15 | 15 | match_level_no_match | P2 MED |
| dingo | 2 | 403 | 16 | low_speciesnet_confidence | P3 LOW |
| hog badger genus | 1 | 18 | 18 | match_level_class | P3 LOW |
| malay tapir | 1 | 21 | 21 | match_level_no_match | P2 MED |
| drill | 1 | 23 | 23 | low_speciesnet_confidence | P3 LOW |
| clouded leopard | 1 | 25 | 25 | family_mismatch_high_confidence | P1 HIGH |
| african civet | 1 | 28 | 28 | match_level_order | P3 LOW |
| aye-aye | 1 | 29 | 29 | match_level_no_match | P2 MED |
| red-necked wallaby | 1 | 30 | 30 | match_level_class | P3 LOW |
| binturong | 1 | 36 | 36 | match_level_no_match | P2 MED |
| spilogale species | 1 | 37 | 37 | match_level_no_match | P2 MED |
| bongo | 1 | 39 | 39 | family_mismatch_high_confidence | P1 HIGH |
| aardwolf | 1 | 47 | 47 | family_mismatch_high_confidence | P1 HIGH |
| saiga | 1 | 50 | 50 | match_level_no_match | P2 MED |
| pangolin family | 1 | 52 | 52 | match_level_no_match | P2 MED |
| snow leopard | 2 | 364 | 53 | match_level_no_match | P2 MED |
| red river hog | 1 | 53 | 53 | low_speciesnet_confidence | P3 LOW |
| aardvark | 1 | 54 | 54 | match_level_class | P3 LOW |
| mangabeys genus | 1 | 54 | 54 | match_level_no_match | P2 MED |
| pinniped clade | 1 | 55 | 55 | match_level_no_match | P2 MED |
| red brocket | 1 | 62 | 62 | family_mismatch_high_confidence | P1 HIGH |
| brown hyaena | 1 | 68 | 68 | low_speciesnet_confidence | P3 LOW |
| wolverine | 1 | 69 | 69 | match_level_no_match | P2 MED |
| fossa | 1 | 79 | 79 | low_speciesnet_confidence | P3 LOW |
| sun bear | 1 | 97 | 83 | low_speciesnet_confidence | P3 LOW |
| asiatic wild ass | 1 | 84 | 84 | match_level_class | P3 LOW |
| maned wolf | 1 | 88 | 88 | match_level_no_match | P2 MED |
| honey badger | 1 | 97 | 97 | match_level_no_match | P2 MED |
| ocelot | 1 | 99 | 99 | match_level_no_match | P2 MED |
| asiatic black bear | 2 | 104 | 104 | match_level_no_match | P2 MED |
| leopard cat | 2 | 105 | 105 | match_level_no_match | P2 MED |
| striped hyaena | 2 | 105 | 105 | match_level_order | P3 LOW |
| callicebus genus | 2 | 119 | 119 | match_level_no_match | P2 MED |
| sloth bear | 2 | 120 | 120 | family_mismatch_high_confidence | P1 HIGH |
| cephalophus species | 2 | 124 | 124 | match_level_order | P3 LOW |
| giant panda | 2 | 172 | 125 | match_level_no_match | P2 MED |
| wild cat | 2 | 126 | 126 | match_level_no_match | P2 MED |
| black-backed jackal | 2 | 127 | 127 | low_speciesnet_confidence | P3 LOW |
| yak | 2 | 127 | 127 | match_level_class | P3 LOW |
| fisher | 2 | 131 | 131 | match_level_no_match | P2 MED |
| water deer | 2 | 152 | 152 | family_mismatch_high_confidence | P1 HIGH |
| walrus | 2 | 158 | 158 | match_level_no_match | P2 MED |
| kinkajou | 2 | 160 | 160 | match_level_no_match | P2 MED |
| canada lynx | 2 | 166 | 166 | match_level_no_match | P2 MED |
| black wildebeest | 2 | 170 | 170 | family_mismatch_high_confidence | P1 HIGH |
| caracal | 2 | 171 | 171 | match_level_no_match | P2 MED |
| patas monkey | 2 | 179 | 179 | match_level_no_match | P2 MED |
| gerenuk | 2 | 180 | 180 | family_mismatch_high_confidence | P1 HIGH |
| domestic goat | 2 | 337 | 183 | family_mismatch_high_confidence | P1 HIGH |
| ringtail | 2 | 186 | 186 | match_level_no_match | P2 MED |
| american mink | 2 | 239 | 194 | match_level_no_match | P2 MED |
| serval | 2 | 196 | 196 | family_mismatch_high_confidence | P1 HIGH |
| red panda | 2 | 297 | 197 | match_level_no_match | P2 MED |
| eurasian lynx | 2 | 200 | 200 | match_level_no_match | P2 MED |
| spectacled bear | 2 | 201 | 201 | match_level_no_match | P2 MED |
| dhole | 2 | 202 | 202 | match_level_class | P3 LOW |
| grevy's zebra | 2 | 209 | 209 | match_level_no_match | P2 MED |
| kirk's dik-dik | 2 | 215 | 215 | match_level_order | P3 LOW |
| genet genus | 2 | 227 | 227 | match_level_no_match | P2 MED |
| bat-eared fox | 2 | 228 | 228 | low_speciesnet_confidence | P3 LOW |
| raccoon dog | 2 | 232 | 232 | match_level_no_match | P2 MED |
| old world porcupine family | 2 | 238 | 238 | match_level_no_match | P2 MED |
| nine-banded armadillo | 2 | 240 | 240 | match_level_no_match | P2 MED |
| european bison | 2 | 265 | 265 | family_mismatch_high_confidence | P1 HIGH |
| leopardus species | 2 | 271 | 271 | match_level_no_match | P2 MED |
| roan antelope | 2 | 278 | 278 | match_level_no_match | P2 MED |
| kob | 2 | 289 | 289 | family_mismatch_high_confidence | P1 HIGH |
| lowland tapir | 2 | 294 | 294 | match_level_class | P3 LOW |
| tayra | 2 | 317 | 317 | match_level_no_match | P2 MED |
| baird's tapir | 2 | 323 | 323 | match_level_no_match | P2 MED |
| chimpanzee | 2 | 335 | 335 | match_level_no_match | P2 MED |
| red kangaroo | 2 | 380 | 344 | match_level_class | P3 LOW |
| giant anteater | 2 | 355 | 355 | match_level_no_match | P2 MED |
| quokka | 2 | 366 | 366 | match_level_class | P3 LOW |
| glaucomys species | 2 | 372 | 372 | match_level_no_match | P2 MED |
| american badger | 2 | 389 | 389 | match_level_no_match | P2 MED |
| bushbuck | 2 | 391 | 391 | match_level_order | P3 LOW |
| meerkat | 2 | 412 | 412 | match_level_no_match | P2 MED |
| dromedary camel | 2 | 420 | 420 | match_level_order | P3 LOW |
| sable antelope | 2 | 426 | 426 | family_mismatch_high_confidence | P1 HIGH |
| eurasian otter | 2 | 434 | 434 | match_level_no_match | P2 MED |
| giant otter | 2 | 442 | 442 | match_level_no_match | P2 MED |
| blackbuck | 2 | 448 | 448 | match_level_no_match | P2 MED |
| puma | 2 | 452 | 452 | match_level_no_match | P2 MED |
| ring-tailed lemur | 2 | 452 | 452 | match_level_class | P3 LOW |
| thomson's gazelle | 2 | 452 | 452 | match_level_no_match | P2 MED |
| eurasian badger | 2 | 460 | 460 | match_level_no_match | P2 MED |
| common duiker | 2 | 472 | 472 | match_level_order | P3 LOW |
| japanese macaque | 2 | 481 | 481 | low_speciesnet_confidence | P3 LOW |
| mountain zebra | 2 | 483 | 483 | match_level_no_match | P2 MED |
