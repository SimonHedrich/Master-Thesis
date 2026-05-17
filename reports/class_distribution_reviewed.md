# Post-Review Class Distribution Report

**Date:** 2026-05-07  
**Base CSV:** /home/debian/Master-Thesis/reports/class_distribution.csv  
**Decisions:** /home/debian/Master-Thesis/reports/review_decisions.jsonl  
**Review stats:** 20,351 images reviewed (16,215 approved · 4,136 declined)  
**Tier changes after review:** 33 classes

---

## Filtering rules by tier

| Tier | Trusted source gate | Unverified source gate |
|---:|---|---|
| 1 & 2 | Manual review (SN skipped); `review_approved` only | SpeciesNet pass |
| 3 | SN-pass valid; coverage-gap classes (no_match/class <20% or order <15% pass rate) → `trusted_quality_pass`; otherwise `trusted_sn_pass + review_approved` | SpeciesNet pass |
| 4 | All quality-pass assumed valid; `trusted_quality_pass` | SpeciesNet pass |

## Tier Summary

| Tier | Pool range | Pre-review classes | Post-review classes | Post-review pool |
|---:|---|---:|---:|---:|
| 1 | < 100 | 30 | 38 | 1,843 |
| 2 | 100–499 | 63 | 81 | 22,194 |
| 3 | 500–1 499 | 52 | 26 | 22,375 |
| 4 | ≥ 1 500 | 81 | 81 | 387,218 |

## Per-Class Table

Columns: `tq_pass` = trusted quality-pass · `tsn_pass` = trusted SN-pass · `uv_pass` = unverified SN-pass · `rev_app` = review approved · `rev_dec` = review declined · `eff_trusted` = effective trusted (tier-aware) · `eff_pool` = effective_trusted + uv_pass

| Class | tq_pass | tsn_pass | uv_pass | tier | rev_app | rev_dec | eff_trusted | eff_pool | final_tier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| squirrel family | 31159 | 7942 | 206 | 4 | 0 | 0 | 31159 | 31365 | 4 |
| eastern gray squirrel | 31099 | 15646 | 0 | 4 | 0 | 0 | 31099 | 31099 | 4 |
| white-tailed deer | 24425 | 17187 | 184 | 4 | 0 | 0 | 24425 | 24609 | 4 |
| mule deer | 18110 | 12709 | 0 | 4 | 0 | 0 | 18110 | 18110 | 4 |
| eastern fox squirrel | 12436 | 6456 | 0 | 4 | 0 | 0 | 12436 | 12436 | 4 |
| eastern cottontail | 11456 | 3567 | 0 | 4 | 0 | 0 | 11456 | 11456 | 4 |
| eurasian red squirrel | 9929 | 2663 | 0 | 4 | 0 | 0 | 9929 | 9929 | 4 |
| eared seals | 9261 | 0 | 0 | 4 | 0 | 0 | 9261 | 9261 | 4 |
| red fox | 8524 | 3515 | 181 | 4 | 0 | 0 | 8524 | 8705 | 4 |
| cottontail rabbits genus | 7757 | 2214 | 0 | 4 | 0 | 0 | 7757 | 7757 | 4 |
| unmatched | 7715 | 0 | 0 | 4 | 0 | 0 | 7715 | 7715 | 4 |
| macaque species | 7557 | 2305 | 0 | 4 | 0 | 0 | 7557 | 7557 | 4 |
| red squirrel | 7526 | 2360 | 0 | 4 | 0 | 0 | 7526 | 7526 | 4 |
| domestic cat | 6980 | 3688 | 179 | 4 | 0 | 0 | 6980 | 7159 | 4 |
| chipmunk genus | 6934 | 1753 | 0 | 4 | 0 | 0 | 6934 | 6934 | 4 |
| coyote | 6865 | 3081 | 0 | 4 | 0 | 0 | 6865 | 6865 | 4 |
| kangaroo family | 6137 | 1280 | 0 | 4 | 0 | 0 | 6137 | 6137 | 4 |
| european roe deer | 6011 | 1547 | 0 | 4 | 0 | 0 | 6011 | 6011 | 4 |
| northern raccoon | 5698 | 2154 | 142 | 4 | 0 | 0 | 5698 | 5840 | 4 |
| california ground squirrel | 5627 | 1201 | 0 | 4 | 0 | 0 | 5627 | 5627 | 4 |
| elk | 5517 | 3507 | 0 | 4 | 0 | 0 | 5517 | 5517 | 4 |
| european rabbit | 5387 | 979 | 37 | 4 | 0 | 0 | 5387 | 5424 | 4 |
| cricetidae family | 4917 | 24 | 0 | 4 | 0 | 0 | 4917 | 4917 | 4 |
| hares and jackrabbits genus | 4836 | 987 | 0 | 4 | 0 | 0 | 4836 | 4836 | 4 |
| african elephant | 4338 | 3303 | 0 | 4 | 0 | 0 | 4338 | 4338 | 4 |
| rattus genus | 3954 | 65 | 0 | 4 | 0 | 0 | 3954 | 3954 | 4 |
| woodchuck | 3872 | 1128 | 0 | 4 | 0 | 0 | 3872 | 3872 | 4 |
| moose | 3508 | 1159 | 0 | 4 | 0 | 0 | 3508 | 3508 | 4 |
| american black bear | 3436 | 1289 | 0 | 4 | 0 | 0 | 3436 | 3436 | 4 |
| muridae family | 3371 | 51 | 0 | 4 | 0 | 0 | 3371 | 3371 | 4 |
| lion | 3094 | 1590 | 224 | 4 | 0 | 0 | 3094 | 3318 | 4 |
| hedgehog family | 3296 | 1 | 0 | 4 | 0 | 0 | 3296 | 3296 | 4 |
| muskrat | 3253 | 52 | 0 | 4 | 0 | 0 | 3253 | 3253 | 4 |
| howler monkey genus | 3206 | 169 | 0 | 4 | 0 | 0 | 3206 | 3206 | 4 |
| opossum family | 3199 | 697 | 0 | 4 | 0 | 0 | 3199 | 3199 | 4 |
| plains zebra | 2858 | 2451 | 337 | 4 | 0 | 0 | 2858 | 3195 | 4 |
| bighorn sheep | 3167 | 535 | 0 | 4 | 0 | 0 | 3167 | 3167 | 4 |
| nutria | 3150 | 923 | 0 | 4 | 0 | 0 | 3150 | 3150 | 4 |
| european hare | 3087 | 603 | 0 | 4 | 0 | 0 | 3087 | 3087 | 4 |
| golden mantled ground squirrel | 2819 | 631 | 0 | 4 | 0 | 0 | 2819 | 2819 | 4 |
| north american river otter | 2772 | 459 | 0 | 4 | 0 | 0 | 2772 | 2772 | 4 |
| eastern grey kangaroo | 2680 | 618 | 0 | 4 | 0 | 0 | 2680 | 2680 | 4 |
| wild boar | 2498 | 930 | 104 | 4 | 0 | 0 | 2498 | 2602 | 4 |
| domestic horse | 2467 | 1215 | 83 | 4 | 0 | 0 | 2467 | 2550 | 4 |
| baboon genus | 2486 | 881 | 0 | 4 | 0 | 0 | 2486 | 2486 | 4 |
| domestic cattle | 2148 | 1566 | 211 | 4 | 0 | 0 | 2148 | 2359 | 4 |
| american bison | 2320 | 1105 | 0 | 4 | 102 | 11 | 2320 | 2320 | 4 |
| pronghorn | 2316 | 1065 | 0 | 4 | 0 | 0 | 2316 | 2316 | 4 |
| impala | 2288 | 1563 | 0 | 4 | 0 | 0 | 2288 | 2288 | 4 |
| white-nosed coati | 2269 | 729 | 0 | 4 | 0 | 0 | 2269 | 2269 | 4 |
| mongoose family | 2257 | 118 | 0 | 4 | 0 | 0 | 2257 | 2257 | 4 |
| beaver genus | 2162 | 162 | 0 | 4 | 0 | 0 | 2162 | 2162 | 4 |
| llama genus | 2085 | 125 | 0 | 4 | 0 | 0 | 2085 | 2085 | 4 |
| african buffalo | 2051 | 718 | 0 | 4 | 65 | 22 | 2051 | 2051 | 4 |
| domestic dog | 2019 | 935 | 0 | 4 | 0 | 0 | 2019 | 2019 | 4 |
| western gray squirrel | 1957 | 962 | 0 | 4 | 0 | 0 | 1957 | 1957 | 4 |
| leopard | 1186 | 646 | 748 | 4 | 0 | 0 | 1186 | 1934 | 4 |
| north american porcupine | 1914 | 481 | 0 | 4 | 0 | 0 | 1914 | 1914 | 4 |
| yellow-bellied marmot | 1898 | 703 | 0 | 4 | 0 | 0 | 1898 | 1898 | 4 |
| bobcat | 1861 | 899 | 0 | 4 | 0 | 0 | 1861 | 1861 | 4 |
| agouti genus | 1853 | 670 | 0 | 4 | 0 | 0 | 1853 | 1853 | 4 |
| cebus species | 1819 | 67 | 0 | 4 | 0 | 0 | 1819 | 1819 | 4 |
| arizona black-tailed prairie dog | 1800 | 827 | 0 | 4 | 0 | 0 | 1800 | 1800 | 4 |
| weasel species | 1795 | 88 | 0 | 4 | 0 | 0 | 1795 | 1795 | 4 |
| koala | 1774 | 9 | 5 | 4 | 0 | 0 | 1774 | 1779 | 4 |
| elephant seal | 1768 | 0 | 0 | 4 | 0 | 0 | 1768 | 1768 | 4 |
| capybara | 1767 | 559 | 0 | 4 | 0 | 0 | 1767 | 1767 | 4 |
| short-beaked echidna | 1767 | 1092 | 0 | 4 | 0 | 0 | 1767 | 1767 | 4 |
| callithrix species | 1736 | 3 | 0 | 4 | 0 | 0 | 1736 | 1736 | 4 |
| greater kudu | 1732 | 577 | 0 | 4 | 0 | 0 | 1732 | 1732 | 4 |
| lycalopex species | 1731 | 549 | 0 | 4 | 593 | 18 | 1731 | 1731 | 4 |
| alpine marmot | 1716 | 340 | 0 | 4 | 0 | 0 | 1716 | 1716 | 4 |
| red deer | 1697 | 828 | 0 | 4 | 0 | 0 | 1697 | 1697 | 4 |
| vervet monkey | 1635 | 656 | 0 | 4 | 0 | 0 | 1635 | 1635 | 4 |
| common wildebeest | 1625 | 895 | 0 | 4 | 0 | 0 | 1625 | 1625 | 4 |
| common fallow deer | 1589 | 439 | 0 | 4 | 329 | 29 | 1589 | 1589 | 4 |
| domestic sheep | 1375 | 480 | 187 | 4 | 0 | 0 | 1375 | 1562 | 4 |
| common warthog | 1553 | 914 | 0 | 4 | 0 | 0 | 1553 | 1553 | 4 |
| brown bear | 1523 | 557 | 0 | 4 | 0 | 0 | 1523 | 1523 | 4 |
| brown-throated sloth | 1518 | 0 | 0 | 4 | 0 | 0 | 1518 | 1518 | 4 |
| rhinoceros family | 1512 | 41 | 0 | 4 | 0 | 0 | 1512 | 1512 | 4 |
| pikas genus | 1484 | 31 | 0 | 3 | 0 | 0 | 1484 | 1484 | 3 |
| ateles species | 1437 | 38 | 0 | 3 | 0 | 0 | 1437 | 1437 | 3 |
| sea otter | 1404 | 0 | 0 | 3 | 0 | 0 | 1404 | 1404 | 3 |
| rock hyrax | 1336 | 7 | 0 | 3 | 0 | 0 | 1336 | 1336 | 3 |
| saimiri species | 1327 | 138 | 0 | 3 | 0 | 0 | 1327 | 1327 | 3 |
| martes species | 1298 | 145 | 0 | 3 | 0 | 0 | 1298 | 1298 | 3 |
| saguinus species | 1065 | 5 | 0 | 3 | 0 | 0 | 1065 | 1065 | 3 |
| steenbok | 957 | 167 | 0 | 3 | 0 | 0 | 957 | 957 | 3 |
| eulemur species | 880 | 2 | 0 | 3 | 0 | 0 | 880 | 880 | 3 |
| cheetah | 1041 | 728 | 135 | 3 | 0 | 0 | 728 | 863 | 3 |
| reindeer | 835 | 105 | 0 | 3 | 0 | 0 | 835 | 835 | 3 |
| tiger | 849 | 540 | 258 | 3 | 0 | 0 | 540 | 798 | 3 |
| mountain goat | 793 | 100 | 0 | 3 | 0 | 0 | 793 | 793 | 3 |
| klipspringer | 701 | 68 | 0 | 3 | 0 | 0 | 701 | 701 | 3 |
| leaf monkeys genus | 693 | 74 | 0 | 3 | 0 | 0 | 693 | 693 | 3 |
| hoffmann's two-toed sloth | 674 | 0 | 0 | 3 | 0 | 0 | 674 | 674 | 3 |
| striped skunk | 666 | 101 | 0 | 3 | 0 | 0 | 666 | 666 | 3 |
| colobus species | 663 | 117 | 0 | 3 | 0 | 0 | 663 | 663 | 3 |
| spotted hyaena | 1233 | 640 | 0 | 3 | 0 | 0 | 640 | 640 | 3 |
| gorilla species | 550 | 82 | 59 | 3 | 0 | 0 | 550 | 609 | 3 |
| gemsbok | 818 | 438 | 0 | 3 | 125 | 3 | 563 | 563 | 3 |
| nyala | 957 | 277 | 0 | 3 | 277 | 9 | 554 | 554 | 3 |
| golden jackal | 781 | 304 | 0 | 3 | 247 | 29 | 551 | 551 | 3 |
| nilgai | 549 | 64 | 0 | 3 | 0 | 0 | 549 | 549 | 3 |
| collared peccary | 1419 | 521 | 0 | 3 | 0 | 0 | 521 | 521 | 3 |
| reedbuck genus | 514 | 65 | 0 | 3 | 0 | 0 | 514 | 514 | 3 |
| giraffe | 324 | 260 | 220 | 3 | 0 | 0 | 260 | 480 | 2 |
| sika deer | 1136 | 354 | 0 | 3 | 117 | 23 | 471 | 471 | 2 |
| mountain zebra | 483 | 407 | 0 | 2 | 467 | 16 | 467 | 467 | 2 |
| blesbok | 823 | 185 | 0 | 3 | 254 | 4 | 439 | 439 | 2 |
| waterbuck | 1444 | 437 | 0 | 3 | 0 | 0 | 437 | 437 | 2 |
| jaguar | 561 | 436 | 0 | 3 | 0 | 0 | 436 | 436 | 2 |
| blackbuck | 448 | 0 | 0 | 2 | 428 | 20 | 428 | 428 | 2 |
| grey fox | 1116 | 425 | 0 | 3 | 0 | 0 | 425 | 425 | 2 |
| thomson's gazelle | 461 | 337 | 0 | 2 | 425 | 36 | 425 | 425 | 2 |
| sable antelope | 426 | 97 | 0 | 2 | 424 | 2 | 424 | 424 | 2 |
| swamp wallaby | 1377 | 420 | 0 | 3 | 0 | 0 | 420 | 420 | 2 |
| hippopotamus | 1319 | 298 | 120 | 3 | 0 | 0 | 298 | 418 | 2 |
| alpine ibex | 1294 | 418 | 0 | 3 | 0 | 0 | 418 | 418 | 2 |
| common duiker | 472 | 66 | 0 | 2 | 413 | 59 | 413 | 413 | 2 |
| japanese macaque | 481 | 126 | 0 | 2 | 412 | 69 | 412 | 412 | 2 |
| south american coati | 1014 | 411 | 0 | 3 | 0 | 0 | 411 | 411 | 2 |
| dingo | 16 | 10 | 387 | 2 | 12 | 4 | 12 | 399 | 2 |
| domestic donkey | 1200 | 325 | 73 | 3 | 0 | 0 | 325 | 398 | 2 |
| hartebeest | 877 | 386 | 0 | 3 | 0 | 0 | 386 | 386 | 2 |
| common eland | 962 | 385 | 0 | 3 | 0 | 0 | 385 | 385 | 2 |
| meerkat | 412 | 1 | 0 | 2 | 364 | 48 | 364 | 364 | 2 |
| snow leopard | 53 | 31 | 311 | 2 | 52 | 0 | 52 | 363 | 2 |
| chital | 1169 | 263 | 0 | 3 | 97 | 19 | 360 | 360 | 2 |
| ring-tailed lemur | 452 | 0 | 0 | 2 | 358 | 94 | 358 | 358 | 2 |
| grey wolf | 550 | 244 | 104 | 3 | 0 | 0 | 244 | 348 | 2 |
| sambar | 851 | 274 | 0 | 3 | 68 | 23 | 342 | 342 | 2 |
| red kangaroo | 344 | 35 | 36 | 2 | 306 | 38 | 306 | 342 | 2 |
| grant's gazelle | 535 | 294 | 0 | 3 | 47 | 2 | 341 | 341 | 2 |
| quokka | 366 | 18 | 0 | 2 | 341 | 25 | 341 | 341 | 2 |
| puma | 452 | 181 | 0 | 2 | 338 | 114 | 338 | 338 | 2 |
| cercopithecus species | 1420 | 337 | 0 | 3 | 0 | 0 | 337 | 337 | 2 |
| northern chamois | 1269 | 315 | 0 | 3 | 0 | 0 | 315 | 315 | 2 |
| bushbuck | 391 | 181 | 0 | 2 | 307 | 84 | 307 | 307 | 2 |
| domestic goat | 183 | 74 | 154 | 2 | 143 | 38 | 143 | 297 | 2 |
| dromedary camel | 420 | 108 | 0 | 2 | 296 | 124 | 296 | 296 | 2 |
| kob | 289 | 79 | 0 | 2 | 286 | 3 | 286 | 286 | 2 |
| red panda | 197 | 36 | 100 | 2 | 183 | 14 | 183 | 283 | 2 |
| bornean orangutan | 423 | 154 | 123 | 3 | 0 | 0 | 154 | 277 | 2 |
| giant anteater | 355 | 114 | 0 | 2 | 258 | 97 | 258 | 258 | 2 |
| giant otter | 442 | 112 | 0 | 2 | 251 | 191 | 251 | 251 | 2 |
| muntjac genus | 1049 | 250 | 0 | 3 | 0 | 0 | 250 | 250 | 2 |
| roan antelope | 278 | 43 | 0 | 2 | 250 | 28 | 250 | 250 | 2 |
| springbok | 759 | 247 | 0 | 3 | 0 | 0 | 247 | 247 | 2 |
| eurasian otter | 434 | 26 | 0 | 2 | 247 | 187 | 247 | 247 | 2 |
| tayra | 317 | 66 | 0 | 2 | 243 | 74 | 243 | 243 | 2 |
| lowland tapir | 294 | 100 | 0 | 2 | 243 | 51 | 243 | 243 | 2 |
| european bison | 265 | 82 | 0 | 2 | 243 | 22 | 243 | 243 | 2 |
| common wombat | 614 | 136 | 105 | 3 | 0 | 0 | 136 | 241 | 2 |
| glaucomys species | 372 | 12 | 0 | 2 | 240 | 132 | 240 | 240 | 2 |
| african wild dog | 907 | 229 | 0 | 3 | 0 | 0 | 229 | 229 | 2 |
| asian elephant | 1301 | 164 | 17 | 3 | 43 | 8 | 207 | 224 | 2 |
| chimpanzee | 335 | 30 | 0 | 2 | 216 | 119 | 216 | 216 | 2 |
| american badger | 389 | 58 | 0 | 2 | 215 | 174 | 215 | 215 | 2 |
| kirk's dik-dik | 226 | 120 | 0 | 2 | 211 | 15 | 211 | 211 | 2 |
| grevy's zebra | 225 | 185 | 0 | 2 | 198 | 27 | 198 | 198 | 2 |
| baird's tapir | 332 | 61 | 0 | 2 | 193 | 139 | 193 | 193 | 2 |
| bat-eared fox | 228 | 18 | 0 | 2 | 190 | 38 | 190 | 190 | 2 |
| dhole | 202 | 46 | 0 | 2 | 190 | 12 | 190 | 190 | 2 |
| gerenuk | 180 | 71 | 0 | 2 | 177 | 3 | 177 | 177 | 2 |
| american mink | 194 | 16 | 45 | 2 | 131 | 62 | 131 | 176 | 2 |
| patas monkey | 179 | 12 | 0 | 2 | 175 | 3 | 175 | 175 | 2 |
| serval | 196 | 85 | 0 | 2 | 170 | 25 | 170 | 170 | 2 |
| giant panda | 125 | 41 | 47 | 2 | 122 | 3 | 122 | 169 | 2 |
| black wildebeest | 170 | 51 | 0 | 2 | 162 | 8 | 162 | 162 | 2 |
| eurasian lynx | 200 | 107 | 0 | 2 | 161 | 39 | 161 | 161 | 2 |
| caracal | 171 | 16 | 0 | 2 | 159 | 12 | 159 | 159 | 2 |
| spectacled bear | 201 | 21 | 0 | 2 | 153 | 48 | 153 | 153 | 2 |
| canada lynx | 166 | 47 | 0 | 2 | 151 | 15 | 151 | 151 | 2 |
| nine-banded armadillo | 240 | 110 | 0 | 2 | 145 | 95 | 145 | 145 | 2 |
| eurasian badger | 460 | 76 | 0 | 2 | 141 | 319 | 141 | 141 | 2 |
| water deer | 152 | 16 | 0 | 2 | 133 | 18 | 133 | 133 | 2 |
| leopardus species | 271 | 57 | 0 | 2 | 132 | 139 | 132 | 132 | 2 |
| genet genus | 227 | 10 | 0 | 2 | 132 | 95 | 132 | 132 | 2 |
| kinkajou | 160 | 2 | 0 | 2 | 125 | 35 | 125 | 125 | 2 |
| ringtail | 186 | 10 | 0 | 2 | 123 | 63 | 123 | 123 | 2 |
| black-backed jackal | 127 | 72 | 0 | 2 | 112 | 15 | 112 | 112 | 2 |
| wild cat | 126 | 54 | 0 | 2 | 112 | 14 | 112 | 112 | 2 |
| callicebus genus | 119 | 0 | 0 | 2 | 112 | 7 | 112 | 112 | 2 |
| raccoon dog | 232 | 3 | 0 | 2 | 105 | 127 | 105 | 105 | 2 |
| old world porcupine family | 238 | 55 | 0 | 2 | 102 | 136 | 102 | 102 | 2 |
| walrus | 158 | 0 | 0 | 2 | 101 | 57 | 101 | 101 | 2 |
| sloth bear | 120 | 15 | 0 | 2 | 98 | 22 | 98 | 98 | 1 |
| yak | 127 | 44 | 0 | 2 | 97 | 30 | 97 | 97 | 1 |
| fisher | 131 | 32 | 0 | 2 | 92 | 39 | 92 | 92 | 1 |
| striped hyaena | 105 | 22 | 0 | 2 | 89 | 16 | 89 | 89 | 1 |
| asiatic black bear | 104 | 35 | 0 | 2 | 88 | 15 | 88 | 88 | 1 |
| leopard cat | 105 | 27 | 0 | 2 | 87 | 18 | 87 | 87 | 1 |
| cephalophus species | 124 | 17 | 0 | 2 | 86 | 38 | 86 | 86 | 1 |
| ocelot | 99 | 43 | 0 | 1 | 84 | 15 | 84 | 84 | 1 |
| domestic water buffalo | 694 | 57 | 0 | 3 | 24 | 26 | 81 | 81 | 1 |
| sun bear | 83 | 18 | 14 | 1 | 64 | 19 | 64 | 78 | 1 |
| asiatic wild ass | 84 | 7 | 0 | 1 | 78 | 6 | 78 | 78 | 1 |
| maned wolf | 88 | 35 | 0 | 1 | 76 | 12 | 76 | 76 | 1 |
| honey badger | 97 | 6 | 0 | 1 | 72 | 25 | 72 | 72 | 1 |
| fossa | 79 | 6 | 0 | 1 | 62 | 17 | 62 | 62 | 1 |
| brown hyaena | 68 | 5 | 0 | 1 | 55 | 13 | 55 | 55 | 1 |
| red brocket | 62 | 18 | 0 | 1 | 47 | 15 | 47 | 47 | 1 |
| pinniped clade | 55 | 0 | 0 | 1 | 47 | 8 | 47 | 47 | 1 |
| saiga | 50 | 0 | 0 | 1 | 47 | 3 | 47 | 47 | 1 |
| wolverine | 69 | 10 | 0 | 1 | 46 | 23 | 46 | 46 | 1 |
| pangolin family | 52 | 2 | 0 | 1 | 45 | 7 | 45 | 45 | 1 |
| mangabeys genus | 54 | 12 | 0 | 1 | 42 | 12 | 42 | 42 | 1 |
| red river hog | 53 | 16 | 0 | 1 | 42 | 7 | 42 | 42 | 1 |
| aardwolf | 47 | 9 | 0 | 1 | 39 | 8 | 39 | 39 | 1 |
| bongo | 39 | 11 | 0 | 1 | 39 | 0 | 39 | 39 | 1 |
| binturong | 36 | 0 | 0 | 1 | 31 | 5 | 31 | 31 | 1 |
| aardvark | 54 | 0 | 0 | 1 | 28 | 25 | 28 | 28 | 1 |
| spilogale species | 37 | 2 | 0 | 1 | 24 | 13 | 24 | 24 | 1 |
| red-necked wallaby | 30 | 15 | 0 | 1 | 23 | 7 | 23 | 23 | 1 |
| clouded leopard | 25 | 0 | 0 | 1 | 23 | 2 | 23 | 23 | 1 |
| malay tapir | 21 | 9 | 0 | 1 | 21 | 0 | 21 | 21 | 1 |
| aye-aye | 29 | 0 | 0 | 1 | 18 | 11 | 18 | 18 | 1 |
| drill | 23 | 2 | 0 | 1 | 15 | 8 | 15 | 15 | 1 |
| domestic pig | 15 | 10 | 0 | 1 | 14 | 1 | 14 | 14 | 1 |
| giant armadillo | 11 | 2 | 0 | 1 | 10 | 1 | 10 | 10 | 1 |
| hog badger genus | 18 | 3 | 0 | 1 | 7 | 11 | 7 | 7 | 1 |
| african civet | 28 | 6 | 0 | 1 | 6 | 21 | 6 | 6 | 1 |
| mouflon | 6 | 2 | 0 | 1 | 6 | 0 | 6 | 6 | 1 |
| human | 5 | 0 | 0 | 1 | 0 | 5 | 0 | 0 | 1 |
