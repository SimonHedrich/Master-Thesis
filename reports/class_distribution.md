# Trust-Aware Class Distribution Report

**Date:** 2026-05-04  
**Thresholds:** md_conf≥0.5  sn_score≥0.3  family_fail≥0.5  
**Mode:** STATS ONLY — no JSONL files modified

---

## Tier Summary

| Tier | Effective pool | Classes | Total effective pool |
|---:|---|---:|---:|
| 1 | < 100 | 36 | 1,490 |
| 2 | 100–499 | 63 | 17,369 |
| 3 | 500–1 499 | 52 | 51,112 |
| 4 | ≥ 1 500 | 81 | 387,218 |

## Per-Class Table

Columns: `tq_pass` = trusted quality-pass · `tsn_pass` = trusted SN-pass · `tsn_fail` = trusted quality-pass but SN-fail · `t_no_sn` = no SN result · `uv_pass` = unverified SN-pass

| Class | tq_pass | tsn_pass | tsn_fail | tsn_fail_reason | t_no_sn | uv_pass | eff_pool | tier |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| squirrel family | 31,159 | 7,942 | 23,217 | match_level_no_match | 0 | 206 | 31,365 | 4 |
| eastern gray squirrel | 31,099 | 15,646 | 15,452 | match_level_no_match | 1 | 0 | 31,099 | 4 |
| white-tailed deer | 24,425 | 17,187 | 7,238 | match_level_no_match | 0 | 184 | 24,609 | 4 |
| mule deer | 18,110 | 12,709 | 5,400 | match_level_no_match | 1 | 0 | 18,110 | 4 |
| eastern fox squirrel | 12,436 | 6,456 | 5,980 | match_level_no_match | 0 | 0 | 12,436 | 4 |
| eastern cottontail | 11,456 | 3,567 | 7,889 | match_level_no_match | 0 | 0 | 11,456 | 4 |
| eurasian red squirrel | 9,929 | 2,663 | 7,266 | match_level_no_match | 0 | 0 | 9,929 | 4 |
| eared seals | 9,261 | 0 | 9,261 | match_level_no_match | 0 | 0 | 9,261 | 4 |
| red fox | 8,524 | 3,515 | 5,009 | match_level_no_match | 0 | 181 | 8,705 | 4 |
| cottontail rabbits genus | 7,757 | 2,214 | 5,543 | match_level_no_match | 0 | 0 | 7,757 | 4 |
| unmatched | 7,715 | 0 | 7,715 | not_in_225_classes | 0 | 0 | 7,715 | 4 |
| macaque species | 7,557 | 2,305 | 5,252 | match_level_no_match | 0 | 0 | 7,557 | 4 |
| red squirrel | 7,526 | 2,360 | 5,166 | match_level_no_match | 0 | 0 | 7,526 | 4 |
| domestic cat | 6,980 | 3,688 | 3,292 | match_level_no_match | 0 | 179 | 7,159 | 4 |
| chipmunk genus | 6,934 | 1,753 | 5,181 | match_level_no_match | 0 | 0 | 6,934 | 4 |
| coyote | 6,865 | 3,081 | 3,784 | match_level_no_match | 0 | 0 | 6,865 | 4 |
| kangaroo family | 6,137 | 1,280 | 4,857 | match_level_no_match | 0 | 0 | 6,137 | 4 |
| european roe deer | 6,011 | 1,547 | 4,464 | match_level_no_match | 0 | 0 | 6,011 | 4 |
| northern raccoon | 5,698 | 2,154 | 3,544 | match_level_no_match | 0 | 142 | 5,840 | 4 |
| california ground squirrel | 5,627 | 1,201 | 4,426 | match_level_no_match | 0 | 0 | 5,627 | 4 |
| elk | 5,517 | 3,507 | 2,010 | match_level_no_match | 0 | 0 | 5,517 | 4 |
| european rabbit | 5,387 | 979 | 4,406 | match_level_no_match | 2 | 37 | 5,424 | 4 |
| cricetidae family | 4,917 | 24 | 4,892 | match_level_no_match | 1 | 0 | 4,917 | 4 |
| hares and jackrabbits genus | 4,836 | 987 | 3,849 | match_level_no_match | 0 | 0 | 4,836 | 4 |
| african elephant | 4,338 | 3,303 | 1,035 | match_level_no_match | 0 | 0 | 4,338 | 4 |
| rattus genus | 3,954 | 65 | 3,889 | match_level_no_match | 0 | 0 | 3,954 | 4 |
| woodchuck | 3,872 | 1,128 | 2,744 | match_level_no_match | 0 | 0 | 3,872 | 4 |
| moose | 3,508 | 1,159 | 2,348 | match_level_no_match | 1 | 0 | 3,508 | 4 |
| american black bear | 3,436 | 1,289 | 2,145 | match_level_no_match | 2 | 0 | 3,436 | 4 |
| muridae family | 3,371 | 51 | 3,320 | match_level_no_match | 0 | 0 | 3,371 | 4 |
| lion | 3,094 | 1,590 | 1,504 | match_level_no_match | 0 | 224 | 3,318 | 4 |
| hedgehog family | 3,296 | 1 | 3,294 | match_level_no_match | 1 | 0 | 3,296 | 4 |
| muskrat | 3,253 | 52 | 3,201 | match_level_no_match | 0 | 0 | 3,253 | 4 |
| howler monkey genus | 3,206 | 169 | 3,037 | match_level_no_match | 0 | 0 | 3,206 | 4 |
| opossum family | 3,199 | 697 | 2,502 | match_level_no_match | 0 | 0 | 3,199 | 4 |
| plains zebra | 2,858 | 2,451 | 407 | match_level_no_match | 0 | 337 | 3,195 | 4 |
| bighorn sheep | 3,167 | 535 | 2,632 | match_level_order | 0 | 0 | 3,167 | 4 |
| nutria | 3,150 | 923 | 2,227 | match_level_no_match | 0 | 0 | 3,150 | 4 |
| european hare | 3,087 | 603 | 2,484 | match_level_no_match | 0 | 0 | 3,087 | 4 |
| golden mantled ground squirrel | 2,819 | 631 | 2,188 | match_level_no_match | 0 | 0 | 2,819 | 4 |
| north american river otter | 2,772 | 459 | 2,313 | match_level_no_match | 0 | 0 | 2,772 | 4 |
| eastern grey kangaroo | 2,680 | 618 | 2,062 | match_level_no_match | 0 | 0 | 2,680 | 4 |
| wild boar | 2,498 | 930 | 1,568 | match_level_no_match | 0 | 104 | 2,602 | 4 |
| domestic horse | 2,467 | 1,215 | 1,252 | match_level_class | 0 | 83 | 2,550 | 4 |
| baboon genus | 2,486 | 881 | 1,605 | match_level_no_match | 0 | 0 | 2,486 | 4 |
| domestic cattle | 2,148 | 1,566 | 582 | match_level_no_match | 0 | 211 | 2,359 | 4 |
| american bison | 2,320 | 1,105 | 1,215 | family_mismatch_high_confidence | 0 | 0 | 2,320 | 4 |
| pronghorn | 2,316 | 1,065 | 1,251 | match_level_no_match | 0 | 0 | 2,316 | 4 |
| impala | 2,288 | 1,563 | 725 | match_level_no_match | 0 | 0 | 2,288 | 4 |
| white-nosed coati | 2,269 | 729 | 1,540 | match_level_no_match | 0 | 0 | 2,269 | 4 |
| mongoose family | 2,257 | 118 | 2,139 | match_level_no_match | 0 | 0 | 2,257 | 4 |
| beaver genus | 2,162 | 162 | 2,000 | match_level_no_match | 0 | 0 | 2,162 | 4 |
| llama genus | 2,085 | 125 | 1,960 | match_level_order | 0 | 0 | 2,085 | 4 |
| african buffalo | 2,051 | 718 | 1,333 | family_mismatch_high_confidence | 0 | 0 | 2,051 | 4 |
| domestic dog | 2,019 | 935 | 1,084 | match_level_no_match | 0 | 0 | 2,019 | 4 |
| western gray squirrel | 1,957 | 962 | 995 | match_level_no_match | 0 | 0 | 1,957 | 4 |
| leopard | 1,186 | 646 | 540 | match_level_no_match | 0 | 748 | 1,934 | 4 |
| north american porcupine | 1,914 | 481 | 1,433 | match_level_no_match | 0 | 0 | 1,914 | 4 |
| yellow-bellied marmot | 1,898 | 703 | 1,195 | match_level_no_match | 0 | 0 | 1,898 | 4 |
| bobcat | 1,861 | 899 | 962 | match_level_no_match | 0 | 0 | 1,861 | 4 |
| agouti genus | 1,853 | 670 | 1,183 | match_level_no_match | 0 | 0 | 1,853 | 4 |
| cebus species | 1,819 | 67 | 1,751 | match_level_no_match | 1 | 0 | 1,819 | 4 |
| arizona black-tailed prairie dog | 1,800 | 827 | 973 | match_level_no_match | 0 | 0 | 1,800 | 4 |
| weasel species | 1,795 | 88 | 1,707 | match_level_no_match | 0 | 0 | 1,795 | 4 |
| koala | 1,774 | 9 | 1,765 | match_level_no_match | 0 | 5 | 1,779 | 4 |
| elephant seal | 1,768 | 0 | 1,768 | match_level_no_match | 0 | 0 | 1,768 | 4 |
| capybara | 1,767 | 559 | 1,208 | match_level_no_match | 0 | 0 | 1,767 | 4 |
| short-beaked echidna | 1,767 | 1,092 | 675 | match_level_no_match | 0 | 0 | 1,767 | 4 |
| callithrix species | 1,736 | 3 | 1,733 | match_level_no_match | 0 | 0 | 1,736 | 4 |
| greater kudu | 1,732 | 577 | 1,155 | match_level_order | 0 | 0 | 1,732 | 4 |
| lycalopex species | 1,731 | 549 | 1,182 | family_mismatch_high_confidence | 0 | 0 | 1,731 | 4 |
| alpine marmot | 1,716 | 340 | 1,376 | match_level_no_match | 0 | 0 | 1,716 | 4 |
| red deer | 1,697 | 828 | 869 | match_level_no_match | 0 | 0 | 1,697 | 4 |
| vervet monkey | 1,635 | 656 | 979 | match_level_no_match | 0 | 0 | 1,635 | 4 |
| common wildebeest | 1,625 | 895 | 730 | match_level_no_match | 0 | 0 | 1,625 | 4 |
| common fallow deer | 1,589 | 439 | 1,150 | family_mismatch_high_confidence | 0 | 0 | 1,589 | 4 |
| domestic sheep | 1,375 | 480 | 894 | match_level_no_match | 1 | 187 | 1,562 | 4 |
| common warthog | 1,553 | 914 | 639 | match_level_no_match | 0 | 0 | 1,553 | 4 |
| brown bear | 1,523 | 557 | 966 | match_level_no_match | 0 | 0 | 1,523 | 4 |
| brown-throated sloth | 1,518 | 0 | 1,518 | match_level_no_match | 0 | 0 | 1,518 | 4 |
| rhinoceros family | 1,512 | 41 | 1,471 | match_level_no_match | 0 | 0 | 1,512 | 4 |
| pikas genus | 1,484 | 31 | 1,453 | match_level_no_match | 0 | 0 | 1,484 | 3 |
| waterbuck | 1,444 | 437 | 1,007 | match_level_order | 0 | 0 | 1,444 | 3 |
| hippopotamus | 1,319 | 298 | 1,021 | match_level_no_match | 0 | 120 | 1,439 | 3 |
| ateles species | 1,437 | 38 | 1,399 | match_level_no_match | 0 | 0 | 1,437 | 3 |
| cercopithecus species | 1,420 | 337 | 1,083 | match_level_no_match | 0 | 0 | 1,420 | 3 |
| collared peccary | 1,419 | 521 | 898 | match_level_no_match | 0 | 0 | 1,419 | 3 |
| sea otter | 1,404 | 0 | 1,404 | match_level_no_match | 0 | 0 | 1,404 | 3 |
| swamp wallaby | 1,377 | 420 | 957 | match_level_no_match | 0 | 0 | 1,377 | 3 |
| rock hyrax | 1,336 | 7 | 1,329 | match_level_class | 0 | 0 | 1,336 | 3 |
| saimiri species | 1,327 | 138 | 1,189 | match_level_no_match | 0 | 0 | 1,327 | 3 |
| asian elephant | 1,301 | 164 | 1,134 | family_mismatch_high_confidence | 3 | 17 | 1,318 | 3 |
| martes species | 1,298 | 145 | 1,153 | match_level_no_match | 0 | 0 | 1,298 | 3 |
| alpine ibex | 1,294 | 418 | 875 | match_level_no_match | 1 | 0 | 1,294 | 3 |
| domestic donkey | 1,200 | 325 | 874 | match_level_class | 1 | 73 | 1,273 | 3 |
| northern chamois | 1,269 | 315 | 954 | match_level_no_match | 0 | 0 | 1,269 | 3 |
| spotted hyaena | 1,233 | 640 | 593 | match_level_no_match | 0 | 0 | 1,233 | 3 |
| cheetah | 1,041 | 728 | 313 | match_level_no_match | 0 | 135 | 1,176 | 3 |
| chital | 1,169 | 263 | 906 | family_mismatch_high_confidence | 0 | 0 | 1,169 | 3 |
| sika deer | 1,136 | 354 | 782 | family_mismatch_high_confidence | 0 | 0 | 1,136 | 3 |
| grey fox | 1,116 | 425 | 691 | match_level_no_match | 0 | 0 | 1,116 | 3 |
| tiger | 849 | 540 | 309 | match_level_no_match | 0 | 258 | 1,107 | 3 |
| saguinus species | 1,065 | 5 | 1,060 | match_level_no_match | 0 | 0 | 1,065 | 3 |
| muntjac genus | 1,049 | 250 | 799 | match_level_no_match | 0 | 0 | 1,049 | 3 |
| south american coati | 1,014 | 411 | 603 | match_level_no_match | 0 | 0 | 1,014 | 3 |
| common eland | 962 | 385 | 577 | match_level_no_match | 0 | 0 | 962 | 3 |
| nyala | 957 | 277 | 680 | low_speciesnet_confidence | 0 | 0 | 957 | 3 |
| steenbok | 957 | 167 | 790 | match_level_no_match | 0 | 0 | 957 | 3 |
| african wild dog | 907 | 229 | 678 | match_level_class | 0 | 0 | 907 | 3 |
| eulemur species | 880 | 2 | 878 | match_level_no_match | 0 | 0 | 880 | 3 |
| hartebeest | 877 | 386 | 491 | match_level_no_match | 0 | 0 | 877 | 3 |
| sambar | 851 | 274 | 577 | family_mismatch_high_confidence | 0 | 0 | 851 | 3 |
| reindeer | 835 | 105 | 730 | match_level_order | 0 | 0 | 835 | 3 |
| blesbok | 823 | 185 | 638 | low_speciesnet_confidence | 0 | 0 | 823 | 3 |
| gemsbok | 818 | 438 | 380 | low_speciesnet_confidence | 0 | 0 | 818 | 3 |
| mountain goat | 793 | 100 | 693 | match_level_no_match | 0 | 0 | 793 | 3 |
| golden jackal | 781 | 304 | 477 | low_speciesnet_confidence | 0 | 0 | 781 | 3 |
| springbok | 759 | 247 | 512 | match_level_no_match | 0 | 0 | 759 | 3 |
| common wombat | 614 | 136 | 477 | match_level_class | 1 | 105 | 719 | 3 |
| klipspringer | 701 | 68 | 633 | match_level_no_match | 0 | 0 | 701 | 3 |
| domestic water buffalo | 694 | 57 | 637 | family_mismatch_high_confidence | 0 | 0 | 694 | 3 |
| leaf monkeys genus | 693 | 74 | 619 | match_level_no_match | 0 | 0 | 693 | 3 |
| hoffmann's two-toed sloth | 667 | 0 | 667 | match_level_no_match | 0 | 0 | 667 | 3 |
| striped skunk | 666 | 101 | 565 | match_level_no_match | 0 | 0 | 666 | 3 |
| colobus species | 663 | 117 | 546 | match_level_no_match | 0 | 0 | 663 | 3 |
| grey wolf | 550 | 244 | 306 | match_level_no_match | 0 | 104 | 654 | 3 |
| gorilla species | 550 | 82 | 468 | match_level_class | 0 | 59 | 609 | 3 |
| jaguar | 561 | 436 | 125 | match_level_no_match | 0 | 0 | 561 | 3 |
| nilgai | 549 | 64 | 485 | match_level_no_match | 0 | 0 | 549 | 3 |
| bornean orangutan | 423 | 154 | 269 | match_level_no_match | 0 | 123 | 546 | 3 |
| giraffe | 324 | 260 | 64 | match_level_no_match | 0 | 220 | 544 | 3 |
| grant's gazelle | 528 | 294 | 234 | family_mismatch_high_confidence | 0 | 0 | 528 | 3 |
| reedbuck genus | 514 | 65 | 449 | match_level_order | 0 | 0 | 514 | 3 |
| mountain zebra | 483 | 407 | 76 | match_level_no_match | 0 | 0 | 483 | 2 |
| japanese macaque | 481 | 126 | 355 | low_speciesnet_confidence | 0 | 0 | 481 | 2 |
| common duiker | 472 | 66 | 406 | match_level_order | 0 | 0 | 472 | 2 |
| eurasian badger | 460 | 76 | 384 | match_level_no_match | 0 | 0 | 460 | 2 |
| puma | 452 | 181 | 271 | match_level_no_match | 0 | 0 | 452 | 2 |
| ring-tailed lemur | 452 | 0 | 452 | match_level_class | 0 | 0 | 452 | 2 |
| thomson's gazelle | 452 | 337 | 115 | match_level_no_match | 0 | 0 | 452 | 2 |
| blackbuck | 448 | 0 | 448 | match_level_no_match | 0 | 0 | 448 | 2 |
| giant otter | 442 | 112 | 330 | match_level_no_match | 0 | 0 | 442 | 2 |
| eurasian otter | 434 | 26 | 408 | match_level_no_match | 0 | 0 | 434 | 2 |
| sable antelope | 426 | 97 | 329 | family_mismatch_high_confidence | 0 | 0 | 426 | 2 |
| dromedary camel | 420 | 108 | 309 | match_level_order | 3 | 0 | 420 | 2 |
| meerkat | 412 | 1 | 411 | match_level_no_match | 0 | 0 | 412 | 2 |
| dingo | 16 | 10 | 6 | low_speciesnet_confidence | 0 | 387 | 403 | 2 |
| bushbuck | 391 | 181 | 210 | match_level_order | 0 | 0 | 391 | 2 |
| american badger | 389 | 58 | 330 | match_level_no_match | 1 | 0 | 389 | 2 |
| red kangaroo | 344 | 35 | 309 | match_level_class | 0 | 36 | 380 | 2 |
| glaucomys species | 372 | 12 | 360 | match_level_no_match | 0 | 0 | 372 | 2 |
| quokka | 366 | 18 | 348 | match_level_class | 0 | 0 | 366 | 2 |
| snow leopard | 53 | 31 | 22 | match_level_no_match | 0 | 311 | 364 | 2 |
| giant anteater | 355 | 114 | 241 | match_level_no_match | 0 | 0 | 355 | 2 |
| domestic goat | 183 | 74 | 109 | family_mismatch_high_confidence | 0 | 154 | 337 | 2 |
| chimpanzee | 335 | 30 | 305 | match_level_no_match | 0 | 0 | 335 | 2 |
| baird's tapir | 323 | 61 | 262 | match_level_no_match | 0 | 0 | 323 | 2 |
| tayra | 317 | 66 | 251 | match_level_no_match | 0 | 0 | 317 | 2 |
| red panda | 197 | 36 | 161 | match_level_no_match | 0 | 100 | 297 | 2 |
| lowland tapir | 294 | 100 | 194 | match_level_class | 0 | 0 | 294 | 2 |
| kob | 289 | 79 | 210 | family_mismatch_high_confidence | 0 | 0 | 289 | 2 |
| roan antelope | 278 | 43 | 235 | match_level_no_match | 0 | 0 | 278 | 2 |
| leopardus species | 271 | 57 | 214 | match_level_no_match | 0 | 0 | 271 | 2 |
| european bison | 265 | 82 | 183 | family_mismatch_high_confidence | 0 | 0 | 265 | 2 |
| nine-banded armadillo | 240 | 110 | 130 | match_level_no_match | 0 | 0 | 240 | 2 |
| american mink | 194 | 16 | 178 | match_level_no_match | 0 | 45 | 239 | 2 |
| old world porcupine family | 238 | 55 | 183 | match_level_no_match | 0 | 0 | 238 | 2 |
| raccoon dog | 232 | 3 | 229 | match_level_no_match | 0 | 0 | 232 | 2 |
| bat-eared fox | 228 | 18 | 210 | low_speciesnet_confidence | 0 | 0 | 228 | 2 |
| genet genus | 227 | 10 | 216 | match_level_no_match | 1 | 0 | 227 | 2 |
| kirk's dik-dik | 215 | 120 | 95 | match_level_order | 0 | 0 | 215 | 2 |
| grevy's zebra | 209 | 185 | 24 | match_level_no_match | 0 | 0 | 209 | 2 |
| dhole | 202 | 46 | 156 | match_level_class | 0 | 0 | 202 | 2 |
| spectacled bear | 201 | 21 | 180 | match_level_no_match | 0 | 0 | 201 | 2 |
| eurasian lynx | 200 | 107 | 93 | match_level_no_match | 0 | 0 | 200 | 2 |
| serval | 196 | 85 | 111 | family_mismatch_high_confidence | 0 | 0 | 196 | 2 |
| ringtail | 186 | 10 | 176 | match_level_no_match | 0 | 0 | 186 | 2 |
| gerenuk | 180 | 71 | 109 | family_mismatch_high_confidence | 0 | 0 | 180 | 2 |
| patas monkey | 179 | 12 | 167 | match_level_no_match | 0 | 0 | 179 | 2 |
| giant panda | 125 | 41 | 84 | match_level_no_match | 0 | 47 | 172 | 2 |
| caracal | 171 | 16 | 155 | match_level_no_match | 0 | 0 | 171 | 2 |
| black wildebeest | 170 | 51 | 119 | family_mismatch_high_confidence | 0 | 0 | 170 | 2 |
| canada lynx | 166 | 47 | 119 | match_level_no_match | 0 | 0 | 166 | 2 |
| kinkajou | 160 | 2 | 158 | match_level_no_match | 0 | 0 | 160 | 2 |
| walrus | 158 | 0 | 158 | match_level_no_match | 0 | 0 | 158 | 2 |
| water deer | 152 | 16 | 136 | family_mismatch_high_confidence | 0 | 0 | 152 | 2 |
| fisher | 131 | 32 | 99 | match_level_no_match | 0 | 0 | 131 | 2 |
| black-backed jackal | 127 | 72 | 55 | low_speciesnet_confidence | 0 | 0 | 127 | 2 |
| yak | 127 | 44 | 83 | match_level_class | 0 | 0 | 127 | 2 |
| wild cat | 126 | 54 | 72 | match_level_no_match | 0 | 0 | 126 | 2 |
| cephalophus species | 124 | 17 | 107 | match_level_order | 0 | 0 | 124 | 2 |
| sloth bear | 120 | 15 | 105 | family_mismatch_high_confidence | 0 | 0 | 120 | 2 |
| callicebus genus | 119 | 0 | 119 | match_level_no_match | 0 | 0 | 119 | 2 |
| leopard cat | 105 | 27 | 78 | match_level_no_match | 0 | 0 | 105 | 2 |
| striped hyaena | 105 | 22 | 83 | match_level_order | 0 | 0 | 105 | 2 |
| asiatic black bear | 104 | 35 | 69 | match_level_no_match | 0 | 0 | 104 | 2 |
| ocelot | 99 | 43 | 56 | match_level_no_match | 0 | 0 | 99 | 1 |
| honey badger | 97 | 6 | 91 | match_level_no_match | 0 | 0 | 97 | 1 |
| sun bear | 83 | 18 | 65 | low_speciesnet_confidence | 0 | 14 | 97 | 1 |
| maned wolf | 88 | 35 | 53 | match_level_no_match | 0 | 0 | 88 | 1 |
| asiatic wild ass | 84 | 7 | 77 | match_level_class | 0 | 0 | 84 | 1 |
| fossa | 79 | 6 | 73 | low_speciesnet_confidence | 0 | 0 | 79 | 1 |
| wolverine | 69 | 10 | 59 | match_level_no_match | 0 | 0 | 69 | 1 |
| brown hyaena | 68 | 5 | 63 | low_speciesnet_confidence | 0 | 0 | 68 | 1 |
| red brocket | 62 | 18 | 44 | family_mismatch_high_confidence | 0 | 0 | 62 | 1 |
| pinniped clade | 55 | 0 | 55 | match_level_no_match | 0 | 0 | 55 | 1 |
| aardvark | 54 | 0 | 54 | match_level_class | 0 | 0 | 54 | 1 |
| mangabeys genus | 54 | 12 | 42 | match_level_no_match | 0 | 0 | 54 | 1 |
| red river hog | 53 | 16 | 37 | low_speciesnet_confidence | 0 | 0 | 53 | 1 |
| pangolin family | 52 | 2 | 50 | match_level_no_match | 0 | 0 | 52 | 1 |
| saiga | 50 | 0 | 50 | match_level_no_match | 0 | 0 | 50 | 1 |
| aardwolf | 47 | 9 | 38 | family_mismatch_high_confidence | 0 | 0 | 47 | 1 |
| bongo | 39 | 11 | 28 | family_mismatch_high_confidence | 0 | 0 | 39 | 1 |
| spilogale species | 37 | 2 | 35 | match_level_no_match | 0 | 0 | 37 | 1 |
| binturong | 36 | 0 | 36 | match_level_no_match | 0 | 0 | 36 | 1 |
| red-necked wallaby | 30 | 15 | 15 | match_level_class | 0 | 0 | 30 | 1 |
| aye-aye | 29 | 0 | 29 | match_level_no_match | 0 | 0 | 29 | 1 |
| african civet | 28 | 6 | 22 | match_level_order | 0 | 0 | 28 | 1 |
| clouded leopard | 25 | 0 | 25 | family_mismatch_high_confidence | 0 | 0 | 25 | 1 |
| drill | 23 | 2 | 21 | low_speciesnet_confidence | 0 | 0 | 23 | 1 |
| malay tapir | 21 | 9 | 12 | match_level_no_match | 0 | 0 | 21 | 1 |
| hog badger genus | 18 | 3 | 15 | match_level_class | 0 | 0 | 18 | 1 |
| grevys zebra | 16 | 0 | 16 | not_in_225_classes | 0 | 0 | 16 | 1 |
| domestic pig | 15 | 10 | 5 | match_level_no_match | 0 | 0 | 15 | 1 |
| giant armadillo | 11 | 2 | 9 | match_level_no_match | 0 | 0 | 11 | 1 |
| kirks dik-dik | 11 | 0 | 11 | not_in_225_classes | 0 | 0 | 11 | 1 |
| bairds tapir | 9 | 0 | 9 | not_in_225_classes | 0 | 0 | 9 | 1 |
| thomsons gazelle | 9 | 0 | 9 | not_in_225_classes | 0 | 0 | 9 | 1 |
| grants gazelle | 7 | 0 | 7 | not_in_225_classes | 0 | 0 | 7 | 1 |
| hoffmanns two-toed sloth | 7 | 0 | 7 | not_in_225_classes | 0 | 0 | 7 | 1 |
| mouflon | 6 | 2 | 4 | family_mismatch_high_confidence | 0 | 0 | 6 | 1 |
| human | 5 | 0 | 5 | match_level_no_match | 0 | 0 | 5 | 1 |
