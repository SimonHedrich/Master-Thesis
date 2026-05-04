# SpeciesNet Filter Report

**Thresholds:** md_conf≥0.5  sn_score≥0.3  family_fail≥0.5

**Mode:** STATS ONLY

---

## all sources combined — 465,130 records

| | Count | % |
|---|---:|---:|
| Pass | 158,667 | 34.1% |
| Fail | 306,463 | 65.9% |

### Fail Reasons

| Reason | Count | % |
|---|---:|---:|
| primary_crop_too_small | 2,556 | 0.5% |
| low_speciesnet_confidence | 66,769 | 14.4% |
| not_in_225_classes | 6,218 | 1.3% |
| family_mismatch_high_confidence | 19,527 | 4.2% |
| match_level_order | 17,408 | 3.7% |
| match_level_class | 29,969 | 6.4% |
| match_level_no_match | 164,016 | 35.3% |

### Match Levels (389,587 images with valid classification)

| Level | Count | % |
|---|---:|---:|
| species | 129,888 | 33.3% |
| genus | 16,355 | 4.2% |
| family | 31,951 | 8.2% |
| order | 17,408 | 4.5% |
| class | 29,969 | 7.7% |
| no_match | 164,016 | 42.1% |

**Multi-animal images:** 80,538 (17.3%)

**prob\_225\_sum** (389,587 images with valid classification): mean=0.552  median=0.728  p10=0.000  p90=0.977  zeros=69097

### Per-Class Breakdown (232 classes)

| Class | Pre-filter | SN Input | Pass | Pass% | Top Fail Reason |
|---|---:|---:|---:|---:|---|
| aardvark | 266 | 53 | 0 | 0.0% | match_level_class=21 |
| aardwolf | 88 | 47 | 9 | 19.1% | family_mismatch_high_confidence=9 |
| african buffalo | 2,240 | 2,050 | 717 | 35.0% | family_mismatch_high_confidence=787 |
| african civet | 164 | 27 | 6 | 22.2% | match_level_order=7 |
| african elephant | 4,620 | 4,335 | 3,301 | 76.1% | match_level_no_match=661 |
| african wild dog | 1,062 | 905 | 227 | 25.1% | match_level_class=183 |
| agouti genus | 2,143 | 1,853 | 670 | 36.2% | match_level_no_match=496 |
| alpine ibex | 1,406 | 1,293 | 418 | 32.3% | match_level_no_match=317 |
| alpine marmot | 2,325 | 1,715 | 339 | 19.8% | match_level_no_match=831 |
| american badger | 884 | 388 | 58 | 14.9% | match_level_no_match=194 |
| american bison | 2,622 | 2,319 | 1,104 | 47.6% | family_mismatch_high_confidence=592 |
| american black bear | 4,129 | 3,433 | 1,288 | 37.5% | match_level_no_match=1451 |
| american mink | 1,623 | 829 | 60 | 7.2% | match_level_no_match=344 |
| arizona black-tailed prairie dog | 1,956 | 1,800 | 827 | 45.9% | match_level_no_match=649 |
| asian elephant | 2,070 | 1,524 | 178 | 11.7% | family_mismatch_high_confidence=779 |
| asiatic black bear | 244 | 103 | 34 | 33.0% | match_level_no_match=21 |
| asiatic wild ass | 179 | 84 | 7 | 8.3% | match_level_class=41 |
| ateles species | 1,746 | 1,437 | 38 | 2.6% | match_level_no_match=652 |
| aye-aye | 61 | 29 | 0 | 0.0% | match_level_no_match=13 |
| baboon genus | 2,759 | 2,486 | 881 | 35.4% | match_level_no_match=627 |
| baird's tapir | 539 | 323 | 61 | 18.9% | match_level_no_match=124 |
| bairds tapir | 9 | 9 | 0 | 0.0% | not_in_225_classes=9 |
| bat-eared fox | 402 | 228 | 18 | 7.9% | low_speciesnet_confidence=93 |
| beaver genus | 3,907 | 2,162 | 162 | 7.5% | match_level_no_match=1151 |
| bighorn sheep | 3,458 | 3,167 | 535 | 16.9% | match_level_order=1089 |
| binturong | 65 | 36 | 0 | 0.0% | match_level_no_match=18 |
| black wildebeest | 293 | 170 | 51 | 30.0% | family_mismatch_high_confidence=52 |
| black-backed jackal | 155 | 127 | 72 | 56.7% | low_speciesnet_confidence=25 |
| blackbuck | 596 | 448 | 0 | 0.0% | match_level_no_match=350 |
| blesbok | 1,140 | 823 | 185 | 22.5% | low_speciesnet_confidence=226 |
| bobcat | 2,474 | 1,861 | 899 | 48.3% | match_level_no_match=732 |
| bongo | 49 | 39 | 11 | 28.2% | family_mismatch_high_confidence=10 |
| bornean orangutan | 1,986 | 1,089 | 276 | 25.3% | match_level_no_match=298 |
| brown bear | 2,459 | 1,521 | 555 | 36.5% | match_level_no_match=385 |
| brown hyaena | 162 | 68 | 5 | 7.4% | low_speciesnet_confidence=20 |
| brown-throated sloth | 1,926 | 1,518 | 0 | 0.0% | match_level_no_match=1109 |
| bushbuck | 468 | 391 | 181 | 46.3% | match_level_order=100 |
| california ground squirrel | 5,873 | 5,627 | 1,201 | 21.3% | match_level_no_match=2627 |
| callicebus genus | 129 | 119 | 0 | 0.0% | match_level_no_match=67 |
| callithrix species | 1,875 | 1,736 | 3 | 0.2% | match_level_no_match=1013 |
| canada lynx | 493 | 166 | 47 | 28.3% | match_level_no_match=61 |
| capybara | 1,995 | 1,767 | 559 | 31.6% | match_level_no_match=421 |
| caracal | 386 | 171 | 16 | 9.4% | match_level_no_match=68 |
| cebus species | 1,994 | 1,816 | 67 | 3.7% | match_level_no_match=788 |
| cephalophus species | 230 | 124 | 17 | 13.7% | match_level_order=51 |
| cercopithecus species | 1,757 | 1,420 | 337 | 23.7% | match_level_no_match=663 |
| cheetah | 1,540 | 1,239 | 863 | 69.7% | match_level_no_match=238 |
| chimpanzee | 518 | 335 | 30 | 9.0% | match_level_no_match=120 |
| chipmunk genus | 7,457 | 6,933 | 1,752 | 25.3% | match_level_no_match=2916 |
| chital | 1,560 | 1,169 | 263 | 22.5% | family_mismatch_high_confidence=526 |
| clouded leopard | 57 | 25 | 0 | 0.0% | family_mismatch_high_confidence=15 |
| collared peccary | 1,816 | 1,419 | 521 | 36.7% | match_level_no_match=372 |
| colobus species | 807 | 663 | 117 | 17.6% | match_level_no_match=413 |
| common duiker | 1,007 | 472 | 66 | 14.0% | match_level_order=192 |
| common eland | 1,283 | 962 | 385 | 40.0% | match_level_no_match=195 |
| common fallow deer | 1,874 | 1,584 | 436 | 27.5% | family_mismatch_high_confidence=479 |
| common warthog | 1,624 | 1,553 | 914 | 58.9% | match_level_no_match=271 |
| common wildebeest | 1,790 | 1,625 | 895 | 55.1% | match_level_no_match=233 |
| common wombat | 2,431 | 1,383 | 240 | 17.4% | match_level_class=440 |
| cottontail rabbits genus | 8,304 | 7,757 | 2,214 | 28.5% | match_level_no_match=2271 |
| coyote | 8,437 | 6,865 | 3,081 | 44.9% | match_level_no_match=2413 |
| cricetidae family | 6,395 | 4,915 | 24 | 0.5% | match_level_no_match=3154 |
| dhole | 283 | 202 | 46 | 22.8% | match_level_class=59 |
| dingo | 1,460 | 781 | 397 | 50.8% | low_speciesnet_confidence=126 |
| domestic cat | 8,849 | 7,298 | 3,867 | 53.0% | match_level_no_match=1916 |
| domestic cattle | 4,077 | 2,394 | 1,777 | 74.2% | match_level_no_match=386 |
| domestic dog | 2,795 | 2,019 | 935 | 46.3% | match_level_no_match=405 |
| domestic donkey | 2,172 | 1,395 | 397 | 28.5% | match_level_class=568 |
| domestic goat | 689 | 501 | 226 | 45.1% | family_mismatch_high_confidence=108 |
| domestic horse | 4,431 | 2,606 | 1,298 | 49.8% | match_level_class=744 |
| domestic pig | 17 | 15 | 10 | 66.7% | match_level_no_match=2 |
| domestic sheep | 2,877 | 1,687 | 666 | 39.5% | match_level_no_match=387 |
| domestic water buffalo | 1,141 | 694 | 57 | 8.2% | family_mismatch_high_confidence=429 |
| drill | 35 | 23 | 2 | 8.7% | low_speciesnet_confidence=9 |
| dromedary camel | 677 | 417 | 108 | 25.9% | match_level_order=110 |
| eared seals | 10,285 | 9,258 | 0 | 0.0% | match_level_no_match=7720 |
| eastern cottontail | 12,264 | 11,456 | 3,567 | 31.1% | match_level_no_match=3744 |
| eastern fox squirrel | 13,014 | 12,436 | 6,456 | 51.9% | match_level_no_match=4003 |
| eastern gray squirrel | 32,925 | 31,095 | 15,644 | 50.3% | match_level_no_match=10030 |
| eastern grey kangaroo | 2,947 | 2,680 | 618 | 23.1% | match_level_no_match=837 |
| elephant seal | 1,882 | 1,768 | 0 | 0.0% | match_level_no_match=1479 |
| elk | 6,232 | 5,517 | 3,507 | 63.6% | match_level_no_match=1140 |
| eulemur species | 1,047 | 880 | 2 | 0.2% | match_level_no_match=378 |
| eurasian badger | 2,417 | 460 | 76 | 16.5% | match_level_no_match=156 |
| eurasian lynx | 507 | 200 | 107 | 53.5% | match_level_no_match=46 |
| eurasian otter | 1,147 | 434 | 26 | 6.0% | match_level_no_match=217 |
| eurasian red squirrel | 10,554 | 9,927 | 2,663 | 26.8% | match_level_no_match=4355 |
| european bison | 443 | 265 | 82 | 30.9% | family_mismatch_high_confidence=105 |
| european hare | 3,423 | 3,086 | 602 | 19.5% | match_level_no_match=1324 |
| european rabbit | 6,468 | 5,631 | 1,016 | 18.0% | match_level_no_match=2179 |
| european roe deer | 6,915 | 6,011 | 1,547 | 25.7% | match_level_no_match=1962 |
| fisher | 605 | 131 | 32 | 24.4% | match_level_no_match=44 |
| fossa | 111 | 79 | 6 | 7.6% | low_speciesnet_confidence=23 |
| gemsbok | 1,149 | 818 | 438 | 53.5% | low_speciesnet_confidence=117 |
| genet genus | 1,068 | 226 | 10 | 4.4% | match_level_no_match=91 |
| gerenuk | 220 | 180 | 71 | 39.4% | family_mismatch_high_confidence=63 |
| giant anteater | 606 | 355 | 114 | 32.1% | match_level_no_match=125 |
| giant armadillo | 40 | 11 | 2 | 18.2% | match_level_no_match=6 |
| giant otter | 562 | 442 | 112 | 25.3% | match_level_no_match=182 |
| giant panda | 941 | 414 | 88 | 21.3% | match_level_class=94 |
| giraffe | 876 | 648 | 479 | 73.9% | match_level_no_match=94 |
| glaucomys species | 1,155 | 372 | 12 | 3.2% | match_level_no_match=164 |
| golden jackal | 1,252 | 780 | 303 | 38.8% | low_speciesnet_confidence=155 |
| golden mantled ground squirrel | 2,937 | 2,819 | 631 | 22.4% | match_level_no_match=921 |
| gorilla species | 2,113 | 1,321 | 141 | 10.7% | match_level_class=581 |
| grant's gazelle | 615 | 528 | 294 | 55.7% | family_mismatch_high_confidence=120 |
| grants gazelle | 9 | 7 | 0 | 0.0% | not_in_225_classes=7 |
| greater kudu | 1,830 | 1,732 | 577 | 33.3% | match_level_order=366 |
| grevy's zebra | 245 | 209 | 185 | 88.5% | match_level_no_match=13 |
| grevys zebra | 17 | 16 | 0 | 0.0% | not_in_225_classes=16 |
| grey fox | 1,898 | 1,116 | 425 | 38.1% | match_level_no_match=408 |
| grey wolf | 2,249 | 782 | 348 | 44.5% | match_level_no_match=179 |
| hares and jackrabbits genus | 5,515 | 4,836 | 987 | 20.4% | match_level_no_match=1679 |
| hartebeest | 1,084 | 877 | 386 | 44.0% | match_level_no_match=124 |
| hedgehog family | 4,533 | 3,289 | 1 | 0.0% | match_level_no_match=2468 |
| hippopotamus | 1,928 | 1,610 | 418 | 26.0% | match_level_no_match=616 |
| hoffmann's two-toed sloth | 988 | 667 | 0 | 0.0% | match_level_no_match=429 |
| hoffmanns two-toed sloth | 8 | 7 | 0 | 0.0% | not_in_225_classes=7 |
| hog badger genus | 37 | 18 | 3 | 16.7% | match_level_class=5 |
| honey badger | 251 | 97 | 6 | 6.2% | match_level_no_match=39 |
| howler monkey genus | 3,355 | 3,206 | 169 | 5.3% | match_level_no_match=1659 |
| human | 107 | 5 | 0 | 0.0% | match_level_no_match=4 |
| impala | 2,409 | 2,286 | 1,562 | 68.3% | match_level_no_match=324 |
| jaguar | 933 | 561 | 436 | 77.7% | match_level_no_match=80 |
| japanese macaque | 590 | 481 | 126 | 26.2% | low_speciesnet_confidence=132 |
| kangaroo family | 7,111 | 6,135 | 1,280 | 20.9% | match_level_no_match=3303 |
| kinkajou | 436 | 160 | 2 | 1.2% | match_level_no_match=73 |
| kirk's dik-dik | 280 | 215 | 120 | 55.8% | match_level_order=40 |
| kirks dik-dik | 11 | 11 | 0 | 0.0% | not_in_225_classes=8 |
| klipspringer | 1,100 | 701 | 68 | 9.7% | match_level_no_match=206 |
| koala | 2,459 | 2,046 | 14 | 0.7% | match_level_no_match=1340 |
| kob | 334 | 289 | 79 | 27.3% | family_mismatch_high_confidence=73 |
| leaf monkeys genus | 825 | 693 | 74 | 10.7% | match_level_no_match=374 |
| leopard | 3,326 | 2,413 | 1,394 | 57.8% | match_level_no_match=502 |
| leopard cat | 286 | 105 | 27 | 25.7% | match_level_no_match=31 |
| leopardus species | 737 | 271 | 57 | 21.0% | match_level_no_match=83 |
| lion | 4,049 | 3,395 | 1,814 | 53.4% | match_level_no_match=688 |
| llama genus | 2,341 | 2,084 | 125 | 6.0% | match_level_order=931 |
| lowland tapir | 690 | 294 | 100 | 34.0% | match_level_class=83 |
| lycalopex species | 2,028 | 1,731 | 549 | 31.7% | family_mismatch_high_confidence=400 |
| macaque species | 9,139 | 7,557 | 2,305 | 30.5% | match_level_no_match=2137 |
| malay tapir | 45 | 21 | 9 | 42.9% | match_level_no_match=6 |
| maned wolf | 301 | 88 | 35 | 39.8% | match_level_no_match=17 |
| mangabeys genus | 110 | 54 | 12 | 22.2% | match_level_no_match=16 |
| martes species | 3,292 | 1,298 | 145 | 11.2% | match_level_no_match=558 |
| meerkat | 509 | 412 | 1 | 0.2% | match_level_no_match=140 |
| mongoose family | 2,563 | 2,257 | 118 | 5.2% | match_level_no_match=1430 |
| moose | 4,238 | 3,502 | 1,157 | 33.0% | match_level_no_match=1201 |
| mouflon | 10 | 6 | 2 | 33.3% | family_mismatch_high_confidence=2 |
| mountain goat | 1,183 | 793 | 100 | 12.6% | match_level_no_match=251 |
| mountain zebra | 612 | 483 | 407 | 84.3% | match_level_no_match=63 |
| mule deer | 20,322 | 18,106 | 12,707 | 70.2% | match_level_no_match=4145 |
| muntjac genus | 1,704 | 1,049 | 250 | 23.8% | match_level_no_match=298 |
| muridae family | 4,346 | 3,370 | 51 | 1.5% | match_level_no_match=2281 |
| muskrat | 3,698 | 3,253 | 52 | 1.6% | match_level_no_match=1698 |
| nilgai | 797 | 548 | 64 | 11.7% | match_level_no_match=137 |
| nine-banded armadillo | 640 | 240 | 110 | 45.8% | match_level_no_match=103 |
| north american porcupine | 2,348 | 1,914 | 481 | 25.1% | match_level_no_match=960 |
| north american river otter | 3,132 | 2,772 | 459 | 16.6% | match_level_no_match=1409 |
| northern chamois | 1,859 | 1,269 | 315 | 24.8% | match_level_no_match=439 |
| northern raccoon | 8,827 | 5,936 | 2,295 | 38.7% | match_level_no_match=2189 |
| nutria | 3,543 | 3,150 | 923 | 29.3% | match_level_no_match=1395 |
| nyala | 1,078 | 957 | 277 | 28.9% | low_speciesnet_confidence=205 |
| ocelot | 585 | 99 | 43 | 43.4% | match_level_no_match=19 |
| old world porcupine family | 1,854 | 238 | 55 | 23.1% | match_level_no_match=135 |
| opossum family | 5,035 | 3,199 | 697 | 21.8% | match_level_no_match=1870 |
| pangolin family | 168 | 52 | 2 | 3.8% | match_level_no_match=39 |
| patas monkey | 237 | 178 | 11 | 6.2% | match_level_no_match=62 |
| pikas genus | 1,852 | 1,484 | 31 | 2.1% | match_level_no_match=557 |
| pinniped clade | 138 | 55 | 0 | 0.0% | match_level_no_match=43 |
| plains zebra | 3,555 | 3,233 | 2,787 | 86.2% | match_level_no_match=295 |
| pronghorn | 2,498 | 2,316 | 1,065 | 46.0% | match_level_no_match=587 |
| puma | 2,326 | 452 | 181 | 40.0% | match_level_no_match=113 |
| quokka | 428 | 366 | 18 | 4.9% | match_level_class=143 |
| raccoon dog | 653 | 232 | 3 | 1.3% | match_level_no_match=68 |
| rattus genus | 5,463 | 3,954 | 65 | 1.6% | match_level_no_match=1858 |
| red brocket | 261 | 62 | 18 | 29.0% | family_mismatch_high_confidence=14 |
| red deer | 2,012 | 1,697 | 828 | 48.8% | match_level_no_match=393 |
| red fox | 10,758 | 8,913 | 3,696 | 41.5% | match_level_no_match=2720 |
| red kangaroo | 901 | 678 | 71 | 10.5% | match_level_class=236 |
| red panda | 635 | 543 | 136 | 25.0% | match_level_class=121 |
| red river hog | 83 | 49 | 12 | 24.5% | low_speciesnet_confidence=10 |
| red squirrel | 8,062 | 7,526 | 2,360 | 31.4% | match_level_no_match=2648 |
| red-necked wallaby | 49 | 30 | 15 | 50.0% | match_level_class=7 |
| reedbuck genus | 656 | 514 | 65 | 12.6% | match_level_order=217 |
| reindeer | 1,748 | 835 | 105 | 12.6% | match_level_order=209 |
| rhinoceros family | 2,002 | 1,510 | 40 | 2.6% | match_level_no_match=1261 |
| ring-tailed lemur | 534 | 452 | 0 | 0.0% | match_level_class=144 |
| ringtail | 929 | 186 | 10 | 5.4% | match_level_no_match=76 |
| roan antelope | 339 | 278 | 43 | 15.5% | match_level_no_match=71 |
| rock hyrax | 1,863 | 1,336 | 7 | 0.5% | match_level_class=486 |
| sable antelope | 525 | 426 | 97 | 22.8% | family_mismatch_high_confidence=154 |
| saguinus species | 1,287 | 1,065 | 5 | 0.5% | match_level_no_match=737 |
| saiga | 85 | 50 | 0 | 0.0% | match_level_no_match=41 |
| saimiri species | 1,700 | 1,327 | 138 | 10.4% | match_level_no_match=823 |
| sambar | 1,341 | 851 | 274 | 32.2% | family_mismatch_high_confidence=255 |
| sea otter | 2,131 | 1,819 | 0 | 0.0% | match_level_no_match=1338 |
| serval | 307 | 195 | 84 | 43.1% | family_mismatch_high_confidence=49 |
| short-beaked echidna | 1,974 | 1,767 | 1,092 | 61.8% | match_level_no_match=569 |
| sika deer | 1,531 | 1,136 | 354 | 31.2% | family_mismatch_high_confidence=375 |
| sloth bear | 215 | 120 | 15 | 12.5% | family_mismatch_high_confidence=47 |
| snow leopard | 1,373 | 754 | 342 | 45.4% | family_mismatch_high_confidence=174 |
| south american coati | 1,235 | 1,014 | 411 | 40.5% | match_level_no_match=261 |
| spectacled bear | 295 | 201 | 21 | 10.4% | match_level_no_match=83 |
| spilogale species | 236 | 37 | 2 | 5.4% | match_level_no_match=13 |
| spotted hyaena | 1,641 | 1,233 | 640 | 51.9% | match_level_no_match=266 |
| springbok | 1,080 | 759 | 247 | 32.5% | match_level_no_match=172 |
| squirrel family | 33,913 | 31,608 | 8,148 | 25.8% | match_level_no_match=16024 |
| steenbok | 1,160 | 957 | 167 | 17.5% | match_level_no_match=253 |
| striped hyaena | 168 | 105 | 22 | 21.0% | match_level_order=36 |
| striped skunk | 2,664 | 666 | 101 | 15.2% | match_level_no_match=389 |
| sun bear | 503 | 268 | 32 | 11.9% | family_mismatch_high_confidence=85 |
| swamp wallaby | 1,624 | 1,377 | 420 | 30.5% | match_level_no_match=536 |
| tayra | 526 | 317 | 66 | 20.8% | match_level_no_match=133 |
| thomson's gazelle | 545 | 452 | 337 | 74.6% | match_level_no_match=43 |
| thomsons gazelle | 12 | 9 | 0 | 0.0% | not_in_225_classes=8 |
| tiger | 1,704 | 1,155 | 797 | 69.0% | match_level_no_match=243 |
| unmatched | 15,209 | 7,715 | 0 | 0.0% | not_in_225_classes=6163 |
| vervet monkey | 1,745 | 1,635 | 656 | 40.1% | match_level_no_match=594 |
| walrus | 265 | 158 | 0 | 0.0% | match_level_no_match=128 |
| water deer | 239 | 151 | 15 | 9.9% | family_mismatch_high_confidence=48 |
| waterbuck | 1,633 | 1,444 | 437 | 30.3% | match_level_order=351 |
| weasel species | 2,546 | 1,795 | 88 | 4.9% | match_level_no_match=706 |
| western gray squirrel | 2,322 | 1,957 | 962 | 49.2% | match_level_no_match=682 |
| white-nosed coati | 2,479 | 2,269 | 729 | 32.1% | match_level_no_match=546 |
| white-tailed deer | 28,744 | 24,805 | 17,371 | 70.0% | match_level_no_match=5899 |
| wild boar | 4,131 | 2,799 | 1,034 | 36.9% | match_level_no_match=925 |
| wild cat | 492 | 126 | 54 | 42.9% | match_level_no_match=33 |
| wolverine | 222 | 69 | 10 | 14.5% | match_level_no_match=32 |
| woodchuck | 4,334 | 3,872 | 1,128 | 29.1% | match_level_no_match=1852 |
| yak | 185 | 127 | 44 | 34.6% | match_level_class=41 |
| yellow-bellied marmot | 2,047 | 1,898 | 703 | 37.0% | match_level_no_match=809 |

### Classes Ranked by Passed Images

| Rank | Class | Passed |
|---:|---|---:|
| 1 | white-tailed deer | 17,371 |
| 2 | eastern gray squirrel | 15,644 |
| 3 | mule deer | 12,707 |
| 4 | squirrel family | 8,148 |
| 5 | eastern fox squirrel | 6,456 |
| 6 | domestic cat | 3,867 |
| 7 | red fox | 3,696 |
| 8 | eastern cottontail | 3,567 |
| 9 | elk | 3,507 |
| 10 | african elephant | 3,301 |
| 11 | coyote | 3,081 |
| 12 | plains zebra | 2,787 |
| 13 | eurasian red squirrel | 2,663 |
| 14 | red squirrel | 2,360 |
| 15 | macaque species | 2,305 |
| 16 | northern raccoon | 2,295 |
| 17 | cottontail rabbits genus | 2,214 |
| 18 | lion | 1,814 |
| 19 | domestic cattle | 1,777 |
| 20 | chipmunk genus | 1,752 |
| 21 | impala | 1,562 |
| 22 | european roe deer | 1,547 |
| 23 | leopard | 1,394 |
| 24 | domestic horse | 1,298 |
| 25 | american black bear | 1,288 |
| 26 | kangaroo family | 1,280 |
| 27 | california ground squirrel | 1,201 |
| 28 | moose | 1,157 |
| 29 | woodchuck | 1,128 |
| 30 | american bison | 1,104 |
| 31 | short-beaked echidna | 1,092 |
| 32 | pronghorn | 1,065 |
| 33 | wild boar | 1,034 |
| 34 | european rabbit | 1,016 |
| 35 | hares and jackrabbits genus | 987 |
| 36 | western gray squirrel | 962 |
| 37 | domestic dog | 935 |
| 38 | nutria | 923 |
| 39 | common warthog | 914 |
| 40 | bobcat | 899 |
| 41 | common wildebeest | 895 |
| 42 | baboon genus | 881 |
| 43 | cheetah | 863 |
| 44 | red deer | 828 |
| 45 | arizona black-tailed prairie dog | 827 |
| 46 | tiger | 797 |
| 47 | white-nosed coati | 729 |
| 48 | african buffalo | 717 |
| 49 | yellow-bellied marmot | 703 |
| 50 | opossum family | 697 |
| 51 | agouti genus | 670 |
| 52 | domestic sheep | 666 |
| 53 | vervet monkey | 656 |
| 54 | spotted hyaena | 640 |
| 55 | golden mantled ground squirrel | 631 |
| 56 | eastern grey kangaroo | 618 |
| 57 | european hare | 602 |
| 58 | greater kudu | 577 |
| 59 | capybara | 559 |
| 60 | brown bear | 555 |
| 61 | lycalopex species | 549 |
| 62 | bighorn sheep | 535 |
| 63 | collared peccary | 521 |
| 64 | north american porcupine | 481 |
| 65 | giraffe | 479 |
| 66 | north american river otter | 459 |
| 67 | gemsbok | 438 |
| 68 | waterbuck | 437 |
| 69 | common fallow deer | 436 |
| 70 | jaguar | 436 |
| 71 | grey fox | 425 |
| 72 | swamp wallaby | 420 |
| 73 | alpine ibex | 418 |
| 74 | hippopotamus | 418 |
| 75 | south american coati | 411 |
| 76 | mountain zebra | 407 |
| 77 | domestic donkey | 397 |
| 78 | dingo | 397 |
| 79 | hartebeest | 386 |
| 80 | common eland | 385 |
| 81 | sika deer | 354 |
| 82 | grey wolf | 348 |
| 83 | snow leopard | 342 |
| 84 | alpine marmot | 339 |
| 85 | cercopithecus species | 337 |
| 86 | thomson's gazelle | 337 |
| 87 | northern chamois | 315 |
| 88 | golden jackal | 303 |
| 89 | grant's gazelle | 294 |
| 90 | nyala | 277 |
| 91 | bornean orangutan | 276 |
| 92 | sambar | 274 |
| 93 | chital | 263 |
| 94 | muntjac genus | 250 |
| 95 | springbok | 247 |
| 96 | common wombat | 240 |
| 97 | african wild dog | 227 |
| 98 | domestic goat | 226 |
| 99 | blesbok | 185 |
| 100 | grevy's zebra | 185 |
| 101 | bushbuck | 181 |
| 102 | puma | 181 |
| 103 | asian elephant | 178 |
| 104 | howler monkey genus | 169 |
| 105 | steenbok | 167 |
| 106 | beaver genus | 162 |
| 107 | martes species | 145 |
| 108 | gorilla species | 141 |
| 109 | saimiri species | 138 |
| 110 | red panda | 136 |
| 111 | japanese macaque | 126 |
| 112 | llama genus | 125 |
| 113 | kirk's dik-dik | 120 |
| 114 | mongoose family | 118 |
| 115 | colobus species | 117 |
| 116 | giant anteater | 114 |
| 117 | giant otter | 112 |
| 118 | nine-banded armadillo | 110 |
| 119 | dromedary camel | 108 |
| 120 | eurasian lynx | 107 |
| 121 | reindeer | 105 |
| 122 | striped skunk | 101 |
| 123 | lowland tapir | 100 |
| 124 | mountain goat | 100 |
| 125 | sable antelope | 97 |
| 126 | giant panda | 88 |
| 127 | weasel species | 88 |
| 128 | serval | 84 |
| 129 | european bison | 82 |
| 130 | kob | 79 |
| 131 | eurasian badger | 76 |
| 132 | leaf monkeys genus | 74 |
| 133 | black-backed jackal | 72 |
| 134 | gerenuk | 71 |
| 135 | red kangaroo | 71 |
| 136 | klipspringer | 68 |
| 137 | cebus species | 67 |
| 138 | common duiker | 66 |
| 139 | tayra | 66 |
| 140 | rattus genus | 65 |
| 141 | reedbuck genus | 65 |
| 142 | nilgai | 64 |
| 143 | baird's tapir | 61 |
| 144 | american mink | 60 |
| 145 | american badger | 58 |
| 146 | domestic water buffalo | 57 |
| 147 | leopardus species | 57 |
| 148 | old world porcupine family | 55 |
| 149 | wild cat | 54 |
| 150 | muskrat | 52 |
| 151 | black wildebeest | 51 |
| 152 | muridae family | 51 |
| 153 | canada lynx | 47 |
| 154 | dhole | 46 |
| 155 | yak | 44 |
| 156 | ocelot | 43 |
| 157 | roan antelope | 43 |
| 158 | rhinoceros family | 40 |
| 159 | ateles species | 38 |
| 160 | maned wolf | 35 |
| 161 | asiatic black bear | 34 |
| 162 | fisher | 32 |
| 163 | sun bear | 32 |
| 164 | pikas genus | 31 |
| 165 | chimpanzee | 30 |
| 166 | leopard cat | 27 |
| 167 | eurasian otter | 26 |
| 168 | cricetidae family | 24 |
| 169 | striped hyaena | 22 |
| 170 | spectacled bear | 21 |
| 171 | bat-eared fox | 18 |
| 172 | quokka | 18 |
| 173 | red brocket | 18 |
| 174 | cephalophus species | 17 |
| 175 | caracal | 16 |
| 176 | red-necked wallaby | 15 |
| 177 | sloth bear | 15 |
| 178 | water deer | 15 |
| 179 | koala | 14 |
| 180 | glaucomys species | 12 |
| 181 | mangabeys genus | 12 |
| 182 | red river hog | 12 |
| 183 | bongo | 11 |
| 184 | patas monkey | 11 |
| 185 | genet genus | 10 |
| 186 | ringtail | 10 |
| 187 | wolverine | 10 |
| 188 | domestic pig | 10 |
| 189 | aardwolf | 9 |
| 190 | malay tapir | 9 |
| 191 | asiatic wild ass | 7 |
| 192 | rock hyrax | 7 |
| 193 | african civet | 6 |
| 194 | fossa | 6 |
| 195 | honey badger | 6 |
| 196 | brown hyaena | 5 |
| 197 | saguinus species | 5 |
| 198 | callithrix species | 3 |
| 199 | hog badger genus | 3 |
| 200 | raccoon dog | 3 |
| 201 | drill | 2 |
| 202 | eulemur species | 2 |
| 203 | giant armadillo | 2 |
| 204 | kinkajou | 2 |
| 205 | mouflon | 2 |
| 206 | pangolin family | 2 |
| 207 | spilogale species | 2 |
| 208 | meerkat | 1 |
| 209 | hedgehog family | 1 |
| 210 | aardvark | 0 |
| 211 | aye-aye | 0 |
| 212 | binturong | 0 |
| 213 | blackbuck | 0 |
| 214 | brown-throated sloth | 0 |
| 215 | callicebus genus | 0 |
| 216 | clouded leopard | 0 |
| 217 | hoffmann's two-toed sloth | 0 |
| 218 | human | 0 |
| 219 | ring-tailed lemur | 0 |
| 220 | saiga | 0 |
| 221 | sea otter | 0 |
| 222 | unmatched | 0 |
| 223 | eared seals | 0 |
| 224 | elephant seal | 0 |
| 225 | walrus | 0 |
| 226 | grevys zebra | 0 |
| 227 | bairds tapir | 0 |
| 228 | hoffmanns two-toed sloth | 0 |
| 229 | kirks dik-dik | 0 |
| 230 | pinniped clade | 0 |
| 231 | grants gazelle | 0 |
| 232 | thomsons gazelle | 0 |

## gbif — 39,388 records

| | Count | % |
|---|---:|---:|
| Pass | 10,145 | 25.8% |
| Fail | 29,243 | 74.2% |

### Fail Reasons

| Reason | Count | % |
|---|---:|---:|
| low_speciesnet_confidence | 6,476 | 16.4% |
| not_in_225_classes | 6,163 | 15.6% |
| family_mismatch_high_confidence | 2,574 | 6.5% |
| match_level_order | 2,654 | 6.7% |
| match_level_class | 3,152 | 8.0% |
| match_level_no_match | 8,224 | 20.9% |

### Match Levels (26,749 images with valid classification)

| Level | Count | % |
|---|---:|---:|
| species | 7,719 | 28.9% |
| genus | 1,088 | 4.1% |
| family | 3,912 | 14.6% |
| order | 2,654 | 9.9% |
| class | 3,152 | 11.8% |
| no_match | 8,224 | 30.7% |

**Multi-animal images:** 8,515 (21.6%)

**prob\_225\_sum** (26,749 images with valid classification): mean=0.638  median=0.823  p10=0.000  p90=0.980  zeros=3043

### Per-Class Breakdown (211 classes)

| Class | Pre-filter | SN Input | Pass | Pass% | Top Fail Reason |
|---|---:|---:|---:|---:|---|
| aardvark | 38 | 12 | 0 | 0.0% | match_level_class=6 |
| aardwolf | 54 | 15 | 1 | 6.7% | match_level_no_match=5 |
| african buffalo | 298 | 244 | 82 | 33.6% | family_mismatch_high_confidence=108 |
| african civet | 61 | 7 | 1 | 14.3% | match_level_order=4 |
| african elephant | 293 | 245 | 191 | 78.0% | match_level_no_match=41 |
| african wild dog | 222 | 157 | 50 | 31.8% | match_level_class=33 |
| agouti genus | 271 | 114 | 37 | 32.5% | match_level_no_match=26 |
| alpine ibex | 298 | 219 | 83 | 37.9% | family_mismatch_high_confidence=37 |
| alpine marmot | 288 | 187 | 41 | 21.9% | match_level_no_match=61 |
| american badger | 241 | 107 | 21 | 19.6% | match_level_no_match=49 |
| american bison | 289 | 203 | 138 | 68.0% | family_mismatch_high_confidence=40 |
| american black bear | 240 | 134 | 59 | 44.0% | match_level_no_match=51 |
| american mink | 283 | 172 | 12 | 7.0% | match_level_no_match=61 |
| arizona black-tailed prairie dog | 297 | 208 | 129 | 62.0% | match_level_no_match=39 |
| asian elephant | 266 | 188 | 26 | 13.8% | family_mismatch_high_confidence=88 |
| asiatic black bear | 70 | 15 | 5 | 33.3% | match_level_no_match=4 |
| asiatic wild ass | 68 | 13 | 1 | 7.7% | match_level_class=8 |
| ateles species | 94 | 61 | 2 | 3.3% | match_level_no_match=21 |
| aye-aye | 20 | 10 | 0 | 0.0% | match_level_class=4 |
| baboon genus | 291 | 223 | 79 | 35.4% | match_level_class=52 |
| baird's tapir | 147 | 88 | 13 | 14.8% | match_level_no_match=39 |
| bat-eared fox | 143 | 68 | 5 | 7.4% | low_speciesnet_confidence=29 |
| beaver genus | 89 | 20 | 0 | 0.0% | match_level_no_match=14 |
| bighorn sheep | 290 | 221 | 57 | 25.8% | match_level_order=81 |
| binturong | 28 | 9 | 0 | 0.0% | match_level_no_match=3 |
| black wildebeest | 101 | 65 | 25 | 38.5% | family_mismatch_high_confidence=25 |
| black-backed jackal | 108 | 87 | 51 | 58.6% | low_speciesnet_confidence=20 |
| blackbuck | 227 | 148 | 0 | 0.0% | match_level_no_match=128 |
| blesbok | 297 | 231 | 52 | 22.5% | low_speciesnet_confidence=68 |
| bobcat | 266 | 107 | 66 | 61.7% | match_level_no_match=32 |
| bongo | 9 | 1 | 0 | 0.0% | match_level_class=1 |
| bornean orangutan | 234 | 124 | 48 | 38.7% | match_level_no_match=33 |
| brown bear | 248 | 146 | 65 | 44.5% | match_level_class=30 |
| brown hyaena | 41 | 17 | 2 | 11.8% | low_speciesnet_confidence=7 |
| brown-throated sloth | 287 | 209 | 0 | 0.0% | match_level_no_match=158 |
| bushbuck | 392 | 324 | 151 | 46.6% | match_level_order=79 |
| california ground squirrel | 299 | 260 | 52 | 20.0% | match_level_no_match=80 |
| callicebus genus | 39 | 34 | 0 | 0.0% | match_level_no_match=17 |
| callithrix species | 291 | 240 | 0 | 0.0% | match_level_no_match=113 |
| canada lynx | 132 | 50 | 16 | 32.0% | match_level_no_match=17 |
| capybara | 270 | 193 | 54 | 28.0% | match_level_class=52 |
| caracal | 94 | 37 | 7 | 18.9% | low_speciesnet_confidence=14 |
| cebus species | 489 | 392 | 12 | 3.1% | match_level_no_match=138 |
| cephalophus species | 191 | 97 | 11 | 11.3% | match_level_order=40 |
| cercopithecus species | 555 | 376 | 103 | 27.4% | match_level_no_match=168 |
| cheetah | 297 | 246 | 190 | 77.2% | match_level_no_match=32 |
| chimpanzee | 178 | 88 | 7 | 8.0% | match_level_no_match=28 |
| chipmunk genus | 282 | 231 | 50 | 21.6% | low_speciesnet_confidence=89 |
| chital | 299 | 227 | 49 | 21.6% | family_mismatch_high_confidence=128 |
| clouded leopard | 11 | 1 | 0 | 0.0% | family_mismatch_high_confidence=1 |
| collared peccary | 292 | 158 | 66 | 41.8% | match_level_order=46 |
| colobus species | 302 | 235 | 50 | 21.3% | match_level_no_match=138 |
| common duiker | 289 | 162 | 21 | 13.0% | match_level_order=78 |
| common eland | 298 | 241 | 115 | 47.7% | family_mismatch_high_confidence=51 |
| common fallow deer | 294 | 157 | 44 | 28.0% | family_mismatch_high_confidence=57 |
| common warthog | 9 | 8 | 4 | 50.0% | match_level_order=1 |
| common wildebeest | 296 | 249 | 148 | 59.4% | family_mismatch_high_confidence=33 |
| common wombat | 260 | 131 | 27 | 20.6% | match_level_class=42 |
| cottontail rabbits genus | 248 | 154 | 46 | 29.9% | match_level_no_match=34 |
| coyote | 289 | 136 | 78 | 57.4% | match_level_no_match=30 |
| cricetidae family | 80 | 20 | 0 | 0.0% | match_level_no_match=15 |
| dhole | 81 | 58 | 12 | 20.7% | match_level_class=15 |
| domestic cat | 286 | 149 | 85 | 57.0% | match_level_no_match=30 |
| domestic cattle | 272 | 137 | 109 | 79.6% | match_level_no_match=17 |
| domestic dog | 280 | 194 | 96 | 49.5% | match_level_class=32 |
| domestic donkey | 293 | 217 | 59 | 27.2% | match_level_class=102 |
| domestic horse | 281 | 183 | 104 | 56.8% | match_level_class=51 |
| domestic sheep | 271 | 110 | 24 | 21.8% | match_level_order=46 |
| domestic water buffalo | 285 | 185 | 11 | 5.9% | family_mismatch_high_confidence=144 |
| drill | 17 | 6 | 0 | 0.0% | family_mismatch_high_confidence=3 |
| dromedary camel | 221 | 144 | 38 | 26.4% | match_level_order=45 |
| eastern fox squirrel | 295 | 241 | 125 | 51.9% | match_level_no_match=55 |
| eastern gray squirrel | 293 | 211 | 85 | 40.3% | match_level_no_match=68 |
| eastern grey kangaroo | 299 | 206 | 65 | 31.6% | match_level_no_match=63 |
| elk | 282 | 163 | 117 | 71.8% | match_level_no_match=20 |
| eulemur species | 263 | 210 | 1 | 0.5% | match_level_no_match=82 |
| eurasian badger | 196 | 34 | 4 | 11.8% | low_speciesnet_confidence=13 |
| eurasian lynx | 239 | 72 | 39 | 54.2% | match_level_no_match=19 |
| eurasian otter | 227 | 97 | 7 | 7.2% | match_level_no_match=33 |
| eurasian red squirrel | 279 | 207 | 50 | 24.2% | low_speciesnet_confidence=63 |
| european bison | 157 | 84 | 29 | 34.5% | family_mismatch_high_confidence=32 |
| european rabbit | 264 | 180 | 54 | 30.0% | match_level_no_match=58 |
| european roe deer | 268 | 147 | 55 | 37.4% | family_mismatch_high_confidence=58 |
| fisher | 263 | 60 | 17 | 28.3% | match_level_no_match=15 |
| fossa | 29 | 19 | 2 | 10.5% | match_level_order=6 |
| gemsbok | 382 | 281 | 175 | 62.3% | low_speciesnet_confidence=31 |
| genet genus | 237 | 63 | 2 | 3.2% | match_level_no_match=26 |
| gerenuk | 114 | 82 | 30 | 36.6% | family_mismatch_high_confidence=32 |
| giant anteater | 222 | 126 | 53 | 42.1% | match_level_no_match=42 |
| giant armadillo | 22 | 6 | 2 | 33.3% | match_level_no_match=2 |
| giant otter | 178 | 122 | 31 | 25.4% | match_level_no_match=45 |
| giant panda | 17 | 3 | 0 | 0.0% | match_level_no_match=2 |
| giraffe | 178 | 145 | 120 | 82.8% | match_level_no_match=16 |
| glaucomys species | 451 | 127 | 6 | 4.7% | match_level_no_match=46 |
| golden jackal | 295 | 165 | 69 | 41.8% | match_level_class=33 |
| golden mantled ground squirrel | 298 | 259 | 53 | 20.5% | low_speciesnet_confidence=80 |
| gorilla species | 212 | 118 | 21 | 17.8% | match_level_class=49 |
| grant's gazelle | 296 | 240 | 148 | 61.7% | family_mismatch_high_confidence=49 |
| greater kudu | 296 | 272 | 102 | 37.5% | match_level_order=72 |
| grevy's zebra | 111 | 85 | 80 | 94.1% | match_level_no_match=4 |
| grey fox | 292 | 25 | 10 | 40.0% | low_speciesnet_confidence=5 |
| grey wolf | 229 | 67 | 30 | 44.8% | match_level_no_match=17 |
| hares and jackrabbits genus | 216 | 95 | 16 | 16.8% | match_level_class=34 |
| hartebeest | 224 | 182 | 110 | 60.4% | match_level_order=26 |
| hippopotamus | 295 | 174 | 44 | 25.3% | match_level_no_match=55 |
| hoffmann's two-toed sloth | 290 | 178 | 0 | 0.0% | match_level_no_match=113 |
| hog badger genus | 21 | 11 | 2 | 18.2% | match_level_no_match=4 |
| honey badger | 83 | 32 | 2 | 6.2% | match_level_no_match=10 |
| howler monkey genus | 297 | 247 | 13 | 5.3% | match_level_no_match=102 |
| human | 107 | 5 | 0 | 0.0% | match_level_no_match=4 |
| impala | 299 | 262 | 180 | 68.7% | match_level_order=33 |
| jaguar | 237 | 109 | 89 | 81.7% | match_level_no_match=10 |
| japanese macaque | 180 | 135 | 41 | 30.4% | match_level_class=43 |
| kangaroo family | 1,523 | 1,047 | 282 | 26.9% | match_level_no_match=525 |
| kinkajou | 155 | 54 | 1 | 1.9% | match_level_no_match=21 |
| kirk's dik-dik | 201 | 150 | 75 | 50.0% | match_level_order=33 |
| klipspringer | 288 | 221 | 23 | 10.4% | match_level_order=70 |
| koala | 279 | 166 | 2 | 1.2% | match_level_no_match=129 |
| kob | 135 | 117 | 37 | 31.6% | match_level_order=28 |
| leaf monkeys genus | 247 | 189 | 17 | 9.0% | match_level_no_match=98 |
| leopard | 277 | 197 | 102 | 51.8% | match_level_no_match=59 |
| leopard cat | 130 | 37 | 8 | 21.6% | match_level_no_match=15 |
| leopardus species | 266 | 48 | 12 | 25.0% | match_level_no_match=15 |
| lion | 293 | 230 | 117 | 50.9% | match_level_no_match=54 |
| llama genus | 660 | 491 | 80 | 16.3% | match_level_order=242 |
| lowland tapir | 208 | 85 | 30 | 35.3% | match_level_class=24 |
| lycalopex species | 599 | 461 | 183 | 39.7% | family_mismatch_high_confidence=115 |
| macaque species | 1,855 | 1,220 | 435 | 35.7% | low_speciesnet_confidence=271 |
| malay tapir | 14 | 3 | 2 | 66.7% | match_level_no_match=1 |
| maned wolf | 77 | 30 | 14 | 46.7% | match_level_class=7 |
| mangabeys genus | 53 | 16 | 1 | 6.2% | match_level_no_match=6 |
| martes species | 867 | 296 | 43 | 14.5% | match_level_no_match=111 |
| meerkat | 117 | 98 | 0 | 0.0% | match_level_class=37 |
| moose | 230 | 125 | 58 | 46.4% | match_level_no_match=31 |
| mouflon | 3 | 1 | 0 | 0.0% | family_mismatch_high_confidence=1 |
| mountain goat | 271 | 153 | 27 | 17.6% | family_mismatch_high_confidence=55 |
| mountain zebra | 166 | 146 | 139 | 95.2% | match_level_no_match=4 |
| mule deer | 295 | 129 | 104 | 80.6% | match_level_no_match=15 |
| muntjac genus | 292 | 189 | 36 | 19.0% | family_mismatch_high_confidence=64 |
| muskrat | 284 | 159 | 4 | 2.5% | match_level_order=66 |
| nilgai | 274 | 191 | 16 | 8.4% | match_level_order=55 |
| nine-banded armadillo | 259 | 43 | 16 | 37.2% | match_level_no_match=20 |
| north american porcupine | 243 | 141 | 35 | 24.8% | match_level_no_match=75 |
| north american river otter | 240 | 117 | 13 | 11.1% | match_level_no_match=56 |
| northern chamois | 284 | 169 | 65 | 38.5% | match_level_no_match=35 |
| northern raccoon | 215 | 59 | 30 | 50.8% | match_level_no_match=18 |
| nutria | 286 | 184 | 77 | 41.8% | match_level_no_match=62 |
| nyala | 300 | 269 | 81 | 30.1% | match_level_order=50 |
| ocelot | 258 | 24 | 12 | 50.0% | family_mismatch_high_confidence=6 |
| old world porcupine family | 202 | 33 | 14 | 42.4% | match_level_no_match=14 |
| pangolin family | 57 | 10 | 0 | 0.0% | match_level_no_match=6 |
| patas monkey | 115 | 75 | 4 | 5.3% | match_level_no_match=23 |
| pikas genus | 272 | 217 | 5 | 2.3% | match_level_class=76 |
| plains zebra | 296 | 257 | 241 | 93.8% | match_level_no_match=8 |
| pronghorn | 287 | 191 | 109 | 57.1% | match_level_order=45 |
| puma | 248 | 39 | 17 | 43.6% | match_level_class=9 |
| quokka | 155 | 131 | 12 | 9.2% | low_speciesnet_confidence=44 |
| raccoon dog | 287 | 91 | 1 | 1.1% | match_level_order=26 |
| rattus genus | 290 | 143 | 3 | 2.1% | match_level_no_match=52 |
| red brocket | 187 | 27 | 12 | 44.4% | family_mismatch_high_confidence=6 |
| red deer | 279 | 153 | 88 | 57.5% | match_level_no_match=32 |
| red fox | 257 | 105 | 52 | 49.5% | match_level_no_match=31 |
| red kangaroo | 157 | 107 | 10 | 9.3% | match_level_class=42 |
| red panda | 24 | 9 | 2 | 22.2% | match_level_class=2 |
| red river hog | 14 | 3 | 0 | 0.0% | match_level_no_match=2 |
| red squirrel | 280 | 214 | 50 | 23.4% | low_speciesnet_confidence=54 |
| red-necked wallaby | 47 | 28 | 14 | 50.0% | match_level_class=6 |
| reedbuck genus | 247 | 195 | 24 | 12.3% | match_level_order=90 |
| reindeer | 250 | 84 | 13 | 15.5% | match_level_order=26 |
| ring-tailed lemur | 143 | 121 | 0 | 0.0% | match_level_class=47 |
| ringtail | 263 | 43 | 3 | 7.0% | low_speciesnet_confidence=16 |
| roan antelope | 126 | 99 | 15 | 15.2% | match_level_order=30 |
| rock hyrax | 292 | 228 | 2 | 0.9% | match_level_class=124 |
| sable antelope | 192 | 154 | 32 | 20.8% | family_mismatch_high_confidence=62 |
| saguinus species | 100 | 67 | 2 | 3.0% | match_level_no_match=40 |
| saiga | 43 | 17 | 0 | 0.0% | match_level_no_match=16 |
| saimiri species | 299 | 182 | 19 | 10.4% | match_level_no_match=103 |
| sambar | 283 | 168 | 52 | 31.0% | family_mismatch_high_confidence=51 |
| sea otter | 292 | 135 | 0 | 0.0% | match_level_no_match=108 |
| serval | 103 | 58 | 31 | 53.4% | match_level_no_match=16 |
| short-beaked echidna | 296 | 202 | 130 | 64.4% | match_level_no_match=65 |
| sika deer | 294 | 225 | 79 | 35.1% | family_mismatch_high_confidence=73 |
| sloth bear | 47 | 22 | 2 | 9.1% | family_mismatch_high_confidence=10 |
| snow leopard | 12 | 2 | 0 | 0.0% | match_level_no_match=2 |
| south american coati | 293 | 233 | 89 | 38.2% | match_level_no_match=49 |
| spectacled bear | 76 | 39 | 8 | 20.5% | match_level_no_match=15 |
| spilogale species | 91 | 5 | 0 | 0.0% | match_level_no_match=3 |
| spotted hyaena | 287 | 210 | 121 | 57.6% | match_level_class=31 |
| springbok | 298 | 202 | 60 | 29.7% | family_mismatch_high_confidence=45 |
| squirrel family | 1,677 | 1,318 | 340 | 25.8% | match_level_no_match=637 |
| steenbok | 297 | 261 | 43 | 16.5% | match_level_order=82 |
| striped hyaena | 53 | 25 | 7 | 28.0% | match_level_order=8 |
| striped skunk | 280 | 31 | 3 | 9.7% | match_level_no_match=20 |
| sun bear | 51 | 15 | 4 | 26.7% | match_level_class=5 |
| swamp wallaby | 300 | 156 | 50 | 32.1% | match_level_no_match=61 |
| tayra | 232 | 118 | 28 | 23.7% | match_level_no_match=47 |
| thomson's gazelle | 294 | 232 | 182 | 78.4% | family_mismatch_high_confidence=15 |
| tiger | 231 | 156 | 91 | 58.3% | match_level_no_match=47 |
| unmatched | 15,209 | 7,715 | 0 | 0.0% | not_in_225_classes=6163 |
| vervet monkey | 295 | 248 | 102 | 41.1% | match_level_no_match=74 |
| water deer | 135 | 80 | 9 | 11.2% | family_mismatch_high_confidence=30 |
| waterbuck | 300 | 263 | 89 | 33.8% | match_level_order=64 |
| weasel species | 896 | 458 | 19 | 4.1% | match_level_no_match=168 |
| western gray squirrel | 274 | 91 | 25 | 27.5% | match_level_no_match=43 |
| white-nosed coati | 299 | 248 | 86 | 34.7% | low_speciesnet_confidence=52 |
| white-tailed deer | 238 | 137 | 91 | 66.4% | match_level_no_match=34 |
| wild boar | 251 | 107 | 52 | 48.6% | match_level_no_match=28 |
| wild cat | 286 | 37 | 11 | 29.7% | match_level_no_match=10 |
| wolverine | 85 | 30 | 4 | 13.3% | match_level_no_match=16 |
| woodchuck | 289 | 215 | 68 | 31.6% | match_level_no_match=86 |
| yellow-bellied marmot | 291 | 216 | 88 | 40.7% | match_level_no_match=76 |

### Classes Ranked by Passed Images

| Rank | Class | Passed |
|---:|---|---:|
| 1 | macaque species | 435 |
| 2 | squirrel family | 340 |
| 3 | kangaroo family | 282 |
| 4 | plains zebra | 241 |
| 5 | african elephant | 191 |
| 6 | cheetah | 190 |
| 7 | lycalopex species | 183 |
| 8 | thomson's gazelle | 182 |
| 9 | impala | 180 |
| 10 | gemsbok | 175 |
| 11 | bushbuck | 151 |
| 12 | common wildebeest | 148 |
| 13 | grant's gazelle | 148 |
| 14 | mountain zebra | 139 |
| 15 | american bison | 138 |
| 16 | short-beaked echidna | 130 |
| 17 | arizona black-tailed prairie dog | 129 |
| 18 | eastern fox squirrel | 125 |
| 19 | spotted hyaena | 121 |
| 20 | giraffe | 120 |
| 21 | elk | 117 |
| 22 | lion | 117 |
| 23 | common eland | 115 |
| 24 | hartebeest | 110 |
| 25 | domestic cattle | 109 |
| 26 | pronghorn | 109 |
| 27 | domestic horse | 104 |
| 28 | mule deer | 104 |
| 29 | cercopithecus species | 103 |
| 30 | greater kudu | 102 |
| 31 | leopard | 102 |
| 32 | vervet monkey | 102 |
| 33 | domestic dog | 96 |
| 34 | tiger | 91 |
| 35 | white-tailed deer | 91 |
| 36 | jaguar | 89 |
| 37 | south american coati | 89 |
| 38 | waterbuck | 89 |
| 39 | red deer | 88 |
| 40 | yellow-bellied marmot | 88 |
| 41 | white-nosed coati | 86 |
| 42 | domestic cat | 85 |
| 43 | eastern gray squirrel | 85 |
| 44 | alpine ibex | 83 |
| 45 | african buffalo | 82 |
| 46 | nyala | 81 |
| 47 | grevy's zebra | 80 |
| 48 | llama genus | 80 |
| 49 | baboon genus | 79 |
| 50 | sika deer | 79 |
| 51 | coyote | 78 |
| 52 | nutria | 77 |
| 53 | kirk's dik-dik | 75 |
| 54 | golden jackal | 69 |
| 55 | woodchuck | 68 |
| 56 | bobcat | 66 |
| 57 | collared peccary | 66 |
| 58 | brown bear | 65 |
| 59 | eastern grey kangaroo | 65 |
| 60 | northern chamois | 65 |
| 61 | springbok | 60 |
| 62 | american black bear | 59 |
| 63 | domestic donkey | 59 |
| 64 | moose | 58 |
| 65 | bighorn sheep | 57 |
| 66 | european roe deer | 55 |
| 67 | capybara | 54 |
| 68 | european rabbit | 54 |
| 69 | giant anteater | 53 |
| 70 | golden mantled ground squirrel | 53 |
| 71 | blesbok | 52 |
| 72 | california ground squirrel | 52 |
| 73 | red fox | 52 |
| 74 | sambar | 52 |
| 75 | wild boar | 52 |
| 76 | black-backed jackal | 51 |
| 77 | african wild dog | 50 |
| 78 | chipmunk genus | 50 |
| 79 | colobus species | 50 |
| 80 | eurasian red squirrel | 50 |
| 81 | red squirrel | 50 |
| 82 | swamp wallaby | 50 |
| 83 | chital | 49 |
| 84 | bornean orangutan | 48 |
| 85 | cottontail rabbits genus | 46 |
| 86 | common fallow deer | 44 |
| 87 | hippopotamus | 44 |
| 88 | martes species | 43 |
| 89 | steenbok | 43 |
| 90 | alpine marmot | 41 |
| 91 | japanese macaque | 41 |
| 92 | eurasian lynx | 39 |
| 93 | dromedary camel | 38 |
| 94 | agouti genus | 37 |
| 95 | kob | 37 |
| 96 | muntjac genus | 36 |
| 97 | north american porcupine | 35 |
| 98 | sable antelope | 32 |
| 99 | giant otter | 31 |
| 100 | serval | 31 |
| 101 | gerenuk | 30 |
| 102 | grey wolf | 30 |
| 103 | lowland tapir | 30 |
| 104 | northern raccoon | 30 |
| 105 | european bison | 29 |
| 106 | tayra | 28 |
| 107 | common wombat | 27 |
| 108 | mountain goat | 27 |
| 109 | asian elephant | 26 |
| 110 | black wildebeest | 25 |
| 111 | western gray squirrel | 25 |
| 112 | domestic sheep | 24 |
| 113 | reedbuck genus | 24 |
| 114 | klipspringer | 23 |
| 115 | american badger | 21 |
| 116 | common duiker | 21 |
| 117 | gorilla species | 21 |
| 118 | saimiri species | 19 |
| 119 | weasel species | 19 |
| 120 | fisher | 17 |
| 121 | leaf monkeys genus | 17 |
| 122 | puma | 17 |
| 123 | canada lynx | 16 |
| 124 | hares and jackrabbits genus | 16 |
| 125 | nilgai | 16 |
| 126 | nine-banded armadillo | 16 |
| 127 | roan antelope | 15 |
| 128 | maned wolf | 14 |
| 129 | old world porcupine family | 14 |
| 130 | red-necked wallaby | 14 |
| 131 | baird's tapir | 13 |
| 132 | howler monkey genus | 13 |
| 133 | north american river otter | 13 |
| 134 | reindeer | 13 |
| 135 | american mink | 12 |
| 136 | cebus species | 12 |
| 137 | dhole | 12 |
| 138 | leopardus species | 12 |
| 139 | ocelot | 12 |
| 140 | quokka | 12 |
| 141 | red brocket | 12 |
| 142 | cephalophus species | 11 |
| 143 | domestic water buffalo | 11 |
| 144 | wild cat | 11 |
| 145 | grey fox | 10 |
| 146 | red kangaroo | 10 |
| 147 | water deer | 9 |
| 148 | leopard cat | 8 |
| 149 | spectacled bear | 8 |
| 150 | caracal | 7 |
| 151 | chimpanzee | 7 |
| 152 | eurasian otter | 7 |
| 153 | striped hyaena | 7 |
| 154 | glaucomys species | 6 |
| 155 | asiatic black bear | 5 |
| 156 | bat-eared fox | 5 |
| 157 | pikas genus | 5 |
| 158 | common warthog | 4 |
| 159 | eurasian badger | 4 |
| 160 | muskrat | 4 |
| 161 | patas monkey | 4 |
| 162 | sun bear | 4 |
| 163 | wolverine | 4 |
| 164 | rattus genus | 3 |
| 165 | ringtail | 3 |
| 166 | striped skunk | 3 |
| 167 | ateles species | 2 |
| 168 | brown hyaena | 2 |
| 169 | fossa | 2 |
| 170 | genet genus | 2 |
| 171 | giant armadillo | 2 |
| 172 | hog badger genus | 2 |
| 173 | honey badger | 2 |
| 174 | koala | 2 |
| 175 | malay tapir | 2 |
| 176 | red panda | 2 |
| 177 | rock hyrax | 2 |
| 178 | saguinus species | 2 |
| 179 | sloth bear | 2 |
| 180 | aardwolf | 1 |
| 181 | african civet | 1 |
| 182 | asiatic wild ass | 1 |
| 183 | eulemur species | 1 |
| 184 | kinkajou | 1 |
| 185 | mangabeys genus | 1 |
| 186 | raccoon dog | 1 |
| 187 | aardvark | 0 |
| 188 | aye-aye | 0 |
| 189 | beaver genus | 0 |
| 190 | binturong | 0 |
| 191 | blackbuck | 0 |
| 192 | bongo | 0 |
| 193 | brown-throated sloth | 0 |
| 194 | callicebus genus | 0 |
| 195 | callithrix species | 0 |
| 196 | clouded leopard | 0 |
| 197 | cricetidae family | 0 |
| 198 | drill | 0 |
| 199 | giant panda | 0 |
| 200 | hoffmann's two-toed sloth | 0 |
| 201 | human | 0 |
| 202 | meerkat | 0 |
| 203 | mouflon | 0 |
| 204 | pangolin family | 0 |
| 205 | red river hog | 0 |
| 206 | ring-tailed lemur | 0 |
| 207 | saiga | 0 |
| 208 | sea otter | 0 |
| 209 | snow leopard | 0 |
| 210 | spilogale species | 0 |
| 211 | unmatched | 0 |

## inaturalist — 400,067 records

| | Count | % |
|---|---:|---:|
| Pass | 138,310 | 34.6% |
| Fail | 261,757 | 65.4% |

### Fail Reasons

| Reason | Count | % |
|---|---:|---:|
| primary_crop_too_small | 2,554 | 0.6% |
| low_speciesnet_confidence | 56,501 | 14.1% |
| family_mismatch_high_confidence | 14,784 | 3.7% |
| match_level_order | 13,341 | 3.3% |
| match_level_class | 23,091 | 5.8% |
| match_level_no_match | 151,486 | 37.9% |

### Match Levels (341,012 images with valid classification)

| Level | Count | % |
|---|---:|---:|
| species | 114,367 | 33.5% |
| genus | 13,727 | 4.0% |
| family | 25,000 | 7.3% |
| order | 13,341 | 3.9% |
| class | 23,091 | 6.8% |
| no_match | 151,486 | 44.4% |

**Multi-animal images:** 65,106 (16.3%)

**prob\_225\_sum** (341,012 images with valid classification): mean=0.532  median=0.691  p10=0.000  p90=0.975  zeros=65032

### Per-Class Breakdown (215 classes)

| Class | Pre-filter | SN Input | Pass | Pass% | Top Fail Reason |
|---|---:|---:|---:|---:|---|
| aardvark | 203 | 27 | 0 | 0.0% | match_level_no_match=14 |
| african buffalo | 1,563 | 1,498 | 529 | 35.3% | family_mismatch_high_confidence=558 |
| african civet | 93 | 12 | 3 | 25.0% | match_level_no_match=3 |
| african elephant | 3,816 | 3,662 | 2,767 | 75.6% | match_level_no_match=598 |
| african wild dog | 427 | 362 | 81 | 22.4% | match_level_no_match=114 |
| agouti genus | 1,818 | 1,695 | 618 | 36.5% | match_level_no_match=465 |
| alpine ibex | 1,043 | 1,019 | 317 | 31.1% | match_level_no_match=277 |
| alpine marmot | 2,000 | 1,498 | 285 | 19.0% | match_level_no_match=764 |
| american badger | 626 | 269 | 35 | 13.0% | match_level_no_match=143 |
| american bison | 2,132 | 2,018 | 903 | 44.7% | family_mismatch_high_confidence=534 |
| american black bear | 3,801 | 3,250 | 1,198 | 36.9% | match_level_no_match=1391 |
| arizona black-tailed prairie dog | 1,606 | 1,542 | 662 | 42.9% | match_level_no_match=610 |
| asian elephant | 1,047 | 972 | 106 | 10.9% | family_mismatch_high_confidence=439 |
| asiatic black bear | 121 | 56 | 14 | 25.0% | low_speciesnet_confidence=15 |
| asiatic wild ass | 96 | 64 | 3 | 4.7% | match_level_class=32 |
| ateles species | 1,600 | 1,340 | 35 | 2.6% | match_level_no_match=623 |
| aye-aye | 31 | 13 | 0 | 0.0% | match_level_no_match=10 |
| baboon genus | 2,346 | 2,168 | 769 | 35.5% | match_level_no_match=587 |
| baird's tapir | 392 | 235 | 48 | 20.4% | match_level_no_match=85 |
| bat-eared fox | 252 | 154 | 12 | 7.8% | low_speciesnet_confidence=61 |
| beaver genus | 3,782 | 2,137 | 162 | 7.6% | match_level_no_match=1137 |
| bighorn sheep | 3,118 | 2,908 | 465 | 16.0% | match_level_order=998 |
| binturong | 26 | 18 | 0 | 0.0% | match_level_no_match=13 |
| black wildebeest | 185 | 101 | 26 | 25.7% | match_level_no_match=27 |
| blackbuck | 348 | 282 | 0 | 0.0% | match_level_no_match=209 |
| blesbok | 826 | 577 | 130 | 22.5% | low_speciesnet_confidence=153 |
| bobcat | 2,184 | 1,740 | 821 | 47.2% | match_level_no_match=700 |
| bongo | 35 | 33 | 10 | 30.3% | family_mismatch_high_confidence=8 |
| bornean orangutan | 348 | 258 | 75 | 29.1% | match_level_no_match=99 |
| brown bear | 1,993 | 1,223 | 430 | 35.2% | match_level_no_match=346 |
| brown hyaena | 110 | 44 | 1 | 2.3% | family_mismatch_high_confidence=12 |
| brown-throated sloth | 1,619 | 1,294 | 0 | 0.0% | match_level_no_match=944 |
| bushbuck | 73 | 64 | 29 | 45.3% | match_level_order=20 |
| california ground squirrel | 5,562 | 5,357 | 1,147 | 21.4% | match_level_no_match=2544 |
| callicebus genus | 82 | 78 | 0 | 0.0% | match_level_no_match=49 |
| callithrix species | 1,526 | 1,447 | 3 | 0.2% | match_level_no_match=882 |
| canada lynx | 353 | 114 | 29 | 25.4% | match_level_no_match=44 |
| capybara | 1,673 | 1,534 | 487 | 31.7% | match_level_no_match=375 |
| caracal | 278 | 123 | 8 | 6.5% | match_level_no_match=61 |
| cebus species | 1,410 | 1,356 | 53 | 3.9% | match_level_no_match=635 |
| cephalophus species | 23 | 17 | 2 | 11.8% | match_level_order=8 |
| cercopithecus species | 1,124 | 978 | 213 | 21.8% | match_level_no_match=481 |
| cheetah | 867 | 689 | 443 | 64.3% | match_level_no_match=191 |
| chimpanzee | 263 | 202 | 14 | 6.9% | match_level_no_match=88 |
| chipmunk genus | 7,072 | 6,628 | 1,686 | 25.4% | match_level_no_match=2843 |
| chital | 1,160 | 862 | 196 | 22.7% | family_mismatch_high_confidence=359 |
| clouded leopard | 37 | 18 | 0 | 0.0% | family_mismatch_high_confidence=10 |
| collared peccary | 1,495 | 1,236 | 441 | 35.7% | match_level_no_match=342 |
| colobus species | 473 | 405 | 62 | 15.3% | match_level_no_match=273 |
| common duiker | 716 | 308 | 44 | 14.3% | match_level_order=113 |
| common eland | 970 | 707 | 263 | 37.2% | match_level_no_match=173 |
| common fallow deer | 1,386 | 1,269 | 323 | 25.5% | family_mismatch_high_confidence=378 |
| common warthog | 1,555 | 1,494 | 881 | 59.0% | match_level_no_match=267 |
| common wildebeest | 1,379 | 1,298 | 708 | 54.5% | match_level_no_match=212 |
| common wombat | 796 | 468 | 105 | 22.4% | low_speciesnet_confidence=137 |
| cottontail rabbits genus | 7,992 | 7,551 | 2,148 | 28.4% | match_level_no_match=2237 |
| coyote | 8,039 | 6,639 | 2,934 | 44.2% | match_level_no_match=2378 |
| cricetidae family | 6,071 | 4,736 | 24 | 0.5% | match_level_no_match=3037 |
| dhole | 169 | 122 | 23 | 18.9% | match_level_no_match=41 |
| domestic cat | 7,747 | 6,625 | 3,496 | 52.8% | match_level_no_match=1840 |
| domestic cattle | 3,305 | 1,994 | 1,441 | 72.3% | match_level_no_match=363 |
| domestic dog | 2,115 | 1,604 | 708 | 44.1% | match_level_no_match=378 |
| domestic donkey | 1,169 | 870 | 228 | 26.2% | match_level_class=325 |
| domestic horse | 1,645 | 1,532 | 665 | 43.4% | match_level_class=413 |
| domestic sheep | 1,810 | 1,060 | 359 | 33.9% | match_level_no_match=326 |
| domestic water buffalo | 789 | 486 | 45 | 9.3% | family_mismatch_high_confidence=268 |
| drill | 18 | 17 | 2 | 11.8% | low_speciesnet_confidence=8 |
| dromedary camel | 321 | 202 | 47 | 23.3% | match_level_no_match=51 |
| eared seals | 9,874 | 8,986 | 0 | 0.0% | match_level_no_match=7498 |
| eastern cottontail | 12,264 | 11,456 | 3,567 | 31.1% | match_level_no_match=3744 |
| eastern fox squirrel | 12,709 | 12,187 | 6,324 | 51.9% | match_level_no_match=3947 |
| eastern gray squirrel | 32,503 | 30,777 | 15,529 | 50.5% | match_level_no_match=9947 |
| eastern grey kangaroo | 2,570 | 2,412 | 533 | 22.1% | match_level_no_match=770 |
| elephant seal | 1,783 | 1,691 | 0 | 0.0% | match_level_no_match=1413 |
| elk | 5,800 | 5,242 | 3,303 | 63.0% | match_level_no_match=1117 |
| eulemur species | 704 | 605 | 1 | 0.2% | match_level_no_match=280 |
| eurasian badger | 2,185 | 411 | 64 | 15.6% | match_level_no_match=147 |
| eurasian lynx | 212 | 92 | 43 | 46.7% | match_level_no_match=24 |
| eurasian otter | 882 | 308 | 17 | 5.5% | match_level_no_match=176 |
| eurasian red squirrel | 9,930 | 9,406 | 2,555 | 27.2% | match_level_no_match=4253 |
| european bison | 247 | 152 | 47 | 30.9% | family_mismatch_high_confidence=55 |
| european hare | 3,372 | 3,043 | 583 | 19.2% | match_level_no_match=1315 |
| european rabbit | 5,675 | 5,176 | 917 | 17.7% | match_level_no_match=2105 |
| european roe deer | 6,519 | 5,762 | 1,455 | 25.3% | match_level_no_match=1935 |
| fisher | 311 | 68 | 15 | 22.1% | match_level_no_match=28 |
| fossa | 67 | 50 | 3 | 6.0% | low_speciesnet_confidence=19 |
| gemsbok | 750 | 524 | 252 | 48.1% | match_level_no_match=93 |
| genet genus | 801 | 153 | 6 | 3.9% | match_level_no_match=62 |
| gerenuk | 99 | 92 | 37 | 40.2% | family_mismatch_high_confidence=29 |
| giant anteater | 368 | 216 | 56 | 25.9% | match_level_no_match=82 |
| giant armadillo | 17 | 5 | 0 | 0.0% | match_level_no_match=4 |
| giant otter | 367 | 306 | 78 | 25.5% | match_level_no_match=135 |
| giant panda | 52 | 41 | 7 | 17.1% | match_level_no_match=22 |
| giraffe | 180 | 155 | 119 | 76.8% | match_level_no_match=29 |
| glaucomys species | 692 | 237 | 6 | 2.5% | match_level_no_match=117 |
| golden jackal | 888 | 557 | 202 | 36.3% | match_level_no_match=121 |
| golden mantled ground squirrel | 2,621 | 2,543 | 571 | 22.5% | match_level_no_match=851 |
| gorilla species | 404 | 338 | 49 | 14.5% | match_level_class=135 |
| grant's gazelle | 319 | 288 | 146 | 50.7% | family_mismatch_high_confidence=71 |
| greater kudu | 1,462 | 1,399 | 457 | 32.7% | match_level_order=278 |
| grevy's zebra | 134 | 124 | 105 | 84.7% | match_level_no_match=9 |
| grey fox | 1,600 | 1,088 | 414 | 38.1% | match_level_no_match=402 |
| grey wolf | 1,291 | 333 | 125 | 37.5% | match_level_no_match=133 |
| hares and jackrabbits genus | 5,207 | 4,667 | 958 | 20.5% | match_level_no_match=1638 |
| hartebeest | 831 | 674 | 263 | 39.0% | match_level_no_match=118 |
| hedgehog family | 4,421 | 3,227 | 1 | 0.0% | match_level_no_match=2421 |
| hippopotamus | 1,155 | 1,099 | 241 | 21.9% | match_level_no_match=452 |
| hoffmann's two-toed sloth | 698 | 489 | 0 | 0.0% | match_level_no_match=316 |
| hog badger genus | 8 | 1 | 0 | 0.0% | match_level_class=1 |
| honey badger | 154 | 58 | 3 | 5.2% | match_level_no_match=27 |
| howler monkey genus | 3,017 | 2,929 | 152 | 5.2% | match_level_no_match=1552 |
| impala | 1,992 | 1,928 | 1,313 | 68.1% | match_level_no_match=297 |
| jaguar | 617 | 383 | 290 | 75.7% | match_level_no_match=64 |
| japanese macaque | 353 | 305 | 73 | 23.9% | low_speciesnet_confidence=88 |
| kangaroo family | 5,307 | 4,914 | 950 | 19.3% | match_level_no_match=2693 |
| kinkajou | 279 | 105 | 1 | 1.0% | match_level_no_match=52 |
| kirk's dik-dik | 79 | 65 | 45 | 69.2% | match_level_no_match=10 |
| klipspringer | 805 | 473 | 44 | 9.3% | match_level_no_match=172 |
| koala | 1,755 | 1,560 | 5 | 0.3% | match_level_no_match=1094 |
| kob | 175 | 155 | 39 | 25.2% | family_mismatch_high_confidence=39 |
| leaf monkeys genus | 440 | 392 | 37 | 9.4% | match_level_no_match=241 |
| leopard | 969 | 845 | 444 | 52.5% | match_level_no_match=285 |
| leopard cat | 125 | 45 | 8 | 17.8% | match_level_no_match=16 |
| leopardus species | 445 | 211 | 40 | 19.0% | match_level_no_match=67 |
| lion | 2,776 | 2,470 | 1,188 | 48.1% | match_level_no_match=576 |
| llama genus | 1,586 | 1,520 | 38 | 2.5% | match_level_order=651 |
| lowland tapir | 466 | 194 | 61 | 31.4% | match_level_class=54 |
| lycalopex species | 1,429 | 1,270 | 366 | 28.8% | family_mismatch_high_confidence=285 |
| macaque species | 6,926 | 6,072 | 1,770 | 29.2% | match_level_no_match=1868 |
| malay tapir | 29 | 17 | 6 | 35.3% | match_level_no_match=5 |
| maned wolf | 219 | 54 | 18 | 33.3% | low_speciesnet_confidence=12 |
| mangabeys genus | 45 | 32 | 9 | 28.1% | match_level_no_match=8 |
| martes species | 2,385 | 975 | 97 | 9.9% | match_level_no_match=439 |
| meerkat | 335 | 260 | 1 | 0.4% | match_level_no_match=112 |
| mongoose family | 2,469 | 2,181 | 113 | 5.2% | match_level_no_match=1391 |
| moose | 3,779 | 3,254 | 1,042 | 32.0% | match_level_no_match=1160 |
| mountain goat | 898 | 629 | 72 | 11.4% | match_level_no_match=208 |
| mountain zebra | 429 | 325 | 257 | 79.1% | match_level_no_match=59 |
| mule deer | 19,922 | 17,901 | 12,535 | 70.0% | match_level_no_match=4127 |
| muntjac genus | 1,382 | 835 | 203 | 24.3% | match_level_no_match=268 |
| muridae family | 4,018 | 3,181 | 48 | 1.5% | match_level_no_match=2164 |
| muskrat | 3,412 | 3,093 | 48 | 1.6% | match_level_no_match=1641 |
| nilgai | 504 | 340 | 43 | 12.6% | match_level_no_match=92 |
| nine-banded armadillo | 373 | 191 | 92 | 48.2% | match_level_no_match=82 |
| north american porcupine | 2,077 | 1,752 | 439 | 25.1% | match_level_no_match=879 |
| north american river otter | 2,857 | 2,630 | 437 | 16.6% | match_level_no_match=1346 |
| northern chamois | 1,533 | 1,069 | 238 | 22.3% | match_level_no_match=400 |
| northern raccoon | 8,238 | 5,583 | 2,100 | 37.6% | match_level_no_match=2129 |
| nutria | 3,218 | 2,934 | 836 | 28.5% | match_level_no_match=1326 |
| nyala | 748 | 658 | 185 | 28.1% | low_speciesnet_confidence=147 |
| ocelot | 327 | 75 | 31 | 41.3% | match_level_no_match=16 |
| old world porcupine family | 1,620 | 190 | 34 | 17.9% | match_level_no_match=116 |
| opossum family | 4,961 | 3,157 | 694 | 22.0% | match_level_no_match=1845 |
| pangolin family | 89 | 30 | 2 | 6.7% | match_level_no_match=23 |
| patas monkey | 113 | 96 | 6 | 6.2% | match_level_no_match=38 |
| pikas genus | 1,549 | 1,243 | 26 | 2.1% | match_level_no_match=504 |
| plains zebra | 2,513 | 2,402 | 2,026 | 84.3% | match_level_no_match=270 |
| pronghorn | 2,182 | 2,099 | 937 | 44.6% | match_level_no_match=560 |
| puma | 2,032 | 383 | 148 | 38.6% | match_level_no_match=105 |
| quokka | 270 | 235 | 6 | 2.6% | match_level_class=103 |
| raccoon dog | 352 | 128 | 2 | 1.6% | match_level_no_match=45 |
| rattus genus | 5,146 | 3,799 | 62 | 1.6% | match_level_no_match=1799 |
| red brocket | 72 | 34 | 6 | 17.6% | match_level_no_match=10 |
| red deer | 1,566 | 1,405 | 636 | 45.3% | match_level_no_match=352 |
| red fox | 9,763 | 8,213 | 3,348 | 40.8% | match_level_no_match=2647 |
| red kangaroo | 286 | 237 | 25 | 10.5% | match_level_class=78 |
| red panda | 139 | 117 | 17 | 14.5% | match_level_no_match=43 |
| red river hog | 51 | 35 | 5 | 14.3% | family_mismatch_high_confidence=9 |
| red squirrel | 7,753 | 7,283 | 2,303 | 31.6% | match_level_no_match=2594 |
| reedbuck genus | 386 | 302 | 40 | 13.2% | match_level_order=117 |
| reindeer | 1,286 | 675 | 83 | 12.3% | match_level_no_match=166 |
| rhinoceros family | 1,824 | 1,410 | 33 | 2.3% | match_level_no_match=1177 |
| ring-tailed lemur | 334 | 279 | 0 | 0.0% | match_level_no_match=92 |
| ringtail | 656 | 136 | 7 | 5.1% | match_level_no_match=66 |
| roan antelope | 194 | 163 | 22 | 13.5% | match_level_no_match=53 |
| rock hyrax | 1,547 | 1,094 | 5 | 0.5% | match_level_no_match=405 |
| sable antelope | 323 | 266 | 63 | 23.7% | family_mismatch_high_confidence=89 |
| saguinus species | 1,165 | 980 | 3 | 0.3% | match_level_no_match=694 |
| saiga | 35 | 29 | 0 | 0.0% | match_level_no_match=22 |
| saimiri species | 1,371 | 1,120 | 113 | 10.1% | match_level_no_match=715 |
| sambar | 1,036 | 663 | 217 | 32.7% | family_mismatch_high_confidence=199 |
| sea otter | 1,312 | 1,237 | 0 | 0.0% | match_level_no_match=901 |
| serval | 164 | 108 | 40 | 37.0% | family_mismatch_high_confidence=29 |
| short-beaked echidna | 1,660 | 1,550 | 950 | 61.3% | match_level_no_match=503 |
| sika deer | 1,188 | 868 | 262 | 30.2% | family_mismatch_high_confidence=286 |
| sloth bear | 155 | 90 | 11 | 12.2% | family_mismatch_high_confidence=33 |
| snow leopard | 50 | 31 | 21 | 67.7% | match_level_no_match=5 |
| south american coati | 905 | 751 | 310 | 41.3% | match_level_no_match=206 |
| spectacled bear | 208 | 155 | 13 | 8.4% | match_level_no_match=67 |
| spilogale species | 142 | 32 | 2 | 6.2% | low_speciesnet_confidence=12 |
| spotted hyaena | 1,300 | 985 | 496 | 50.4% | match_level_no_match=235 |
| springbok | 768 | 545 | 182 | 33.4% | match_level_no_match=151 |
| squirrel family | 31,089 | 29,316 | 7,422 | 25.3% | match_level_no_match=15057 |
| steenbok | 854 | 687 | 122 | 17.8% | match_level_no_match=199 |
| striped hyaena | 73 | 50 | 6 | 12.0% | match_level_order=23 |
| striped skunk | 2,367 | 630 | 95 | 15.1% | match_level_no_match=369 |
| sun bear | 82 | 53 | 11 | 20.8% | low_speciesnet_confidence=16 |
| swamp wallaby | 1,316 | 1,213 | 362 | 29.8% | match_level_no_match=475 |
| tayra | 281 | 190 | 38 | 20.0% | match_level_no_match=82 |
| thomson's gazelle | 251 | 220 | 155 | 70.5% | match_level_no_match=29 |
| tiger | 737 | 533 | 315 | 59.1% | match_level_no_match=172 |
| vervet monkey | 1,414 | 1,355 | 544 | 40.1% | match_level_no_match=515 |
| walrus | 230 | 136 | 0 | 0.0% | match_level_no_match=107 |
| water deer | 100 | 69 | 5 | 7.2% | match_level_no_match=29 |
| waterbuck | 1,281 | 1,136 | 331 | 29.1% | match_level_order=276 |
| weasel species | 1,495 | 1,234 | 54 | 4.4% | match_level_no_match=528 |
| western gray squirrel | 2,048 | 1,866 | 937 | 50.2% | match_level_no_match=639 |
| white-nosed coati | 2,165 | 2,007 | 636 | 31.7% | match_level_no_match=497 |
| white-tailed deer | 27,905 | 24,206 | 17,034 | 70.4% | match_level_no_match=5835 |
| wild boar | 3,196 | 2,276 | 816 | 35.9% | match_level_no_match=860 |
| wild cat | 193 | 76 | 38 | 50.0% | match_level_no_match=20 |
| wolverine | 124 | 28 | 5 | 17.9% | match_level_no_match=15 |
| woodchuck | 4,045 | 3,657 | 1,060 | 29.0% | match_level_no_match=1766 |
| yak | 128 | 90 | 25 | 27.8% | match_level_class=31 |
| yellow-bellied marmot | 1,756 | 1,682 | 615 | 36.6% | match_level_no_match=733 |

### Classes Ranked by Passed Images

| Rank | Class | Passed |
|---:|---|---:|
| 1 | white-tailed deer | 17,034 |
| 2 | eastern gray squirrel | 15,529 |
| 3 | mule deer | 12,535 |
| 4 | squirrel family | 7,422 |
| 5 | eastern fox squirrel | 6,324 |
| 6 | eastern cottontail | 3,567 |
| 7 | domestic cat | 3,496 |
| 8 | red fox | 3,348 |
| 9 | elk | 3,303 |
| 10 | coyote | 2,934 |
| 11 | african elephant | 2,767 |
| 12 | eurasian red squirrel | 2,555 |
| 13 | red squirrel | 2,303 |
| 14 | cottontail rabbits genus | 2,148 |
| 15 | northern raccoon | 2,100 |
| 16 | plains zebra | 2,026 |
| 17 | macaque species | 1,770 |
| 18 | chipmunk genus | 1,686 |
| 19 | european roe deer | 1,455 |
| 20 | domestic cattle | 1,441 |
| 21 | impala | 1,313 |
| 22 | american black bear | 1,198 |
| 23 | lion | 1,188 |
| 24 | california ground squirrel | 1,147 |
| 25 | woodchuck | 1,060 |
| 26 | moose | 1,042 |
| 27 | hares and jackrabbits genus | 958 |
| 28 | kangaroo family | 950 |
| 29 | short-beaked echidna | 950 |
| 30 | pronghorn | 937 |
| 31 | western gray squirrel | 937 |
| 32 | european rabbit | 917 |
| 33 | american bison | 903 |
| 34 | common warthog | 881 |
| 35 | nutria | 836 |
| 36 | bobcat | 821 |
| 37 | wild boar | 816 |
| 38 | baboon genus | 769 |
| 39 | common wildebeest | 708 |
| 40 | domestic dog | 708 |
| 41 | opossum family | 694 |
| 42 | domestic horse | 665 |
| 43 | arizona black-tailed prairie dog | 662 |
| 44 | red deer | 636 |
| 45 | white-nosed coati | 636 |
| 46 | agouti genus | 618 |
| 47 | yellow-bellied marmot | 615 |
| 48 | european hare | 583 |
| 49 | golden mantled ground squirrel | 571 |
| 50 | vervet monkey | 544 |
| 51 | eastern grey kangaroo | 533 |
| 52 | african buffalo | 529 |
| 53 | spotted hyaena | 496 |
| 54 | capybara | 487 |
| 55 | bighorn sheep | 465 |
| 56 | greater kudu | 457 |
| 57 | leopard | 444 |
| 58 | cheetah | 443 |
| 59 | collared peccary | 441 |
| 60 | north american porcupine | 439 |
| 61 | north american river otter | 437 |
| 62 | brown bear | 430 |
| 63 | grey fox | 414 |
| 64 | lycalopex species | 366 |
| 65 | swamp wallaby | 362 |
| 66 | domestic sheep | 359 |
| 67 | waterbuck | 331 |
| 68 | common fallow deer | 323 |
| 69 | alpine ibex | 317 |
| 70 | tiger | 315 |
| 71 | south american coati | 310 |
| 72 | jaguar | 290 |
| 73 | alpine marmot | 285 |
| 74 | common eland | 263 |
| 75 | hartebeest | 263 |
| 76 | sika deer | 262 |
| 77 | mountain zebra | 257 |
| 78 | gemsbok | 252 |
| 79 | hippopotamus | 241 |
| 80 | northern chamois | 238 |
| 81 | domestic donkey | 228 |
| 82 | sambar | 217 |
| 83 | cercopithecus species | 213 |
| 84 | muntjac genus | 203 |
| 85 | golden jackal | 202 |
| 86 | chital | 196 |
| 87 | nyala | 185 |
| 88 | springbok | 182 |
| 89 | beaver genus | 162 |
| 90 | thomson's gazelle | 155 |
| 91 | howler monkey genus | 152 |
| 92 | puma | 148 |
| 93 | grant's gazelle | 146 |
| 94 | blesbok | 130 |
| 95 | grey wolf | 125 |
| 96 | steenbok | 122 |
| 97 | giraffe | 119 |
| 98 | mongoose family | 113 |
| 99 | saimiri species | 113 |
| 100 | asian elephant | 106 |
| 101 | common wombat | 105 |
| 102 | grevy's zebra | 105 |
| 103 | martes species | 97 |
| 104 | striped skunk | 95 |
| 105 | nine-banded armadillo | 92 |
| 106 | reindeer | 83 |
| 107 | african wild dog | 81 |
| 108 | giant otter | 78 |
| 109 | bornean orangutan | 75 |
| 110 | japanese macaque | 73 |
| 111 | mountain goat | 72 |
| 112 | eurasian badger | 64 |
| 113 | sable antelope | 63 |
| 114 | colobus species | 62 |
| 115 | rattus genus | 62 |
| 116 | lowland tapir | 61 |
| 117 | giant anteater | 56 |
| 118 | weasel species | 54 |
| 119 | cebus species | 53 |
| 120 | gorilla species | 49 |
| 121 | baird's tapir | 48 |
| 122 | muridae family | 48 |
| 123 | muskrat | 48 |
| 124 | dromedary camel | 47 |
| 125 | european bison | 47 |
| 126 | domestic water buffalo | 45 |
| 127 | kirk's dik-dik | 45 |
| 128 | common duiker | 44 |
| 129 | klipspringer | 44 |
| 130 | eurasian lynx | 43 |
| 131 | nilgai | 43 |
| 132 | leopardus species | 40 |
| 133 | reedbuck genus | 40 |
| 134 | serval | 40 |
| 135 | kob | 39 |
| 136 | llama genus | 38 |
| 137 | tayra | 38 |
| 138 | wild cat | 38 |
| 139 | gerenuk | 37 |
| 140 | leaf monkeys genus | 37 |
| 141 | american badger | 35 |
| 142 | ateles species | 35 |
| 143 | old world porcupine family | 34 |
| 144 | rhinoceros family | 33 |
| 145 | ocelot | 31 |
| 146 | bushbuck | 29 |
| 147 | canada lynx | 29 |
| 148 | black wildebeest | 26 |
| 149 | pikas genus | 26 |
| 150 | red kangaroo | 25 |
| 151 | yak | 25 |
| 152 | cricetidae family | 24 |
| 153 | dhole | 23 |
| 154 | roan antelope | 22 |
| 155 | snow leopard | 21 |
| 156 | maned wolf | 18 |
| 157 | eurasian otter | 17 |
| 158 | red panda | 17 |
| 159 | fisher | 15 |
| 160 | asiatic black bear | 14 |
| 161 | chimpanzee | 14 |
| 162 | spectacled bear | 13 |
| 163 | bat-eared fox | 12 |
| 164 | sloth bear | 11 |
| 165 | sun bear | 11 |
| 166 | bongo | 10 |
| 167 | mangabeys genus | 9 |
| 168 | caracal | 8 |
| 169 | leopard cat | 8 |
| 170 | giant panda | 7 |
| 171 | ringtail | 7 |
| 172 | genet genus | 6 |
| 173 | glaucomys species | 6 |
| 174 | malay tapir | 6 |
| 175 | patas monkey | 6 |
| 176 | quokka | 6 |
| 177 | red brocket | 6 |
| 178 | striped hyaena | 6 |
| 179 | koala | 5 |
| 180 | red river hog | 5 |
| 181 | rock hyrax | 5 |
| 182 | water deer | 5 |
| 183 | wolverine | 5 |
| 184 | african civet | 3 |
| 185 | asiatic wild ass | 3 |
| 186 | callithrix species | 3 |
| 187 | fossa | 3 |
| 188 | honey badger | 3 |
| 189 | saguinus species | 3 |
| 190 | cephalophus species | 2 |
| 191 | drill | 2 |
| 192 | pangolin family | 2 |
| 193 | raccoon dog | 2 |
| 194 | spilogale species | 2 |
| 195 | brown hyaena | 1 |
| 196 | eulemur species | 1 |
| 197 | hedgehog family | 1 |
| 198 | kinkajou | 1 |
| 199 | meerkat | 1 |
| 200 | aardvark | 0 |
| 201 | aye-aye | 0 |
| 202 | binturong | 0 |
| 203 | blackbuck | 0 |
| 204 | brown-throated sloth | 0 |
| 205 | callicebus genus | 0 |
| 206 | clouded leopard | 0 |
| 207 | eared seals | 0 |
| 208 | elephant seal | 0 |
| 209 | giant armadillo | 0 |
| 210 | hoffmann's two-toed sloth | 0 |
| 211 | hog badger genus | 0 |
| 212 | ring-tailed lemur | 0 |
| 213 | saiga | 0 |
| 214 | sea otter | 0 |
| 215 | walrus | 0 |

## wikimedia — 12,486 records

| | Count | % |
|---|---:|---:|
| Pass | 5,076 | 40.7% |
| Fail | 7,410 | 59.3% |

### Fail Reasons

| Reason | Count | % |
|---|---:|---:|
| low_speciesnet_confidence | 1,800 | 14.4% |
| not_in_225_classes | 55 | 0.4% |
| family_mismatch_high_confidence | 937 | 7.5% |
| match_level_order | 805 | 6.4% |
| match_level_class | 1,636 | 13.1% |
| match_level_no_match | 2,177 | 17.4% |

### Match Levels (10,631 images with valid classification)

| Level | Count | % |
|---|---:|---:|
| species | 4,120 | 38.8% |
| genus | 509 | 4.8% |
| family | 1,384 | 13.0% |
| order | 805 | 7.6% |
| class | 1,636 | 15.4% |
| no_match | 2,177 | 20.5% |

**Multi-animal images:** 3,807 (30.5%)

**prob\_225\_sum** (10,631 images with valid classification): mean=0.763  median=0.898  p10=0.166  p90=0.989  zeros=495

### Per-Class Breakdown (213 classes)

| Class | Pre-filter | SN Input | Pass | Pass% | Top Fail Reason |
|---|---:|---:|---:|---:|---|
| aardvark | 25 | 14 | 0 | 0.0% | match_level_class=7 |
| aardwolf | 34 | 32 | 8 | 25.0% | family_mismatch_high_confidence=8 |
| african buffalo | 379 | 308 | 106 | 34.4% | family_mismatch_high_confidence=121 |
| african civet | 10 | 8 | 2 | 25.0% | low_speciesnet_confidence=3 |
| african elephant | 510 | 428 | 343 | 80.1% | match_level_class=52 |
| african wild dog | 413 | 386 | 96 | 24.9% | match_level_class=91 |
| agouti genus | 54 | 44 | 15 | 34.1% | low_speciesnet_confidence=13 |
| alpine ibex | 65 | 55 | 18 | 32.7% | family_mismatch_high_confidence=20 |
| alpine marmot | 37 | 30 | 13 | 43.3% | match_level_no_match=6 |
| american badger | 17 | 12 | 2 | 16.7% | low_speciesnet_confidence=7 |
| american bison | 201 | 98 | 63 | 64.3% | family_mismatch_high_confidence=18 |
| american black bear | 88 | 49 | 31 | 63.3% | match_level_no_match=9 |
| american mink | 45 | 21 | 3 | 14.3% | low_speciesnet_confidence=8 |
| arizona black-tailed prairie dog | 53 | 50 | 36 | 72.0% | match_level_class=5 |
| asian elephant | 265 | 130 | 29 | 22.3% | family_mismatch_high_confidence=74 |
| asiatic black bear | 53 | 32 | 15 | 46.9% | match_level_class=8 |
| asiatic wild ass | 15 | 7 | 3 | 42.9% | match_level_no_match=2 |
| ateles species | 52 | 36 | 1 | 2.8% | low_speciesnet_confidence=12 |
| aye-aye | 10 | 6 | 0 | 0.0% | low_speciesnet_confidence=4 |
| baboon genus | 122 | 95 | 33 | 34.7% | match_level_class=21 |
| bairds tapir | 9 | 9 | 0 | 0.0% | not_in_225_classes=9 |
| bat-eared fox | 7 | 6 | 1 | 16.7% | low_speciesnet_confidence=3 |
| beaver genus | 36 | 5 | 0 | 0.0% | match_level_class=2 |
| bighorn sheep | 50 | 38 | 13 | 34.2% | match_level_order=10 |
| binturong | 11 | 9 | 0 | 0.0% | match_level_class=5 |
| black wildebeest | 7 | 4 | 0 | 0.0% | match_level_class=2 |
| black-backed jackal | 47 | 40 | 21 | 52.5% | low_speciesnet_confidence=5 |
| blackbuck | 21 | 18 | 0 | 0.0% | match_level_no_match=13 |
| blesbok | 17 | 15 | 3 | 20.0% | low_speciesnet_confidence=5 |
| bobcat | 24 | 14 | 12 | 85.7% | family_mismatch_high_confidence=2 |
| bongo | 5 | 5 | 1 | 20.0% | family_mismatch_high_confidence=2 |
| bornean orangutan | 47 | 40 | 30 | 75.0% | match_level_no_match=6 |
| brown bear | 218 | 152 | 60 | 39.5% | match_level_class=43 |
| brown hyaena | 11 | 7 | 2 | 28.6% | family_mismatch_high_confidence=1 |
| brown-throated sloth | 20 | 15 | 0 | 0.0% | match_level_no_match=7 |
| bushbuck | 3 | 3 | 1 | 33.3% | low_speciesnet_confidence=1 |
| california ground squirrel | 12 | 10 | 2 | 20.0% | match_level_no_match=3 |
| callicebus genus | 8 | 7 | 0 | 0.0% | match_level_order=2 |
| callithrix species | 58 | 49 | 0 | 0.0% | match_level_no_match=18 |
| canada lynx | 8 | 2 | 2 | 100.0% |  |
| capybara | 52 | 40 | 18 | 45.0% | match_level_class=8 |
| caracal | 14 | 11 | 1 | 9.1% | family_mismatch_high_confidence=4 |
| cebus species | 95 | 68 | 2 | 2.9% | match_level_order=21 |
| cephalophus species | 16 | 10 | 4 | 40.0% | low_speciesnet_confidence=3 |
| cercopithecus species | 78 | 66 | 21 | 31.8% | match_level_no_match=14 |
| cheetah | 129 | 106 | 95 | 89.6% | family_mismatch_high_confidence=5 |
| chimpanzee | 77 | 45 | 9 | 20.0% | match_level_class=14 |
| chipmunk genus | 103 | 74 | 16 | 21.6% | low_speciesnet_confidence=26 |
| chital | 101 | 80 | 18 | 22.5% | family_mismatch_high_confidence=39 |
| clouded leopard | 9 | 6 | 0 | 0.0% | family_mismatch_high_confidence=4 |
| collared peccary | 29 | 25 | 14 | 56.0% | match_level_order=5 |
| colobus species | 32 | 23 | 5 | 21.7% | low_speciesnet_confidence=7 |
| common duiker | 2 | 2 | 1 | 50.0% | match_level_order=1 |
| common eland | 15 | 14 | 7 | 50.0% | family_mismatch_high_confidence=4 |
| common fallow deer | 194 | 158 | 69 | 43.7% | family_mismatch_high_confidence=44 |
| common warthog | 60 | 51 | 29 | 56.9% | match_level_order=8 |
| common wildebeest | 115 | 78 | 39 | 50.0% | match_level_order=10 |
| common wombat | 20 | 13 | 3 | 23.1% | match_level_class=7 |
| cottontail rabbits genus | 64 | 52 | 20 | 38.5% | low_speciesnet_confidence=15 |
| coyote | 109 | 90 | 69 | 76.7% | low_speciesnet_confidence=7 |
| cricetidae family | 244 | 159 | 0 | 0.0% | match_level_no_match=102 |
| dhole | 33 | 22 | 11 | 50.0% | match_level_class=7 |
| dingo | 24 | 16 | 10 | 62.5% | low_speciesnet_confidence=2 |
| domestic cat | 321 | 206 | 107 | 51.9% | match_level_class=34 |
| domestic cattle | 30 | 17 | 16 | 94.1% | family_mismatch_high_confidence=1 |
| domestic dog | 400 | 221 | 131 | 59.3% | match_level_class=47 |
| domestic donkey | 230 | 110 | 37 | 33.6% | match_level_class=49 |
| domestic goat | 246 | 181 | 72 | 39.8% | family_mismatch_high_confidence=46 |
| domestic horse | 2,011 | 752 | 446 | 59.3% | match_level_class=239 |
| domestic pig | 17 | 15 | 10 | 66.7% | match_level_no_match=2 |
| domestic sheep | 298 | 202 | 96 | 47.5% | family_mismatch_high_confidence=44 |
| domestic water buffalo | 67 | 23 | 1 | 4.3% | family_mismatch_high_confidence=17 |
| dromedary camel | 135 | 71 | 23 | 32.4% | match_level_order=19 |
| eared seals | 411 | 272 | 0 | 0.0% | match_level_no_match=222 |
| eastern fox squirrel | 10 | 8 | 7 | 87.5% | match_level_no_match=1 |
| eastern gray squirrel | 129 | 107 | 30 | 28.0% | low_speciesnet_confidence=30 |
| eastern grey kangaroo | 78 | 62 | 20 | 32.3% | match_level_class=27 |
| elephant seal | 99 | 77 | 0 | 0.0% | match_level_no_match=66 |
| elk | 150 | 112 | 87 | 77.7% | match_level_order=11 |
| eulemur species | 80 | 65 | 0 | 0.0% | low_speciesnet_confidence=20 |
| eurasian badger | 36 | 15 | 8 | 53.3% | match_level_order=3 |
| eurasian lynx | 56 | 36 | 25 | 69.4% | match_level_no_match=3 |
| eurasian otter | 38 | 29 | 2 | 6.9% | match_level_class=8 |
| eurasian red squirrel | 345 | 314 | 58 | 18.5% | low_speciesnet_confidence=109 |
| european bison | 39 | 29 | 6 | 20.7% | family_mismatch_high_confidence=18 |
| european hare | 51 | 43 | 19 | 44.2% | match_level_no_match=9 |
| european rabbit | 38 | 29 | 8 | 27.6% | match_level_class=10 |
| european roe deer | 128 | 102 | 37 | 36.3% | family_mismatch_high_confidence=46 |
| fisher | 31 | 3 | 0 | 0.0% | match_level_no_match=1 |
| fossa | 15 | 10 | 1 | 10.0% | match_level_order=4 |
| gemsbok | 17 | 13 | 11 | 84.6% | match_level_no_match=1 |
| genet genus | 30 | 10 | 2 | 20.0% | match_level_no_match=3 |
| gerenuk | 7 | 6 | 4 | 66.7% | family_mismatch_high_confidence=2 |
| giant anteater | 16 | 13 | 5 | 38.5% | match_level_class=6 |
| giant otter | 17 | 14 | 3 | 21.4% | match_level_class=6 |
| giant panda | 112 | 81 | 34 | 42.0% | match_level_class=13 |
| giraffe | 31 | 23 | 20 | 87.0% | match_level_class=2 |
| glaucomys species | 12 | 8 | 0 | 0.0% | low_speciesnet_confidence=4 |
| golden jackal | 69 | 58 | 32 | 55.2% | low_speciesnet_confidence=12 |
| golden mantled ground squirrel | 18 | 17 | 7 | 41.2% | match_level_class=4 |
| gorilla species | 137 | 93 | 12 | 12.9% | match_level_class=43 |
| grants gazelle | 9 | 7 | 0 | 0.0% | not_in_225_classes=7 |
| greater kudu | 72 | 61 | 18 | 29.5% | match_level_order=16 |
| grevys zebra | 17 | 16 | 0 | 0.0% | not_in_225_classes=16 |
| grey fox | 6 | 3 | 1 | 33.3% | match_level_no_match=1 |
| grey wolf | 231 | 150 | 89 | 59.3% | match_level_class=21 |
| hares and jackrabbits genus | 92 | 74 | 13 | 17.6% | match_level_class=20 |
| hartebeest | 29 | 21 | 13 | 61.9% | match_level_order=4 |
| hedgehog family | 112 | 62 | 0 | 0.0% | match_level_no_match=47 |
| hippopotamus | 91 | 46 | 13 | 28.3% | match_level_no_match=20 |
| hoffmanns two-toed sloth | 8 | 7 | 0 | 0.0% | not_in_225_classes=7 |
| hog badger genus | 8 | 6 | 1 | 16.7% | match_level_class=2 |
| honey badger | 14 | 7 | 1 | 14.3% | match_level_class=3 |
| howler monkey genus | 41 | 30 | 4 | 13.3% | low_speciesnet_confidence=8 |
| impala | 118 | 96 | 69 | 71.9% | match_level_order=15 |
| jaguar | 79 | 69 | 57 | 82.6% | match_level_no_match=6 |
| japanese macaque | 57 | 41 | 12 | 29.3% | low_speciesnet_confidence=15 |
| kangaroo family | 281 | 174 | 48 | 27.6% | match_level_no_match=85 |
| kinkajou | 2 | 1 | 0 | 0.0% | match_level_order=1 |
| kirks dik-dik | 11 | 11 | 0 | 0.0% | not_in_225_classes=8 |
| klipspringer | 7 | 7 | 1 | 14.3% | match_level_order=4 |
| koala | 61 | 48 | 2 | 4.2% | match_level_no_match=18 |
| kob | 24 | 17 | 3 | 17.6% | family_mismatch_high_confidence=7 |
| leaf monkeys genus | 138 | 112 | 20 | 17.9% | match_level_no_match=35 |
| leopard | 183 | 144 | 100 | 69.4% | match_level_no_match=20 |
| leopard cat | 31 | 23 | 11 | 47.8% | family_mismatch_high_confidence=6 |
| leopardus species | 26 | 12 | 5 | 41.7% | family_mismatch_high_confidence=4 |
| lion | 485 | 393 | 285 | 72.5% | match_level_class=44 |
| llama genus | 95 | 73 | 7 | 9.6% | match_level_order=38 |
| lowland tapir | 16 | 15 | 9 | 60.0% | match_level_class=5 |
| macaque species | 358 | 265 | 100 | 37.7% | low_speciesnet_confidence=61 |
| malay tapir | 2 | 1 | 1 | 100.0% |  |
| maned wolf | 5 | 4 | 3 | 75.0% | low_speciesnet_confidence=1 |
| mangabeys genus | 12 | 6 | 2 | 33.3% | match_level_no_match=2 |
| martes species | 40 | 27 | 5 | 18.5% | match_level_no_match=8 |
| meerkat | 57 | 54 | 0 | 0.0% | match_level_class=21 |
| mongoose family | 94 | 76 | 5 | 6.6% | match_level_no_match=39 |
| moose | 229 | 123 | 57 | 46.3% | match_level_order=33 |
| mouflon | 7 | 5 | 2 | 40.0% | match_level_order=1 |
| mountain goat | 14 | 11 | 1 | 9.1% | match_level_class=5 |
| mountain zebra | 17 | 12 | 11 | 91.7% | match_level_class=1 |
| mule deer | 105 | 76 | 68 | 89.5% | match_level_no_match=3 |
| muntjac genus | 30 | 25 | 11 | 44.0% | low_speciesnet_confidence=8 |
| muridae family | 328 | 189 | 3 | 1.6% | match_level_no_match=117 |
| muskrat | 2 | 1 | 0 | 0.0% | match_level_no_match=1 |
| nilgai | 19 | 17 | 5 | 29.4% | low_speciesnet_confidence=6 |
| nine-banded armadillo | 8 | 6 | 2 | 33.3% | match_level_class=2 |
| north american porcupine | 28 | 21 | 7 | 33.3% | match_level_no_match=6 |
| north american river otter | 35 | 25 | 9 | 36.0% | match_level_class=8 |
| northern chamois | 42 | 31 | 12 | 38.7% | match_level_order=7 |
| northern raccoon | 68 | 54 | 23 | 42.6% | low_speciesnet_confidence=16 |
| nutria | 39 | 32 | 10 | 31.2% | match_level_no_match=7 |
| nyala | 30 | 30 | 11 | 36.7% | low_speciesnet_confidence=8 |
| old world porcupine family | 32 | 15 | 7 | 46.7% | match_level_no_match=5 |
| opossum family | 74 | 42 | 3 | 7.1% | match_level_no_match=25 |
| pangolin family | 22 | 12 | 0 | 0.0% | match_level_no_match=10 |
| patas monkey | 9 | 7 | 1 | 14.3% | family_mismatch_high_confidence=3 |
| pikas genus | 31 | 24 | 0 | 0.0% | match_level_class=12 |
| pinniped clade | 138 | 55 | 0 | 0.0% | match_level_no_match=43 |
| plains zebra | 247 | 198 | 183 | 92.4% | match_level_class=7 |
| pronghorn | 29 | 26 | 19 | 73.1% | low_speciesnet_confidence=5 |
| puma | 46 | 30 | 16 | 53.3% | family_mismatch_high_confidence=6 |
| raccoon dog | 14 | 13 | 0 | 0.0% | low_speciesnet_confidence=6 |
| rattus genus | 27 | 12 | 0 | 0.0% | match_level_no_match=7 |
| red brocket | 2 | 1 | 0 | 0.0% | match_level_order=1 |
| red deer | 167 | 139 | 104 | 74.8% | family_mismatch_high_confidence=12 |
| red fox | 256 | 206 | 115 | 55.8% | match_level_class=30 |
| red panda | 73 | 71 | 17 | 23.9% | match_level_class=21 |
| red river hog | 18 | 11 | 7 | 63.6% | low_speciesnet_confidence=2 |
| red squirrel | 29 | 29 | 7 | 24.1% | low_speciesnet_confidence=12 |
| red-necked wallaby | 2 | 2 | 1 | 50.0% | match_level_class=1 |
| reedbuck genus | 23 | 17 | 1 | 5.9% | match_level_order=10 |
| reindeer | 212 | 76 | 9 | 11.8% | match_level_order=24 |
| rhinoceros family | 178 | 100 | 7 | 7.0% | match_level_no_match=84 |
| ring-tailed lemur | 57 | 52 | 0 | 0.0% | match_level_class=21 |
| ringtail | 10 | 7 | 0 | 0.0% | low_speciesnet_confidence=6 |
| roan antelope | 19 | 16 | 6 | 37.5% | match_level_order=4 |
| rock hyrax | 24 | 14 | 0 | 0.0% | match_level_class=8 |
| sable antelope | 10 | 6 | 2 | 33.3% | family_mismatch_high_confidence=3 |
| saguinus species | 22 | 18 | 0 | 0.0% | match_level_class=6 |
| saiga | 7 | 4 | 0 | 0.0% | match_level_no_match=3 |
| saimiri species | 30 | 25 | 6 | 24.0% | low_speciesnet_confidence=6 |
| sambar | 22 | 20 | 5 | 25.0% | family_mismatch_high_confidence=5 |
| sea otter | 42 | 32 | 0 | 0.0% | match_level_no_match=26 |
| serval | 40 | 29 | 13 | 44.8% | family_mismatch_high_confidence=12 |
| short-beaked echidna | 18 | 15 | 12 | 80.0% | low_speciesnet_confidence=2 |
| sika deer | 49 | 43 | 13 | 30.2% | family_mismatch_high_confidence=16 |
| sloth bear | 13 | 8 | 2 | 25.0% | family_mismatch_high_confidence=4 |
| snow leopard | 25 | 19 | 10 | 52.6% | family_mismatch_high_confidence=3 |
| south american coati | 37 | 30 | 12 | 40.0% | low_speciesnet_confidence=7 |
| spectacled bear | 11 | 7 | 0 | 0.0% | match_level_class=3 |
| spotted hyaena | 54 | 38 | 23 | 60.5% | match_level_order=5 |
| springbok | 14 | 12 | 5 | 41.7% | family_mismatch_high_confidence=5 |
| squirrel family | 647 | 524 | 180 | 34.4% | match_level_no_match=207 |
| steenbok | 9 | 9 | 2 | 22.2% | match_level_order=3 |
| striped hyaena | 42 | 30 | 9 | 30.0% | family_mismatch_high_confidence=11 |
| striped skunk | 17 | 5 | 3 | 60.0% | low_speciesnet_confidence=1 |
| sun bear | 18 | 15 | 3 | 20.0% | match_level_class=7 |
| swamp wallaby | 8 | 8 | 8 | 100.0% |  |
| tayra | 13 | 9 | 0 | 0.0% | match_level_no_match=4 |
| thomsons gazelle | 12 | 9 | 0 | 0.0% | not_in_225_classes=8 |
| tiger | 236 | 159 | 133 | 83.6% | match_level_class=11 |
| vervet monkey | 36 | 32 | 10 | 31.2% | low_speciesnet_confidence=7 |
| walrus | 35 | 22 | 0 | 0.0% | match_level_no_match=21 |
| water deer | 4 | 2 | 1 | 50.0% | low_speciesnet_confidence=1 |
| waterbuck | 52 | 45 | 17 | 37.8% | match_level_order=11 |
| weasel species | 155 | 103 | 15 | 14.6% | low_speciesnet_confidence=31 |
| white-nosed coati | 15 | 14 | 7 | 50.0% | low_speciesnet_confidence=3 |
| white-tailed deer | 102 | 82 | 62 | 75.6% | match_level_no_match=7 |
| wild boar | 184 | 114 | 62 | 54.4% | match_level_no_match=17 |
| wild cat | 13 | 13 | 5 | 38.5% | match_level_no_match=3 |
| wolverine | 13 | 11 | 1 | 9.1% | match_level_order=4 |
| yak | 57 | 37 | 19 | 51.4% | match_level_class=10 |

### Classes Ranked by Passed Images

| Rank | Class | Passed |
|---:|---|---:|
| 1 | domestic horse | 446 |
| 2 | african elephant | 343 |
| 3 | lion | 285 |
| 4 | plains zebra | 183 |
| 5 | squirrel family | 180 |
| 6 | tiger | 133 |
| 7 | domestic dog | 131 |
| 8 | red fox | 115 |
| 9 | domestic cat | 107 |
| 10 | african buffalo | 106 |
| 11 | red deer | 104 |
| 12 | leopard | 100 |
| 13 | macaque species | 100 |
| 14 | african wild dog | 96 |
| 15 | domestic sheep | 96 |
| 16 | cheetah | 95 |
| 17 | grey wolf | 89 |
| 18 | elk | 87 |
| 19 | domestic goat | 72 |
| 20 | coyote | 69 |
| 21 | impala | 69 |
| 22 | common fallow deer | 69 |
| 23 | mule deer | 68 |
| 24 | american bison | 63 |
| 25 | white-tailed deer | 62 |
| 26 | wild boar | 62 |
| 27 | brown bear | 60 |
| 28 | eurasian red squirrel | 58 |
| 29 | jaguar | 57 |
| 30 | moose | 57 |
| 31 | kangaroo family | 48 |
| 32 | common wildebeest | 39 |
| 33 | domestic donkey | 37 |
| 34 | european roe deer | 37 |
| 35 | arizona black-tailed prairie dog | 36 |
| 36 | giant panda | 34 |
| 37 | baboon genus | 33 |
| 38 | golden jackal | 32 |
| 39 | american black bear | 31 |
| 40 | bornean orangutan | 30 |
| 41 | eastern gray squirrel | 30 |
| 42 | asian elephant | 29 |
| 43 | common warthog | 29 |
| 44 | eurasian lynx | 25 |
| 45 | dromedary camel | 23 |
| 46 | northern raccoon | 23 |
| 47 | spotted hyaena | 23 |
| 48 | cercopithecus species | 21 |
| 49 | black-backed jackal | 21 |
| 50 | cottontail rabbits genus | 20 |
| 51 | leaf monkeys genus | 20 |
| 52 | eastern grey kangaroo | 20 |
| 53 | giraffe | 20 |
| 54 | european hare | 19 |
| 55 | yak | 19 |
| 56 | pronghorn | 19 |
| 57 | alpine ibex | 18 |
| 58 | capybara | 18 |
| 59 | chital | 18 |
| 60 | greater kudu | 18 |
| 61 | red panda | 17 |
| 62 | waterbuck | 17 |
| 63 | chipmunk genus | 16 |
| 64 | puma | 16 |
| 65 | domestic cattle | 16 |
| 66 | agouti genus | 15 |
| 67 | weasel species | 15 |
| 68 | asiatic black bear | 15 |
| 69 | collared peccary | 14 |
| 70 | hares and jackrabbits genus | 13 |
| 71 | hippopotamus | 13 |
| 72 | serval | 13 |
| 73 | alpine marmot | 13 |
| 74 | bighorn sheep | 13 |
| 75 | hartebeest | 13 |
| 76 | sika deer | 13 |
| 77 | gorilla species | 12 |
| 78 | bobcat | 12 |
| 79 | japanese macaque | 12 |
| 80 | northern chamois | 12 |
| 81 | short-beaked echidna | 12 |
| 82 | south american coati | 12 |
| 83 | leopard cat | 11 |
| 84 | dhole | 11 |
| 85 | mountain zebra | 11 |
| 86 | muntjac genus | 11 |
| 87 | nyala | 11 |
| 88 | gemsbok | 11 |
| 89 | nutria | 10 |
| 90 | snow leopard | 10 |
| 91 | vervet monkey | 10 |
| 92 | domestic pig | 10 |
| 93 | dingo | 10 |
| 94 | chimpanzee | 9 |
| 95 | reindeer | 9 |
| 96 | striped hyaena | 9 |
| 97 | lowland tapir | 9 |
| 98 | north american river otter | 9 |
| 99 | aardwolf | 8 |
| 100 | european rabbit | 8 |
| 101 | eurasian badger | 8 |
| 102 | swamp wallaby | 8 |
| 103 | llama genus | 7 |
| 104 | rhinoceros family | 7 |
| 105 | eastern fox squirrel | 7 |
| 106 | north american porcupine | 7 |
| 107 | white-nosed coati | 7 |
| 108 | golden mantled ground squirrel | 7 |
| 109 | old world porcupine family | 7 |
| 110 | red river hog | 7 |
| 111 | common eland | 7 |
| 112 | red squirrel | 7 |
| 113 | european bison | 6 |
| 114 | roan antelope | 6 |
| 115 | saimiri species | 6 |
| 116 | leopardus species | 5 |
| 117 | martes species | 5 |
| 118 | mongoose family | 5 |
| 119 | wild cat | 5 |
| 120 | giant anteater | 5 |
| 121 | nilgai | 5 |
| 122 | colobus species | 5 |
| 123 | sambar | 5 |
| 124 | springbok | 5 |
| 125 | gerenuk | 4 |
| 126 | howler monkey genus | 4 |
| 127 | cephalophus species | 4 |
| 128 | american mink | 3 |
| 129 | asiatic wild ass | 3 |
| 130 | muridae family | 3 |
| 131 | opossum family | 3 |
| 132 | striped skunk | 3 |
| 133 | blesbok | 3 |
| 134 | common wombat | 3 |
| 135 | giant otter | 3 |
| 136 | kob | 3 |
| 137 | sun bear | 3 |
| 138 | maned wolf | 3 |
| 139 | african civet | 2 |
| 140 | american badger | 2 |
| 141 | cebus species | 2 |
| 142 | koala | 2 |
| 143 | eurasian otter | 2 |
| 144 | genet genus | 2 |
| 145 | mangabeys genus | 2 |
| 146 | nine-banded armadillo | 2 |
| 147 | sable antelope | 2 |
| 148 | sloth bear | 2 |
| 149 | steenbok | 2 |
| 150 | mouflon | 2 |
| 151 | california ground squirrel | 2 |
| 152 | canada lynx | 2 |
| 153 | brown hyaena | 2 |
| 154 | caracal | 1 |
| 155 | hog badger genus | 1 |
| 156 | ateles species | 1 |
| 157 | domestic water buffalo | 1 |
| 158 | fossa | 1 |
| 159 | mountain goat | 1 |
| 160 | patas monkey | 1 |
| 161 | bat-eared fox | 1 |
| 162 | bongo | 1 |
| 163 | honey badger | 1 |
| 164 | klipspringer | 1 |
| 165 | malay tapir | 1 |
| 166 | reedbuck genus | 1 |
| 167 | wolverine | 1 |
| 168 | bushbuck | 1 |
| 169 | grey fox | 1 |
| 170 | common duiker | 1 |
| 171 | red-necked wallaby | 1 |
| 172 | water deer | 1 |
| 173 | aardvark | 0 |
| 174 | binturong | 0 |
| 175 | callithrix species | 0 |
| 176 | clouded leopard | 0 |
| 177 | cricetidae family | 0 |
| 178 | eared seals | 0 |
| 179 | elephant seal | 0 |
| 180 | hedgehog family | 0 |
| 181 | kinkajou | 0 |
| 182 | pangolin family | 0 |
| 183 | pikas genus | 0 |
| 184 | raccoon dog | 0 |
| 185 | sea otter | 0 |
| 186 | walrus | 0 |
| 187 | aye-aye | 0 |
| 188 | black wildebeest | 0 |
| 189 | blackbuck | 0 |
| 190 | brown-throated sloth | 0 |
| 191 | callicebus genus | 0 |
| 192 | eulemur species | 0 |
| 193 | grevys zebra | 0 |
| 194 | meerkat | 0 |
| 195 | ring-tailed lemur | 0 |
| 196 | ringtail | 0 |
| 197 | rock hyrax | 0 |
| 198 | tayra | 0 |
| 199 | bairds tapir | 0 |
| 200 | beaver genus | 0 |
| 201 | glaucomys species | 0 |
| 202 | hoffmanns two-toed sloth | 0 |
| 203 | kirks dik-dik | 0 |
| 204 | rattus genus | 0 |
| 205 | saguinus species | 0 |
| 206 | saiga | 0 |
| 207 | spectacled bear | 0 |
| 208 | fisher | 0 |
| 209 | pinniped clade | 0 |
| 210 | grants gazelle | 0 |
| 211 | thomsons gazelle | 0 |
| 212 | muskrat | 0 |
| 213 | red brocket | 0 |

## openimages — 7,688 records

| | Count | % |
|---|---:|---:|
| Pass | 3,506 | 45.6% |
| Fail | 4,182 | 54.4% |

### Fail Reasons

| Reason | Count | % |
|---|---:|---:|
| low_speciesnet_confidence | 876 | 11.4% |
| family_mismatch_high_confidence | 848 | 11.0% |
| match_level_order | 378 | 4.9% |
| match_level_class | 991 | 12.9% |
| match_level_no_match | 1,089 | 14.2% |

### Match Levels (6,812 images with valid classification)

| Level | Count | % |
|---|---:|---:|
| species | 2,879 | 42.3% |
| genus | 418 | 6.1% |
| family | 1,057 | 15.5% |
| order | 378 | 5.5% |
| class | 991 | 14.5% |
| no_match | 1,089 | 16.0% |

**Multi-animal images:** 2,402 (31.2%)

**prob\_225\_sum** (6,812 images with valid classification): mean=0.802  median=0.924  p10=0.278  p90=0.992  zeros=239

### Per-Class Breakdown (26 classes)

| Class | Pre-filter | SN Input | Pass | Pass% | Top Fail Reason |
|---|---:|---:|---:|---:|---|
| asian elephant | 492 | 234 | 17 | 7.3% | family_mismatch_high_confidence=178 |
| cheetah | 247 | 198 | 135 | 68.2% | family_mismatch_high_confidence=47 |
| domestic cat | 495 | 318 | 179 | 56.3% | match_level_class=50 |
| domestic cattle | 470 | 246 | 211 | 85.8% | match_level_class=12 |
| domestic donkey | 480 | 198 | 73 | 36.9% | match_level_class=92 |
| domestic goat | 443 | 320 | 154 | 48.1% | family_mismatch_high_confidence=62 |
| domestic horse | 494 | 139 | 83 | 59.7% | match_level_class=41 |
| domestic sheep | 498 | 315 | 187 | 59.4% | family_mismatch_high_confidence=56 |
| european rabbit | 491 | 246 | 37 | 15.0% | match_level_class=140 |
| giraffe | 487 | 325 | 220 | 67.7% | match_level_no_match=48 |
| grey wolf | 498 | 232 | 104 | 44.8% | match_level_class=57 |
| hippopotamus | 387 | 291 | 120 | 41.2% | match_level_no_match=89 |
| koala | 364 | 272 | 5 | 1.8% | match_level_no_match=99 |
| leopard | 497 | 393 | 210 | 53.4% | family_mismatch_high_confidence=138 |
| lion | 495 | 302 | 224 | 74.2% | match_level_class=32 |
| northern raccoon | 306 | 240 | 142 | 59.2% | match_level_no_match=33 |
| plains zebra | 499 | 376 | 337 | 89.6% | match_level_class=21 |
| red fox | 482 | 389 | 181 | 46.5% | family_mismatch_high_confidence=59 |
| red kangaroo | 458 | 334 | 36 | 10.8% | match_level_class=116 |
| red panda | 327 | 281 | 85 | 30.2% | match_level_class=64 |
| sea otter | 485 | 415 | 0 | 0.0% | match_level_no_match=303 |
| squirrel family | 500 | 450 | 206 | 45.8% | match_level_no_match=123 |
| sun bear | 352 | 185 | 14 | 7.6% | family_mismatch_high_confidence=74 |
| tiger | 500 | 307 | 258 | 84.0% | match_level_no_match=17 |
| white-tailed deer | 499 | 380 | 184 | 48.4% | family_mismatch_high_confidence=78 |
| wild boar | 500 | 302 | 104 | 34.4% | match_level_order=94 |

### Classes Ranked by Passed Images

| Rank | Class | Passed |
|---:|---|---:|
| 1 | plains zebra | 337 |
| 2 | tiger | 258 |
| 3 | lion | 224 |
| 4 | giraffe | 220 |
| 5 | domestic cattle | 211 |
| 6 | leopard | 210 |
| 7 | squirrel family | 206 |
| 8 | domestic sheep | 187 |
| 9 | white-tailed deer | 184 |
| 10 | red fox | 181 |
| 11 | domestic cat | 179 |
| 12 | domestic goat | 154 |
| 13 | northern raccoon | 142 |
| 14 | cheetah | 135 |
| 15 | hippopotamus | 120 |
| 16 | grey wolf | 104 |
| 17 | wild boar | 104 |
| 18 | red panda | 85 |
| 19 | domestic horse | 83 |
| 20 | domestic donkey | 73 |
| 21 | european rabbit | 37 |
| 22 | red kangaroo | 36 |
| 23 | asian elephant | 17 |
| 24 | sun bear | 14 |
| 25 | koala | 5 |
| 26 | sea otter | 0 |

## images_cv — 5,501 records

| | Count | % |
|---|---:|---:|
| Pass | 1,630 | 29.6% |
| Fail | 3,871 | 70.4% |

### Fail Reasons

| Reason | Count | % |
|---|---:|---:|
| primary_crop_too_small | 2 | 0.0% |
| low_speciesnet_confidence | 1,116 | 20.3% |
| family_mismatch_high_confidence | 384 | 7.0% |
| match_level_order | 230 | 4.2% |
| match_level_class | 1,099 | 20.0% |
| match_level_no_match | 1,040 | 18.9% |

### Match Levels (4,383 images with valid classification)

| Level | Count | % |
|---|---:|---:|
| species | 803 | 18.3% |
| genus | 613 | 14.0% |
| family | 598 | 13.6% |
| order | 230 | 5.2% |
| class | 1,099 | 25.1% |
| no_match | 1,040 | 23.7% |

**Multi-animal images:** 708 (12.9%)

**prob\_225\_sum** (4,383 images with valid classification): mean=0.691  median=0.840  p10=0.045  p90=0.971  zeros=288

### Per-Class Breakdown (9 classes)

| Class | Pre-filter | SN Input | Pass | Pass% | Top Fail Reason |
|---|---:|---:|---:|---:|---|
| american mink | 1,295 | 636 | 45 | 7.1% | match_level_no_match=281 |
| bornean orangutan | 1,357 | 667 | 123 | 18.4% | match_level_class=228 |
| common wombat | 1,355 | 771 | 105 | 13.6% | low_speciesnet_confidence=273 |
| dingo | 1,436 | 765 | 387 | 50.6% | low_speciesnet_confidence=124 |
| giant panda | 760 | 289 | 47 | 16.3% | match_level_class=77 |
| gorilla species | 1,360 | 772 | 59 | 7.6% | match_level_class=354 |
| leopard | 1,400 | 834 | 538 | 64.5% | family_mismatch_high_confidence=123 |
| red panda | 72 | 65 | 15 | 23.1% | low_speciesnet_confidence=16 |
| snow leopard | 1,286 | 702 | 311 | 44.3% | family_mismatch_high_confidence=169 |

### Classes Ranked by Passed Images

| Rank | Class | Passed |
|---:|---|---:|
| 1 | leopard | 538 |
| 2 | dingo | 387 |
| 3 | snow leopard | 311 |
| 4 | bornean orangutan | 123 |
| 5 | common wombat | 105 |
| 6 | gorilla species | 59 |
| 7 | giant panda | 47 |
| 8 | american mink | 45 |
| 9 | red panda | 15 |
