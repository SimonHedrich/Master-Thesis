# Multi-Animal Contamination Report

**Thresholds:** md_conf≥0.5  sn_score≥0.3  tolerance=family

**Sources:** gbif, inaturalist, wikimedia, openimages, images_cv

---

## Summary

| Metric | Count |
|---|---:|
| Total images classified | 465,130 |
| Images with ≥2 significant detections | 64,835 |
| Expected class not in 225 (skipped) | 1,317 |
| **Flagged images** (≥1 confident mismatch) | **11,354** |
| Uncertain-only images (low-confidence mismatch) | 3,517 |
| Consistent multi-box images | 26,636 |

> **Note:** 11,354 flagged images — this should be FAR below the naive upper bound of 32,401 (images with any different SpeciesNet index).  
Ratio: 1:3 — tolerance band is working as intended.


## Breakdown by Source

| Source | Flagged Images |
|---|---:|
| inaturalist | 8,048 |
| gbif | 1,583 |
| wikimedia | 1,089 |
| openimages | 539 |
| images_cv | 95 |

## Offending Boxes — Match Level Breakdown

| Match Level | Count |
|---|---:|
| order | 7,691 |
| class | 8,442 |
| no_match | 3,597 |

## Offending Box Verdict Breakdown

| Verdict | Count |
|---|---:|
| flag | 14,511 |
| uncertain | 5,219 |

## Top 30 Contaminated Classes (by Flagged Images)

| Rank | Class | Flagged Images |
|---:|---|---:|
| 1 | eared seals | 890 |
| 2 | bighorn sheep | 528 |
| 3 | llama genus | 509 |
| 4 | domestic horse | 449 |
| 5 | african elephant | 299 |
| 6 | macaque species | 269 |
| 7 | eastern grey kangaroo | 264 |
| 8 | domestic donkey | 262 |
| 9 | rhinoceros family | 253 |
| 10 | elk | 237 |
| 11 | kangaroo family | 225 |
| 12 | capybara | 208 |
| 13 | impala | 199 |
| 14 | wild boar | 192 |
| 15 | pronghorn | 187 |
| 16 | common wildebeest | 182 |
| 17 | domestic sheep | 172 |
| 18 | white-tailed deer | 160 |
| 19 | waterbuck | 159 |
| 20 | lion | 152 |
| 21 | hippopotamus | 149 |
| 22 | elephant seal | 145 |
| 23 | greater kudu | 130 |
| 24 | plains zebra | 130 |
| 25 | mule deer | 118 |
| 26 | baboon genus | 114 |
| 27 | european rabbit | 112 |
| 28 | common warthog | 106 |
| 29 | african buffalo | 105 |
| 30 | african wild dog | 100 |

## Projected Per-Class Image-Count Delta

If every flagged image were removed, each class would lose at most this many images.  Actual loss may be lower (reviewer may choose to edit individual boxes rather than discard whole images).

| Class | Flagged Images (max loss) |
|---|---:|
| eared seals | 890 |
| bighorn sheep | 528 |
| llama genus | 509 |
| domestic horse | 449 |
| african elephant | 299 |
| macaque species | 269 |
| eastern grey kangaroo | 264 |
| domestic donkey | 262 |
| rhinoceros family | 253 |
| elk | 237 |
| kangaroo family | 225 |
| capybara | 208 |
| impala | 199 |
| wild boar | 192 |
| pronghorn | 187 |
| common wildebeest | 182 |
| domestic sheep | 172 |
| white-tailed deer | 160 |
| waterbuck | 159 |
| lion | 152 |
| hippopotamus | 149 |
| elephant seal | 145 |
| greater kudu | 130 |
| plains zebra | 130 |
| mule deer | 118 |
| baboon genus | 114 |
| european rabbit | 112 |
| common warthog | 106 |
| african buffalo | 105 |
| african wild dog | 100 |
| dromedary camel | 100 |
| american bison | 99 |
| common eland | 98 |
| domestic cattle | 96 |
| red fox | 94 |
| collared peccary | 90 |
| asian elephant | 88 |
| sea otter | 87 |
| blackbuck | 85 |
| domestic dog | 77 |
| spotted hyaena | 77 |
| alpine ibex | 76 |
| common fallow deer | 75 |
| moose | 75 |
| red kangaroo | 75 |
| reindeer | 75 |
| springbok | 75 |
| blesbok | 73 |
| gorilla species | 69 |
| hartebeest | 68 |
| nyala | 67 |
| mongoose family | 65 |
| red deer | 64 |
| brown bear | 59 |
| white-nosed coati | 59 |
| northern raccoon | 58 |
| domestic cat | 55 |
| gemsbok | 53 |
| ring-tailed lemur | 51 |
| domestic goat | 49 |
| chital | 48 |
| mountain goat | 48 |
| european roe deer | 46 |
| rock hyrax | 44 |
| saimiri species | 44 |
| squirrel family | 44 |
| howler monkey genus | 43 |
| american black bear | 42 |
| klipspringer | 42 |
| muridae family | 42 |
| meerkat | 38 |
| cebus species | 37 |
| nilgai | 37 |
| nutria | 37 |
| vervet monkey | 37 |
| asiatic wild ass | 36 |
| northern chamois | 35 |
| european hare | 34 |
| coyote | 33 |
| rattus genus | 33 |
| sable antelope | 33 |
| sika deer | 33 |
| eastern cottontail | 33 |
| eulemur species | 32 |
| reedbuck genus | 32 |
| domestic water buffalo | 31 |
| kob | 31 |
| arizona black-tailed prairie dog | 30 |
| japanese macaque | 30 |
| south american coati | 30 |
| eastern gray squirrel | 30 |
| north american river otter | 28 |
| bornean orangutan | 27 |
| callithrix species | 27 |
| golden jackal | 27 |
| saguinus species | 27 |
| common wombat | 26 |
| grey wolf | 25 |
| hares and jackrabbits genus | 25 |
| sambar | 24 |
| cricetidae family | 24 |
| giant panda | 23 |
| cottontail rabbits genus | 22 |
| giraffe | 21 |
| giant otter | 20 |
| chimpanzee | 19 |
| muskrat | 19 |
| opossum family | 19 |
| ateles species | 18 |
| walrus | 18 |
| beaver genus | 17 |
| yak | 17 |
| european bison | 16 |
| eurasian red squirrel | 16 |
| weasel species | 15 |
| cercopithecus species | 13 |
| quokka | 13 |
| alpine marmot | 12 |
| leaf monkeys genus | 12 |
| roan antelope | 12 |
| saiga | 12 |
| red panda | 12 |
| dingo | 12 |
| black wildebeest | 11 |
| eastern fox squirrel | 11 |
| grant's gazelle | 11 |
| spectacled bear | 11 |
| thomson's gazelle | 11 |
| swamp wallaby | 11 |
| dhole | 10 |
| koala | 10 |
| tiger | 10 |
| cheetah | 9 |
| mountain zebra | 9 |
| steenbok | 9 |
| agouti genus | 9 |
| bat-eared fox | 8 |
| eurasian otter | 8 |
| lowland tapir | 8 |
| puma | 8 |
| raccoon dog | 8 |
| sun bear | 8 |
| pinniped clade | 8 |
| black-backed jackal | 7 |
| colobus species | 7 |
| lycalopex species | 7 |
| woodchuck | 7 |
| baird's tapir | 6 |
| bushbuck | 6 |
| grevy's zebra | 6 |
| red river hog | 6 |
| hedgehog family | 6 |
| leopard | 6 |
| american mink | 5 |
| striped hyaena | 5 |
| american badger | 4 |
| gerenuk | 4 |
| muntjac genus | 4 |
| sloth bear | 4 |
| asiatic black bear | 4 |
| grey fox | 4 |
| aardvark | 3 |
| caracal | 3 |
| common duiker | 3 |
| kirk's dik-dik | 3 |
| north american porcupine | 3 |
| pikas genus | 3 |
| brown-throated sloth | 2 |
| fossa | 2 |
| honey badger | 2 |
| martes species | 2 |
| red-necked wallaby | 2 |
| water deer | 2 |
| chipmunk genus | 2 |
| eurasian lynx | 2 |
| golden mantled ground squirrel | 2 |
| leopardus species | 2 |
| malay tapir | 2 |
| red squirrel | 2 |
| yellow-bellied marmot | 2 |
| mouflon | 2 |
| domestic pig | 2 |
| wolverine | 2 |
| giant anteater | 1 |
| bobcat | 1 |
| california ground squirrel | 1 |
| callicebus genus | 1 |
| eurasian badger | 1 |
| genet genus | 1 |
| hoffmann's two-toed sloth | 1 |
| old world porcupine family | 1 |
| patas monkey | 1 |
| serval | 1 |
| spilogale species | 1 |
| western gray squirrel | 1 |
| binturong | 1 |
| cephalophus species | 1 |
| striped skunk | 1 |
| brown hyaena | 1 |
| glaucomys species | 1 |
| mangabeys genus | 1 |
