# Dataset Coverage Report

Generated: 2026-04-08  
Target classes: **225**  
Datasets: GBIF · iNaturalist · Wikimedia · Open Images  
Quality buffer applied: **20%** (estimated post-filtering loss)

## Coverage Summary

Ultralytics guideline: **≥1,500 images per class** (≥1,200 after buffer).  
Total passed images across all datasets: **487,398**  
Estimated usable after 20% buffer: **389,917**

| Tier | Range | Classes |
|------|-------|---------|
| 🟢 Excellent | ≥1500 imgs | 82 |
| 🟡 Good | 1000–1499 imgs | 20 |
| 🟠 Marginal | 500–999 imgs | 31 |
| 🔴 Low | 100–499 imgs | 59 |
| ⛔ Critical | <100 imgs | 33 |

## Per-Dataset Summary

| Dataset | Passed | Failed | Total | Pass% | Classes covered |
|---------|-------:|-------:|------:|------:|----------------:|
| gbif | 26,520 | 40,361 | 66,881 | 39.7% | 208 |
| inaturalist | 440,525 | 458,224 | 898,749 | 49.0% | 214 |
| wikimedia | 1,175 | 244 | 1,419 | 82.8% | 104 |
| openimages | 23,507 | 1,793 | 25,300 | 92.9% | 54 |

## All Classes by Coverage

Sorted by total passed images (highest first).  
**Buffer** = total × 0.80 (estimated usable). **Gap** = images still needed to reach 1,200 usable.

| # | Class | Scientific name | iNaturalist | GBIF | Open Images | Wikimedia | **Total** | Buffer | Gap | Status |
|--:|-------|-----------------|------------:|-----:|------------:|----------:|----------:|-------:|----:|--------|
| 1 | eastern gray squirrel | *sciurus carolinensis* | 32,499 | 162 | 477 | 4 | **33,142** | 26,514 | — | 🟢 Excellent |
| 2 | squirrel family | *sciuridae* | 31,089 | 301 | 477 | 7 | **31,874** | 25,499 | — | 🟢 Excellent |
| 3 | white-tailed deer | *odocoileus virginianus* | 27,905 | 149 | 473 | 5 | **28,532** | 22,826 | — | 🟢 Excellent |
| 4 | mule deer | *odocoileus hemionus* | 19,921 | 206 | 473 | 0 | **20,600** | 16,480 | — | 🟢 Excellent |
| 5 | eastern fox squirrel | *sciurus niger* | 12,708 | 205 | 477 | 0 | **13,390** | 10,712 | — | 🟢 Excellent |
| 6 | eastern cottontail | *sylvilagus floridanus* | 12,262 | 0 | 435 | 0 | **12,697** | 10,158 | — | 🟢 Excellent |
| 7 | eurasian red squirrel | *sciurus vulgaris* | 9,930 | 130 | 477 | 2 | **10,539** | 8,431 | — | 🟢 Excellent |
| 8 | red fox | *vulpes vulpes* | 9,760 | 114 | 455 | 3 | **10,332** | 8,266 | — | 🟢 Excellent |
| 9 | eared seals | *otariidae* | 9,872 | 0 | 0 | 11 | **9,883** | 7,906 | — | 🟢 Excellent |
| 10 | northern raccoon | *procyon lotor* | 8,226 | 68 | 280 | 2 | **8,576** | 6,861 | — | 🟢 Excellent |
| 11 | red squirrel | *tamiasciurus hudsonicus* | 7,753 | 126 | 477 | 0 | **8,356** | 6,685 | — | 🟢 Excellent |
| 12 | domestic cat | *felis catus* | 7,731 | 155 | 427 | 0 | **8,313** | 6,650 | — | 🟢 Excellent |
| 13 | coyote | *canis latrans* | 8,024 | 152 | 0 | 7 | **8,183** | 6,546 | — | 🟢 Excellent |
| 14 | cottontail rabbits genus | *sylvilagus* | 7,988 | 66 | 0 | 6 | **8,060** | 6,448 | — | 🟢 Excellent |
| 15 | chipmunk genus | *tamias* | 7,069 | 163 | 477 | 3 | **7,712** | 6,170 | — | 🟢 Excellent |
| 16 | macaque species | *macaca* | 6,926 | 758 | 0 | 7 | **7,691** | 6,153 | — | 🟢 Excellent |
| 17 | european roe deer | *capreolus capreolus* | 6,517 | 149 | 473 | 0 | **7,139** | 5,711 | — | 🟢 Excellent |
| 18 | elk | *cervus canadensis* | 5,800 | 137 | 473 | 4 | **6,414** | 5,131 | — | 🟢 Excellent |
| 19 | kangaroo family | *macropodidae* | 5,307 | 594 | 441 | 9 | **6,351** | 5,081 | — | 🟢 Excellent |
| 20 | european rabbit | *oryctolagus cuniculus* | 5,675 | 108 | 435 | 2 | **6,220** | 4,976 | — | 🟢 Excellent |
| 21 | cricetidae family | *cricetidae* | 6,066 | 3 | 0 | 15 | **6,084** | 4,867 | — | 🟢 Excellent |
| 22 | california ground squirrel | *otospermophilus beecheyi* | 5,559 | 37 | 0 | 0 | **5,596** | 4,477 | — | 🟢 Excellent |
| 23 | hares and jackrabbits genus | *lepus* | 5,206 | 55 | 0 | 7 | **5,268** | 4,214 | — | 🟢 Excellent |
| 24 | rattus genus | *rattus* | 5,146 | 28 | 0 | 1 | **5,175** | 4,140 | — | 🟢 Excellent |
| 25 | opossum family | *didelphidae* | 4,952 | 0 | 0 | 6 | **4,958** | 3,966 | — | 🟢 Excellent |
| 26 | african elephant | *loxodonta africana* | 3,797 | 250 | 473 | 249 | **4,769** | 3,815 | — | 🟢 Excellent |
| 27 | hedgehog family | *erinaceidae* | 4,420 | 0 | 0 | 2 | **4,422** | 3,538 | — | 🟢 Excellent |
| 28 | moose | *alces alces* | 3,777 | 121 | 473 | 7 | **4,378** | 3,502 | — | 🟢 Excellent |
| 29 | american black bear | *ursus americanus* | 3,785 | 134 | 306 | 5 | **4,230** | 3,384 | — | 🟢 Excellent |
| 30 | woodchuck | *marmota monax* | 4,045 | 173 | 0 | 0 | **4,218** | 3,374 | — | 🟢 Excellent |
| 31 | muridae family | *muridae* | 4,010 | 0 | 0 | 11 | **4,021** | 3,217 | — | 🟢 Excellent |
| 32 | beaver genus | *castor* | 3,776 | 16 | 0 | 0 | **3,792** | 3,034 | — | 🟢 Excellent |
| 33 | wild boar | *sus scrofa* | 3,196 | 123 | 460 | 3 | **3,782** | 3,026 | — | 🟢 Excellent |
| 34 | lion | *panthera leo* | 2,776 | 236 | 471 | 8 | **3,491** | 2,793 | — | 🟢 Excellent |
| 35 | muskrat | *ondatra zibethicus* | 3,412 | 67 | 0 | 0 | **3,479** | 2,783 | — | 🟢 Excellent |
| 36 | nutria | *myocastor coypus* | 3,218 | 175 | 0 | 1 | **3,394** | 2,715 | — | 🟢 Excellent |
| 37 | european hare | *lepus europaeus* | 3,371 | 0 | 0 | 2 | **3,373** | 2,698 | — | 🟢 Excellent |
| 38 | north american river otter | *lontra canadensis* | 2,854 | 60 | 459 | 0 | **3,373** | 2,698 | — | 🟢 Excellent |
| 39 | bighorn sheep | *ovis canadensis* | 3,118 | 163 | 0 | 0 | **3,281** | 2,625 | — | 🟢 Excellent |
| 40 | plains zebra | *equus quagga* | 2,501 | 255 | 457 | 3 | **3,216** | 2,573 | — | 🟢 Excellent |
| 41 | eastern grey kangaroo | *macropus giganteus* | 2,569 | 183 | 441 | 0 | **3,193** | 2,554 | — | 🟢 Excellent |
| 42 | howler monkey genus | *alouatta* | 3,017 | 28 | 0 | 1 | **3,046** | 2,437 | — | 🟢 Excellent |
| 43 | domestic dog | *canis familiaris* | 2,101 | 181 | 446 | 0 | **2,728** | 2,182 | — | 🟢 Excellent |
| 44 | golden mantled ground squirrel | *callospermophilus lateralis* | 2,621 | 71 | 0 | 0 | **2,692** | 2,154 | — | 🟢 Excellent |
| 45 | baboon genus | *papio* | 2,341 | 129 | 0 | 3 | **2,473** | 1,978 | — | 🟢 Excellent |
| 46 | mongoose family | *herpestidae* | 2,466 | 0 | 0 | 3 | **2,469** | 1,975 | — | 🟢 Excellent |
| 47 | striped skunk | *mephitis mephitis* | 2,367 | 65 | 0 | 1 | **2,433** | 1,946 | — | 🟢 Excellent |
| 48 | pronghorn | *antilocapra americana* | 2,182 | 195 | 0 | 0 | **2,377** | 1,902 | — | 🟢 Excellent |
| 49 | american bison | *bison bison* | 2,127 | 205 | 0 | 3 | **2,335** | 1,868 | — | 🟢 Excellent |
| 50 | bobcat | *lynx rufus* | 2,180 | 126 | 0 | 0 | **2,306** | 1,845 | — | 🟢 Excellent |
| 51 | white-nosed coati | *nasua narica* | 2,165 | 124 | 0 | 0 | **2,289** | 1,831 | — | 🟢 Excellent |
| 52 | domestic horse | *equus caballus* | 1,644 | 179 | 456 | 0 | **2,279** | 1,823 | — | 🟢 Excellent |
| 53 | western gray squirrel | *sciurus griseus* | 2,048 | 194 | 0 | 0 | **2,242** | 1,794 | — | 🟢 Excellent |
| 54 | impala | *aepyceros melampus* | 1,978 | 247 | 0 | 1 | **2,226** | 1,781 | — | 🟢 Excellent |
| 55 | north american porcupine | *erethizon dorsatum* | 2,077 | 126 | 0 | 0 | **2,203** | 1,762 | — | 🟢 Excellent |
| 56 | red deer | *cervus elaphus* | 1,566 | 149 | 472 | 2 | **2,189** | 1,751 | — | 🟢 Excellent |
| 57 | koala | *phascolarctos cinereus* | 1,755 | 33 | 356 | 1 | **2,145** | 1,716 | — | 🟢 Excellent |
| 58 | grey fox | *urocyon cinereoargenteus* | 1,597 | 52 | 455 | 0 | **2,104** | 1,683 | — | 🟢 Excellent |
| 59 | agouti genus | *dasyprocta* | 1,817 | 181 | 0 | 26 | **2,024** | 1,619 | — | 🟢 Excellent |
| 60 | african buffalo | *syncerus caffer* | 1,538 | 217 | 0 | 158 | **1,913** | 1,530 | — | 🟢 Excellent |
| 61 | yellow-bellied marmot | *marmota flaviventris* | 1,756 | 133 | 0 | 0 | **1,889** | 1,511 | — | 🟢 Excellent |
| 62 | llama genus | *lama* | 1,586 | 291 | 0 | 2 | **1,879** | 1,503 | — | 🟢 Excellent |
| 63 | short-beaked echidna | *tachyglossus aculeatus* | 1,660 | 204 | 0 | 0 | **1,864** | 1,491 | 9 | 🟢 Excellent |
| 64 | capybara | *hydrochoerus hydrochaeris* | 1,671 | 184 | 0 | 3 | **1,858** | 1,486 | 14 | 🟢 Excellent |
| 65 | sea otter | *enhydra lutris* | 1,312 | 37 | 459 | 6 | **1,814** | 1,451 | 49 | 🟢 Excellent |
| 66 | elephant seal | *mirounga* | 1,783 | 0 | 0 | 4 | **1,787** | 1,430 | 70 | 🟢 Excellent |
| 67 | arizona black-tailed prairie dog | *cynomys ludovicianus* | 1,605 | 178 | 0 | 3 | **1,786** | 1,429 | 71 | 🟢 Excellent |
| 68 | lycalopex species | *lycalopex* | 1,429 | 294 | 0 | 0 | **1,723** | 1,378 | 122 | 🟢 Excellent |
| 69 | collared peccary | *pecari tajacu* | 1,495 | 213 | 0 | 1 | **1,709** | 1,367 | 133 | 🟢 Excellent |
| 70 | greater kudu | *tragelaphus strepsiceros* | 1,457 | 244 | 0 | 0 | **1,701** | 1,361 | 139 | 🟢 Excellent |
| 71 | alpine ibex | *capra ibex* | 1,043 | 164 | 474 | 16 | **1,697** | 1,358 | 142 | 🟢 Excellent |
| 72 | asian elephant | *elephas maximus* | 1,039 | 151 | 472 | 2 | **1,664** | 1,331 | 169 | 🟢 Excellent |
| 73 | vervet monkey | *chlorocebus pygerythrus* | 1,414 | 237 | 0 | 1 | **1,652** | 1,322 | 178 | 🟢 Excellent |
| 74 | leopard | *panthera pardus* | 969 | 182 | 480 | 6 | **1,637** | 1,310 | 190 | 🟢 Excellent |
| 75 | hippopotamus | *hippopotamus amphibius* | 1,155 | 101 | 364 | 2 | **1,622** | 1,298 | 202 | 🟢 Excellent |
| 76 | common wildebeest | *connochaetes taurinus* | 1,362 | 238 | 0 | 0 | **1,600** | 1,280 | 220 | 🟢 Excellent |
| 77 | weasel species | *mustela* | 1,495 | 89 | 0 | 10 | **1,594** | 1,275 | 225 | 🟢 Excellent |
| 78 | common warthog | *phacochoerus africanus* | 1,549 | 9 | 0 | 1 | **1,559** | 1,247 | 253 | 🟢 Excellent |
| 79 | cebus species | *cebus* | 1,410 | 145 | 0 | 1 | **1,556** | 1,245 | 255 | 🟢 Excellent |
| 80 | callithrix species | *callithrix* | 1,526 | 7 | 0 | 2 | **1,535** | 1,228 | 272 | 🟢 Excellent |
| 81 | swamp wallaby | *wallabia bicolor* | 1,316 | 200 | 0 | 0 | **1,516** | 1,213 | 287 | 🟢 Excellent |
| 82 | common fallow deer | *dama dama* | 1,386 | 125 | 0 | 0 | **1,511** | 1,209 | 291 | 🟢 Excellent |
| 83 | brown bear | *ursus arctos* | 1,013 | 129 | 306 | 9 | **1,457** | 1,166 | 334 | 🟡 Good |
| 84 | alpine marmot | *marmota marmota* | 1,371 | 38 | 0 | 0 | **1,409** | 1,127 | 373 | 🟡 Good |
| 85 | cheetah | *acinonyx jubatus* | 652 | 260 | 484 | 3 | **1,399** | 1,119 | 381 | 🟡 Good |
| 86 | waterbuck | *kobus ellipsiprymnus* | 1,120 | 201 | 0 | 0 | **1,321** | 1,057 | 443 | 🟡 Good |
| 87 | rhinoceros family | *rhinocerotidae* | 1,292 | 0 | 0 | 7 | **1,299** | 1,039 | 461 | 🟡 Good |
| 88 | spotted hyaena | *crocuta crocuta* | 1,047 | 227 | 0 | 1 | **1,275** | 1,020 | 480 | 🟡 Good |
| 89 | reindeer | *rangifer tarandus* | 719 | 39 | 473 | 3 | **1,234** | 987 | 513 | 🟡 Good |
| 90 | martes species | *martes* | 1,101 | 129 | 0 | 1 | **1,231** | 985 | 515 | 🟡 Good |
| 91 | pikas genus | *ochotona* | 1,161 | 63 | 0 | 1 | **1,225** | 980 | 520 | 🟡 Good |
| 92 | brown-throated sloth | *bradypus variegatus* | 1,208 | 5 | 0 | 0 | **1,213** | 970 | 530 | 🟡 Good |
| 93 | domestic donkey | *equus asinus* | 600 | 153 | 450 | 1 | **1,204** | 963 | 537 | 🟡 Good |
| 94 | cercopithecus species | *cercopithecus* | 915 | 276 | 0 | 0 | **1,191** | 953 | 547 | 🟡 Good |
| 95 | ateles species | *ateles* | 1,173 | 12 | 0 | 0 | **1,185** | 948 | 552 | 🟡 Good |
| 96 | rock hyrax | *procavia capensis* | 1,135 | 45 | 0 | 0 | **1,180** | 944 | 556 | 🟡 Good |
| 97 | northern chamois | *rupicapra rupicapra* | 1,019 | 132 | 0 | 0 | **1,151** | 921 | 579 | 🟡 Good |
| 98 | sika deer | *cervus nippon* | 966 | 157 | 0 | 0 | **1,123** | 898 | 602 | 🟡 Good |
| 99 | domestic cattle | *bos taurus* | 475 | 188 | 451 | 0 | **1,114** | 891 | 609 | 🟡 Good |
| 100 | saimiri species | *saimiri* | 1,029 | 73 | 0 | 0 | **1,102** | 882 | 618 | 🟡 Good |
| 101 | muntjac genus | *muntiacus* | 938 | 135 | 0 | 0 | **1,073** | 858 | 642 | 🟡 Good |
| 102 | chital | *axis axis* | 885 | 124 | 0 | 0 | **1,009** | 807 | 693 | 🟡 Good |
| 103 | tiger | *panthera tigris* | 336 | 163 | 490 | 8 | **997** | 798 | 702 | 🟠 Marginal |
| 104 | south american coati | *nasua nasua* | 756 | 230 | 0 | 0 | **986** | 789 | 711 | 🟠 Marginal |
| 105 | saguinus species | *saguinus* | 899 | 9 | 0 | 0 | **908** | 726 | 774 | 🟠 Marginal |
| 106 | mountain zebra | *equus zebra* | 308 | 131 | 457 | 0 | **896** | 717 | 783 | 🟠 Marginal |
| 107 | common eland | *tragelaphus oryx* | 673 | 220 | 0 | 0 | **893** | 714 | 786 | 🟠 Marginal |
| 108 | domestic sheep | *ovis aries* | 330 | 116 | 442 | 3 | **891** | 713 | 787 | 🟠 Marginal |
| 109 | african wild dog | *lycaon pictus* | 359 | 137 | 0 | 367 | **863** | 690 | 810 | 🟠 Marginal |
| 110 | sambar | *rusa unicolor* | 706 | 154 | 0 | 0 | **860** | 688 | 812 | 🟠 Marginal |
| 111 | hartebeest | *alcelaphus buselaphus* | 672 | 178 | 0 | 0 | **850** | 680 | 820 | 🟠 Marginal |
| 112 | giant otter | *pteronura brasiliensis* | 287 | 82 | 459 | 0 | **828** | 662 | 838 | 🟠 Marginal |
| 113 | eurasian otter | *lutra lutra* | 313 | 50 | 459 | 0 | **822** | 658 | 842 | 🟠 Marginal |
| 114 | golden jackal | *canis aureus* | 697 | 122 | 0 | 3 | **822** | 658 | 842 | 🟠 Marginal |
| 115 | steenbok | *raphicerus campestris* | 708 | 112 | 0 | 0 | **820** | 656 | 844 | 🟠 Marginal |
| 116 | grey wolf | *canis lupus* | 261 | 80 | 446 | 8 | **795** | 636 | 864 | 🟠 Marginal |
| 117 | eurasian badger | *meles meles* | 720 | 39 | 0 | 1 | **760** | 608 | 892 | 🟠 Marginal |
| 118 | gemsbok | *oryx gazella* | 515 | 239 | 0 | 0 | **754** | 603 | 897 | 🟠 Marginal |
| 119 | giraffe | *giraffa camelopardalis* | 130 | 140 | 473 | 0 | **743** | 594 | 906 | 🟠 Marginal |
| 120 | nyala | *tragelaphus angasii* | 648 | 86 | 0 | 0 | **734** | 587 | 913 | 🟠 Marginal |
| 121 | mountain goat | *oreamnos americanus* | 635 | 75 | 0 | 0 | **710** | 568 | 932 | 🟠 Marginal |
| 122 | red kangaroo | *osphranter rufus* | 187 | 38 | 441 | 0 | **666** | 533 | 967 | 🟠 Marginal |
| 123 | common wombat | *vombatus ursinus* | 545 | 108 | 0 | 0 | **653** | 522 | 978 | 🟠 Marginal |
| 124 | blesbok | *damaliscus pygargus* | 553 | 91 | 0 | 0 | **644** | 515 | 985 | 🟠 Marginal |
| 125 | springbok | *antidorcas marsupialis* | 506 | 136 | 0 | 0 | **642** | 514 | 986 | 🟠 Marginal |
| 126 | colobus species | *colobus* | 377 | 197 | 0 | 1 | **575** | 460 | 1,040 | 🟠 Marginal |
| 127 | jaguar | *panthera onca* | 364 | 203 | 0 | 6 | **573** | 458 | 1,042 | 🟠 Marginal |
| 128 | common duiker | *sylvicapra grimmia* | 431 | 128 | 0 | 0 | **559** | 447 | 1,053 | 🟠 Marginal |
| 129 | klipspringer | *oreotragus oreotragus* | 502 | 46 | 0 | 0 | **548** | 438 | 1,062 | 🟠 Marginal |
| 130 | puma | *puma concolor* | 427 | 84 | 0 | 3 | **514** | 411 | 1,089 | 🟠 Marginal |
| 131 | eulemur species | *eulemur* | 480 | 28 | 0 | 0 | **508** | 406 | 1,094 | 🟠 Marginal |
| 132 | hoffmann's two-toed sloth | *choloepus hoffmanni* | 480 | 22 | 0 | 0 | **502** | 402 | 1,098 | 🟠 Marginal |
| 133 | old world porcupine family | *hystricidae* | 440 | 62 | 0 | 0 | **502** | 402 | 1,098 | 🟠 Marginal |
| 134 | american badger | *taxidea taxus* | 395 | 97 | 0 | 1 | **493** | 394 | 1,106 | 🔴 Low |
| 135 | grant's gazelle | *nanger granti* | 291 | 201 | 0 | 0 | **492** | 394 | 1,106 | 🔴 Low |
| 136 | domestic water buffalo | *bubalus bubalis* | 288 | 193 | 0 | 0 | **481** | 385 | 1,115 | 🔴 Low |
| 137 | domestic goat | *capra aegagrus hircus* | 0 | 0 | 474 | 4 | **478** | 382 | 1,118 | 🔴 Low |
| 138 | domestic pig | *sus scrofa scrofa* | 0 | 0 | 460 | 0 | **460** | 368 | 1,132 | 🔴 Low |
| 139 | nine-banded armadillo | *dasypus novemcinctus* | 249 | 193 | 0 | 0 | **442** | 354 | 1,146 | 🔴 Low |
| 140 | thomson's gazelle | *eudorcas thomsonii* | 214 | 226 | 0 | 0 | **440** | 352 | 1,148 | 🔴 Low |
| 141 | leaf monkeys genus | *presbytis* | 362 | 64 | 0 | 2 | **428** | 342 | 1,158 | 🔴 Low |
| 142 | spectacled bear | *tremarctos ornatus* | 59 | 45 | 306 | 0 | **410** | 328 | 1,172 | 🔴 Low |
| 143 | sloth bear | *melursus ursinus* | 75 | 21 | 306 | 0 | **402** | 322 | 1,178 | 🔴 Low |
| 144 | leopardus species | *leopardus* | 245 | 149 | 0 | 5 | **399** | 319 | 1,181 | 🔴 Low |
| 145 | giant anteater | *myrmecophaga tridactyla* | 228 | 154 | 0 | 0 | **382** | 306 | 1,194 | 🔴 Low |
| 146 | nilgai | *boselaphus tragocamelus* | 346 | 34 | 0 | 0 | **380** | 304 | 1,196 | 🔴 Low |
| 147 | asiatic black bear | *ursus thibetanus* | 49 | 20 | 306 | 1 | **376** | 301 | 1,199 | 🔴 Low |
| 148 | gorilla species | *gorilla* | 259 | 110 | 0 | 1 | **370** | 296 | 1,204 | 🔴 Low |
| 149 | lowland tapir | *tapirus terrestris* | 226 | 130 | 0 | 0 | **356** | 285 | 1,215 | 🔴 Low |
| 150 | sun bear | *helarctos malayanus* | 29 | 16 | 306 | 0 | **351** | 281 | 1,219 | 🔴 Low |
| 151 | glaucomys species | *glaucomys* | 305 | 42 | 0 | 0 | **347** | 278 | 1,222 | 🔴 Low |
| 152 | bushbuck | *tragelaphus scriptus* | 63 | 283 | 0 | 0 | **346** | 277 | 1,223 | 🔴 Low |
| 153 | reedbuck genus | *redunca* | 292 | 52 | 0 | 0 | **344** | 275 | 1,225 | 🔴 Low |
| 154 | red panda | *ailurus fulgens* | 15 | 11 | 317 | 1 | **344** | 275 | 1,225 | 🔴 Low |
| 155 | tayra | *eira barbara* | 197 | 144 | 0 | 0 | **341** | 273 | 1,227 | 🔴 Low |
| 156 | japanese macaque | *macaca fuscata* | 296 | 45 | 0 | 0 | **341** | 273 | 1,227 | 🔴 Low |
| 157 | bornean orangutan | *pongo pygmaeus* | 225 | 102 | 0 | 1 | **328** | 262 | 1,238 | 🔴 Low |
| 158 | blackbuck | *antilope cervicapra* | 260 | 40 | 0 | 0 | **300** | 240 | 1,260 | 🔴 Low |
| 159 | sable antelope | *hippotragus niger* | 242 | 44 | 0 | 0 | **286** | 229 | 1,271 | 🔴 Low |
| 160 | dromedary camel | *camelus dromedarius* | 134 | 144 | 0 | 1 | **279** | 223 | 1,277 | 🔴 Low |
| 161 | genet genus | *genetta* | 252 | 22 | 0 | 2 | **276** | 221 | 1,279 | 🔴 Low |
| 162 | quokka | *setonix brachyurus* | 241 | 17 | 0 | 0 | **258** | 206 | 1,294 | 🔴 Low |
| 163 | ringtail | *bassariscus astutus* | 226 | 29 | 0 | 0 | **255** | 204 | 1,296 | 🔴 Low |
| 164 | ocelot | *leopardus pardalis* | 92 | 158 | 0 | 0 | **250** | 200 | 1,300 | 🔴 Low |
| 165 | baird's tapir | *tapirus bairdii* | 183 | 66 | 0 | 0 | **249** | 199 | 1,301 | 🔴 Low |
| 166 | ring-tailed lemur | *lemur catta* | 208 | 32 | 0 | 0 | **240** | 192 | 1,308 | 🔴 Low |
| 167 | wild cat | *felis silvestris* | 85 | 122 | 0 | 2 | **209** | 167 | 1,333 | 🔴 Low |
| 168 | chimpanzee | *pan troglodytes* | 143 | 62 | 0 | 1 | **206** | 165 | 1,335 | 🔴 Low |
| 169 | kob | *kobus kob* | 154 | 49 | 0 | 0 | **203** | 162 | 1,338 | 🔴 Low |
| 170 | bat-eared fox | *otocyon megalotis* | 166 | 37 | 0 | 0 | **203** | 162 | 1,338 | 🔴 Low |
| 171 | raccoon dog | *nyctereutes procyonoides* | 160 | 34 | 0 | 2 | **196** | 157 | 1,343 | 🔴 Low |
| 172 | serval | *leptailurus serval* | 115 | 66 | 0 | 4 | **185** | 148 | 1,352 | 🔴 Low |
| 173 | grevy's zebra | *equus grevyi* | 102 | 77 | 0 | 0 | **179** | 143 | 1,357 | 🔴 Low |
| 174 | caracal | *caracal caracal* | 145 | 30 | 0 | 2 | **177** | 142 | 1,358 | 🔴 Low |
| 175 | kinkajou | *potos flavus* | 164 | 10 | 0 | 1 | **175** | 140 | 1,360 | 🔴 Low |
| 176 | fisher | *pekania pennanti* | 99 | 67 | 0 | 0 | **166** | 133 | 1,367 | 🔴 Low |
| 177 | roan antelope | *hippotragus equinus* | 156 | 7 | 0 | 0 | **163** | 130 | 1,370 | 🔴 Low |
| 178 | dhole | *cuon alpinus* | 109 | 48 | 0 | 0 | **157** | 126 | 1,374 | 🔴 Low |
| 179 | kirk's dik-dik | *madoqua kirkii* | 66 | 89 | 0 | 0 | **155** | 124 | 1,376 | 🔴 Low |
| 180 | meerkat | *suricata suricatta* | 137 | 11 | 0 | 0 | **148** | 118 | 1,382 | 🔴 Low |
| 181 | leopard cat | *prionailurus bengalensis* | 95 | 49 | 0 | 2 | **146** | 117 | 1,383 | 🔴 Low |
| 182 | patas monkey | *erythrocebus patas* | 92 | 54 | 0 | 0 | **146** | 117 | 1,383 | 🔴 Low |
| 183 | black wildebeest | *connochaetes gnou* | 94 | 51 | 0 | 0 | **145** | 116 | 1,384 | 🔴 Low |
| 184 | red brocket | *mazama americana* | 36 | 109 | 0 | 0 | **145** | 116 | 1,384 | 🔴 Low |
| 185 | canada lynx | *lynx canadensis* | 98 | 45 | 0 | 1 | **144** | 115 | 1,385 | 🔴 Low |
| 186 | european bison | *bison bonasus* | 60 | 70 | 0 | 6 | **136** | 109 | 1,391 | 🔴 Low |
| 187 | eurasian lynx | *lynx lynx* | 45 | 85 | 0 | 0 | **130** | 104 | 1,396 | 🔴 Low |
| 188 | gerenuk | *litocranius walleri* | 93 | 34 | 0 | 1 | **128** | 102 | 1,398 | 🔴 Low |
| 189 | walrus | *odobenus rosmarus* | 113 | 0 | 0 | 6 | **119** | 95 | 1,405 | 🔴 Low |
| 190 | water deer | *hydropotes inermis* | 76 | 33 | 0 | 2 | **111** | 89 | 1,411 | 🔴 Low |
| 191 | honey badger | *mellivora capensis* | 85 | 24 | 0 | 0 | **109** | 87 | 1,413 | 🔴 Low |
| 192 | maned wolf | *chrysocyon brachyurus* | 65 | 43 | 0 | 0 | **108** | 86 | 1,414 | 🔴 Low |
| 193 | striped hyaena | *hyaena hyaena* | 49 | 31 | 0 | 1 | **81** | 65 | 1,435 | ⛔ Critical |
| 194 | brown hyaena | *parahyaena brunnea* | 61 | 13 | 0 | 0 | **74** | 59 | 1,441 | ⛔ Critical |
| 195 | callicebus genus | *callicebus* | 71 | 2 | 0 | 0 | **73** | 58 | 1,442 | ⛔ Critical |
| 196 | asiatic wild ass | *equus hemionus* | 57 | 5 | 0 | 0 | **62** | 50 | 1,450 | ⛔ Critical |
| 197 | black-backed jackal | *canis mesomelas* | 0 | 61 | 0 | 0 | **61** | 49 | 1,451 | ⛔ Critical |
| 198 | spilogale species | *spilogale* | 53 | 7 | 0 | 0 | **60** | 48 | 1,452 | ⛔ Critical |
| 199 | yak | *bos grunniens* | 58 | 0 | 0 | 1 | **59** | 47 | 1,453 | ⛔ Critical |
| 200 | african civet | *civettictis civetta* | 36 | 17 | 0 | 2 | **55** | 44 | 1,456 | ⛔ Critical |
| 201 | cephalophus species | *cephalophus* | 5 | 47 | 0 | 0 | **52** | 42 | 1,458 | ⛔ Critical |
| 202 | aardvark | *orycteropus afer* | 33 | 11 | 0 | 8 | **52** | 42 | 1,458 | ⛔ Critical |
| 203 | fossa | *cryptoprocta ferox* | 45 | 7 | 0 | 0 | **52** | 42 | 1,458 | ⛔ Critical |
| 204 | american mink | *neovison vison* | 0 | 45 | 0 | 1 | **46** | 37 | 1,463 | ⛔ Critical |
| 205 | pangolin family | *manidae* | 40 | 0 | 0 | 4 | **44** | 35 | 1,465 | ⛔ Critical |
| 206 | aardwolf | *proteles cristata* | 0 | 12 | 0 | 24 | **36** | 29 | 1,471 | ⛔ Critical |
| 207 | wolverine | *gulo gulo* | 15 | 17 | 0 | 0 | **32** | 26 | 1,474 | ⛔ Critical |
| 208 | saiga | *saiga tatarica* | 28 | 4 | 0 | 0 | **32** | 26 | 1,474 | ⛔ Critical |
| 209 | aye-aye | *daubentonia madagascariensis* | 19 | 4 | 0 | 0 | **23** | 18 | 1,482 | ⛔ Critical |
| 210 | malay tapir | *tapirus indicus* | 14 | 7 | 0 | 0 | **21** | 17 | 1,483 | ⛔ Critical |
| 211 | giant armadillo | *priodontes maximus* | 7 | 13 | 0 | 0 | **20** | 16 | 1,484 | ⛔ Critical |
| 212 | mangabeys genus | *cercocebus* | 10 | 6 | 0 | 0 | **16** | 13 | 1,487 | ⛔ Critical |
| 213 | red-necked wallaby | *macropus rufogriseus* | 0 | 15 | 0 | 0 | **15** | 12 | 1,488 | ⛔ Critical |
| 214 | hog badger genus | *arctonyx* | 4 | 10 | 0 | 1 | **15** | 12 | 1,488 | ⛔ Critical |
| 215 | binturong | *arctictis binturong* | 10 | 3 | 0 | 1 | **14** | 11 | 1,489 | ⛔ Critical |
| 216 | giant panda | *ailuropoda melanoleuca* | 3 | 7 | 0 | 2 | **12** | 10 | 1,490 | ⛔ Critical |
| 217 | clouded leopard | *neofelis nebulosa* | 6 | 4 | 0 | 1 | **11** | 9 | 1,491 | ⛔ Critical |
| 218 | snow leopard | *panthera uncia* | 4 | 4 | 0 | 3 | **11** | 9 | 1,491 | ⛔ Critical |
| 219 | red river hog | *potamochoerus porcus* | 7 | 2 | 0 | 0 | **9** | 7 | 1,493 | ⛔ Critical |
| 220 | drill | *mandrillus leucophaeus* | 6 | 2 | 0 | 0 | **8** | 6 | 1,494 | ⛔ Critical |
| 221 | human | *homo sapiens* | 0 | 2 | 0 | 0 | **2** | 2 | 1,498 | ⛔ Critical |
| 222 | mouflon | *ovis orientalis* | 0 | 1 | 0 | 0 | **1** | 1 | 1,499 | ⛔ Critical |
| 223 | bongo | *tragelaphus eurycerus* | 0 | 0 | 0 | 0 | **0** | 0 | 1,500 | ⛔ Critical |
| 224 | dingo | *canis lupus dingo* | 0 | 0 | 0 | 0 | **0** | 0 | 1,500 | ⛔ Critical |
| 225 | pinniped clade | *pinniped* | 0 | 0 | 0 | 0 | **0** | 0 | 1,500 | ⛔ Critical |

## Critical Classes — Urgent Supplementation Needed

**33 classes** with fewer than 100 passed images.

| Class | Scientific name | iNat | GBIF | OI | Wiki | Total | Gap |
|-------|-----------------|-----:|-----:|---:|-----:|------:|----:|
| dingo | *canis lupus dingo* | 0 | 0 | 0 | 0 | 0 | 1,500 |
| bongo | *tragelaphus eurycerus* | 0 | 0 | 0 | 0 | 0 | 1,500 |
| pinniped clade | *pinniped* | 0 | 0 | 0 | 0 | 0 | 1,500 |
| mouflon | *ovis orientalis* | 0 | 1 | 0 | 0 | 1 | 1,499 |
| human | *homo sapiens* | 0 | 2 | 0 | 0 | 2 | 1,498 |
| drill | *mandrillus leucophaeus* | 6 | 2 | 0 | 0 | 8 | 1,494 |
| red river hog | *potamochoerus porcus* | 7 | 2 | 0 | 0 | 9 | 1,493 |
| clouded leopard | *neofelis nebulosa* | 6 | 4 | 0 | 1 | 11 | 1,491 |
| snow leopard | *panthera uncia* | 4 | 4 | 0 | 3 | 11 | 1,491 |
| giant panda | *ailuropoda melanoleuca* | 3 | 7 | 0 | 2 | 12 | 1,490 |
| binturong | *arctictis binturong* | 10 | 3 | 0 | 1 | 14 | 1,489 |
| red-necked wallaby | *macropus rufogriseus* | 0 | 15 | 0 | 0 | 15 | 1,488 |
| hog badger genus | *arctonyx* | 4 | 10 | 0 | 1 | 15 | 1,488 |
| mangabeys genus | *cercocebus* | 10 | 6 | 0 | 0 | 16 | 1,487 |
| giant armadillo | *priodontes maximus* | 7 | 13 | 0 | 0 | 20 | 1,484 |
| malay tapir | *tapirus indicus* | 14 | 7 | 0 | 0 | 21 | 1,483 |
| aye-aye | *daubentonia madagascariensis* | 19 | 4 | 0 | 0 | 23 | 1,482 |
| saiga | *saiga tatarica* | 28 | 4 | 0 | 0 | 32 | 1,474 |
| wolverine | *gulo gulo* | 15 | 17 | 0 | 0 | 32 | 1,474 |
| aardwolf | *proteles cristata* | 0 | 12 | 0 | 24 | 36 | 1,471 |
| pangolin family | *manidae* | 40 | 0 | 0 | 4 | 44 | 1,465 |
| american mink | *neovison vison* | 0 | 45 | 0 | 1 | 46 | 1,463 |
| fossa | *cryptoprocta ferox* | 45 | 7 | 0 | 0 | 52 | 1,458 |
| cephalophus species | *cephalophus* | 5 | 47 | 0 | 0 | 52 | 1,458 |
| aardvark | *orycteropus afer* | 33 | 11 | 0 | 8 | 52 | 1,458 |
| african civet | *civettictis civetta* | 36 | 17 | 0 | 2 | 55 | 1,456 |
| yak | *bos grunniens* | 58 | 0 | 0 | 1 | 59 | 1,453 |
| spilogale species | *spilogale* | 53 | 7 | 0 | 0 | 60 | 1,452 |
| black-backed jackal | *canis mesomelas* | 0 | 61 | 0 | 0 | 61 | 1,451 |
| asiatic wild ass | *equus hemionus* | 57 | 5 | 0 | 0 | 62 | 1,450 |
| callicebus genus | *callicebus* | 71 | 2 | 0 | 0 | 73 | 1,442 |
| brown hyaena | *parahyaena brunnea* | 61 | 13 | 0 | 0 | 74 | 1,441 |
| striped hyaena | *hyaena hyaena* | 49 | 31 | 0 | 1 | 81 | 1,435 |

## Low Classes — Supplementation Recommended

**59 classes** with 100–499 passed images.

| Class | Scientific name | iNat | GBIF | OI | Wiki | Total | Gap |
|-------|-----------------|-----:|-----:|---:|-----:|------:|----:|
| maned wolf | *chrysocyon brachyurus* | 65 | 43 | 0 | 0 | 108 | 1,414 |
| honey badger | *mellivora capensis* | 85 | 24 | 0 | 0 | 109 | 1,413 |
| water deer | *hydropotes inermis* | 76 | 33 | 0 | 2 | 111 | 1,411 |
| walrus | *odobenus rosmarus* | 113 | 0 | 0 | 6 | 119 | 1,405 |
| gerenuk | *litocranius walleri* | 93 | 34 | 0 | 1 | 128 | 1,398 |
| eurasian lynx | *lynx lynx* | 45 | 85 | 0 | 0 | 130 | 1,396 |
| european bison | *bison bonasus* | 60 | 70 | 0 | 6 | 136 | 1,391 |
| canada lynx | *lynx canadensis* | 98 | 45 | 0 | 1 | 144 | 1,385 |
| red brocket | *mazama americana* | 36 | 109 | 0 | 0 | 145 | 1,384 |
| black wildebeest | *connochaetes gnou* | 94 | 51 | 0 | 0 | 145 | 1,384 |
| leopard cat | *prionailurus bengalensis* | 95 | 49 | 0 | 2 | 146 | 1,383 |
| patas monkey | *erythrocebus patas* | 92 | 54 | 0 | 0 | 146 | 1,383 |
| meerkat | *suricata suricatta* | 137 | 11 | 0 | 0 | 148 | 1,382 |
| kirk's dik-dik | *madoqua kirkii* | 66 | 89 | 0 | 0 | 155 | 1,376 |
| dhole | *cuon alpinus* | 109 | 48 | 0 | 0 | 157 | 1,374 |
| roan antelope | *hippotragus equinus* | 156 | 7 | 0 | 0 | 163 | 1,370 |
| fisher | *pekania pennanti* | 99 | 67 | 0 | 0 | 166 | 1,367 |
| kinkajou | *potos flavus* | 164 | 10 | 0 | 1 | 175 | 1,360 |
| caracal | *caracal caracal* | 145 | 30 | 0 | 2 | 177 | 1,358 |
| grevy's zebra | *equus grevyi* | 102 | 77 | 0 | 0 | 179 | 1,357 |
| serval | *leptailurus serval* | 115 | 66 | 0 | 4 | 185 | 1,352 |
| raccoon dog | *nyctereutes procyonoides* | 160 | 34 | 0 | 2 | 196 | 1,343 |
| bat-eared fox | *otocyon megalotis* | 166 | 37 | 0 | 0 | 203 | 1,338 |
| kob | *kobus kob* | 154 | 49 | 0 | 0 | 203 | 1,338 |
| chimpanzee | *pan troglodytes* | 143 | 62 | 0 | 1 | 206 | 1,335 |
| wild cat | *felis silvestris* | 85 | 122 | 0 | 2 | 209 | 1,333 |
| ring-tailed lemur | *lemur catta* | 208 | 32 | 0 | 0 | 240 | 1,308 |
| baird's tapir | *tapirus bairdii* | 183 | 66 | 0 | 0 | 249 | 1,301 |
| ocelot | *leopardus pardalis* | 92 | 158 | 0 | 0 | 250 | 1,300 |
| ringtail | *bassariscus astutus* | 226 | 29 | 0 | 0 | 255 | 1,296 |
| quokka | *setonix brachyurus* | 241 | 17 | 0 | 0 | 258 | 1,294 |
| genet genus | *genetta* | 252 | 22 | 0 | 2 | 276 | 1,279 |
| dromedary camel | *camelus dromedarius* | 134 | 144 | 0 | 1 | 279 | 1,277 |
| sable antelope | *hippotragus niger* | 242 | 44 | 0 | 0 | 286 | 1,271 |
| blackbuck | *antilope cervicapra* | 260 | 40 | 0 | 0 | 300 | 1,260 |
| bornean orangutan | *pongo pygmaeus* | 225 | 102 | 0 | 1 | 328 | 1,238 |
| tayra | *eira barbara* | 197 | 144 | 0 | 0 | 341 | 1,227 |
| japanese macaque | *macaca fuscata* | 296 | 45 | 0 | 0 | 341 | 1,227 |
| red panda | *ailurus fulgens* | 15 | 11 | 317 | 1 | 344 | 1,225 |
| reedbuck genus | *redunca* | 292 | 52 | 0 | 0 | 344 | 1,225 |
| bushbuck | *tragelaphus scriptus* | 63 | 283 | 0 | 0 | 346 | 1,223 |
| glaucomys species | *glaucomys* | 305 | 42 | 0 | 0 | 347 | 1,222 |
| sun bear | *helarctos malayanus* | 29 | 16 | 306 | 0 | 351 | 1,219 |
| lowland tapir | *tapirus terrestris* | 226 | 130 | 0 | 0 | 356 | 1,215 |
| gorilla species | *gorilla* | 259 | 110 | 0 | 1 | 370 | 1,204 |
| asiatic black bear | *ursus thibetanus* | 49 | 20 | 306 | 1 | 376 | 1,199 |
| nilgai | *boselaphus tragocamelus* | 346 | 34 | 0 | 0 | 380 | 1,196 |
| giant anteater | *myrmecophaga tridactyla* | 228 | 154 | 0 | 0 | 382 | 1,194 |
| leopardus species | *leopardus* | 245 | 149 | 0 | 5 | 399 | 1,181 |
| sloth bear | *melursus ursinus* | 75 | 21 | 306 | 0 | 402 | 1,178 |
| spectacled bear | *tremarctos ornatus* | 59 | 45 | 306 | 0 | 410 | 1,172 |
| leaf monkeys genus | *presbytis* | 362 | 64 | 0 | 2 | 428 | 1,158 |
| thomson's gazelle | *eudorcas thomsonii* | 214 | 226 | 0 | 0 | 440 | 1,148 |
| nine-banded armadillo | *dasypus novemcinctus* | 249 | 193 | 0 | 0 | 442 | 1,146 |
| domestic pig | *sus scrofa scrofa* | 0 | 0 | 460 | 0 | 460 | 1,132 |
| domestic goat | *capra aegagrus hircus* | 0 | 0 | 474 | 4 | 478 | 1,118 |
| domestic water buffalo | *bubalus bubalis* | 288 | 193 | 0 | 0 | 481 | 1,115 |
| grant's gazelle | *nanger granti* | 291 | 201 | 0 | 0 | 492 | 1,106 |
| american badger | *taxidea taxus* | 395 | 97 | 0 | 1 | 493 | 1,106 |
