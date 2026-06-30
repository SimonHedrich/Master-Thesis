# Lookalike Candidates

SpeciesNet confusion pairs mined from manually-rejected multi-animal images.
Each pair shows what SpeciesNet predicted on a bounding box that belongs to
the expected class (or its environment).  High counts indicate a systematic
confusion that may warrant a curated look-alike group.

The `review_decision` column in the companion CSV is for manual annotation:
`lookalike` | `different` | `skip`.

## llama_genus  (434 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 227 | order |
| pronghorn | antilocapra americana | 120 | order |
| domestic cattle | bos taurus | 69 | order |
| mule deer | odocoileus hemionus | 61 | order |
| red deer | cervus elaphus | 48 | order |
| thomson's gazelle | eudorcas thomsonii | 40 | order |
| domestic sheep | ovis aries | 25 | order |
| springbok | antidorcas marsupialis | 19 | order |
| domestic dog | canis familiaris | 12 | class |
| guenther's dik-dik | madoqua guentheri | 10 | order |
| coyote | canis latrans | 10 | class |
| sika deer | cervus nippon | 9 | order |
| impala | aepyceros melampus | 8 | order |
| capra species | capra | 6 | order |
| blue sheep | pseudois nayaur | 5 | order |
| bushbuck | tragelaphus scriptus | 5 | order |
| red fox | vulpes vulpes | 5 | class |
| domestic horse | equus caballus | 5 | class |
| european roe deer | capreolus capreolus | 5 | order |
| common fallow deer | dama dama | 5 | order |
| pampas deer | ozotoceros bezoarticus | 4 | order |
| domestic donkey | equus asinus | 3 | class |
| wild goat | capra aegagrus | 3 | order |
| mouflon | ovis orientalis | 3 | order |
| puku | kobus vardonii | 3 | order |
| domestic cat | felis catus | 2 | class |
| bobcat | lynx rufus | 2 | class |
| hylobatidae family |  | 2 | class |
| american bison | bison bison | 1 | order |
| ovis species | ovis | 1 | order |
| artiodactyla order |  | 1 | order |
| elk | cervus canadensis | 1 | order |
| plains zebra | equus quagga | 1 | class |
| common eland | tragelaphus oryx | 1 | order |
| odocoileus species | odocoileus | 1 | order |
| black-backed jackal | canis mesomelas | 1 | class |
| przewalski's horse | equus ferus | 1 | class |
| vervet monkey | chlorocebus pygerythrus | 1 | class |
| grant's gazelle | nanger granti | 1 | order |
| collared peccary | pecari tajacu | 1 | order |
| common warthog | phacochoerus africanus | 1 | order |
| eastern fox squirrel | sciurus niger | 1 | class |
| sitatunga | tragelaphus spekii | 1 | order |
| lion | panthera leo | 1 | class |
| spotted hyaena | crocuta crocuta | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | order |

## domestic_horse  (300 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 252 | class |
| red deer | cervus elaphus | 31 | class |
| white-tailed deer | odocoileus virginianus | 24 | class |
| domestic goat | capra aegagrus hircus | 14 | class |
| common wildebeest | connochaetes taurinus | 9 | class |
| elk | cervus canadensis | 7 | class |
| african elephant | loxodonta africana | 3 | class |
| domestic dog | canis familiaris | 3 | class |
| mule deer | odocoileus hemionus | 3 | class |
| human | homo sapiens | 3 | class |
| dromedary camel | camelus dromedarius | 2 | class |
| american bison | bison bison | 2 | class |
| puku | kobus vardonii | 2 | class |
| impala | aepyceros melampus | 2 | class |
| wild goat | capra aegagrus | 2 | class |
| pronghorn | antilocapra americana | 2 | class |
| thomson's gazelle | eudorcas thomsonii | 2 | class |
| sika deer | cervus nippon | 1 | class |
| moose | alces alces | 1 | class |
| hartebeest | alcelaphus buselaphus | 1 | class |
| capra species | capra | 1 | class |
| lion | panthera leo | 1 | class |

## bighorn_sheep  (290 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 102 | order |
| mule deer | odocoileus hemionus | 95 | order |
| red deer | cervus elaphus | 90 | order |
| pronghorn | antilocapra americana | 34 | order |
| elk | cervus canadensis | 18 | order |
| domestic dog | canis familiaris | 8 | class |
| domestic horse | equus caballus | 6 | class |
| coyote | canis latrans | 4 | class |
| domestic cat | felis catus | 3 | class |
| sika deer | cervus nippon | 3 | order |
| yellow-bellied marmot | marmota flaviventris | 2 | class |
| common warthog | phacochoerus africanus | 2 | order |
| human | homo sapiens | 1 | class |
| domestic donkey | equus asinus | 1 | class |
| reindeer | rangifer tarandus | 1 | order |
| common fallow deer | dama dama | 1 | order |
| wild boar | sus scrofa | 1 | order |
| black-tailed jackrabbit | lepus californicus | 1 | class |
| spotted hyaena | crocuta crocuta | 1 | class |
| eastern gray squirrel | sciurus carolinensis | 1 | class |
| moose | alces alces | 1 | order |
| dromedary camel | camelus dromedarius | 1 | order |

## rhinoceros_family  (232 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 120 | no_match |
| african elephant | loxodonta africana | 56 | no_match |
| common warthog | phacochoerus africanus | 33 | no_match |
| domestic horse | equus caballus | 11 | no_match |
| african buffalo | syncerus caffer | 11 | no_match |
| human | homo sapiens | 8 | no_match |
| white-tailed deer | odocoileus virginianus | 5 | no_match |
| dromedary camel | camelus dromedarius | 5 | no_match |
| hippopotamus | hippopotamus amphibius | 5 | no_match |
| common wildebeest | connochaetes taurinus | 4 | no_match |
| lowland tapir | tapirus terrestris | 4 | no_match |
| giraffe | giraffa camelopardalis | 4 | no_match |
| plains zebra | equus quagga | 3 | no_match |
| red deer | cervus elaphus | 1 | no_match |
| domestic sheep | ovis aries | 1 | no_match |
| malay tapir | tapirus indicus | 1 | no_match |
| domestic dog | canis familiaris | 1 | no_match |
| wild goat | capra aegagrus | 1 | no_match |
| mule deer | odocoileus hemionus | 1 | no_match |

## domestic_donkey  (220 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 154 | class |
| white-tailed deer | odocoileus virginianus | 22 | class |
| wild goat | capra aegagrus | 14 | class |
| domestic goat | capra aegagrus hircus | 11 | class |
| domestic sheep | ovis aries | 10 | class |
| red deer | cervus elaphus | 7 | class |
| mule deer | odocoileus hemionus | 7 | class |
| blue sheep | pseudois nayaur | 5 | class |
| pronghorn | antilocapra americana | 5 | class |
| giraffe | giraffa camelopardalis | 4 | class |
| human | homo sapiens | 4 | class |
| common wildebeest | connochaetes taurinus | 3 | class |
| capra species | capra | 3 | class |
| guenther's dik-dik | madoqua guentheri | 3 | class |
| domestic dog | canis familiaris | 2 | class |
| hartebeest | alcelaphus buselaphus | 2 | class |
| sika deer | cervus nippon | 1 | class |
| northern chamois | rupicapra rupicapra | 1 | class |
| thomson's gazelle | eudorcas thomsonii | 1 | class |
| greater kudu | tragelaphus strepsiceros | 1 | class |
| wild boar | sus scrofa | 1 | class |
| coyote | canis latrans | 1 | class |
| impala | aepyceros melampus | 1 | class |
| common fallow deer | dama dama | 1 | class |
| american black bear | ursus americanus | 1 | class |
| common warthog | phacochoerus africanus | 1 | class |
| canis species | canis | 1 | class |
| lion | panthera leo | 1 | class |
| gemsbok | oryx gazella | 1 | class |
| elk | cervus canadensis | 1 | class |

## eastern_grey_kangaroo  (199 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 153 | class |
| domestic cattle | bos taurus | 25 | class |
| mule deer | odocoileus hemionus | 11 | class |
| sika deer | cervus nippon | 6 | class |
| african elephant | loxodonta africana | 5 | class |
| domestic dog | canis familiaris | 4 | class |
| coyote | canis latrans | 4 | class |
| red deer | cervus elaphus | 4 | class |
| vervet monkey | chlorocebus pygerythrus | 3 | class |
| thomson's gazelle | eudorcas thomsonii | 3 | class |
| domestic sheep | ovis aries | 3 | class |
| eastern gray squirrel | sciurus carolinensis | 3 | class |
| human | homo sapiens | 2 | class |
| black-tailed jackrabbit | lepus californicus | 2 | class |
| pronghorn | antilocapra americana | 2 | class |
| kinda baboon | papio kindae | 1 | class |
| grey wolf | canis lupus | 1 | class |
| moose | alces alces | 1 | class |
| lion | panthera leo | 1 | class |
| guenther's dik-dik | madoqua guentheri | 1 | class |
| common wildebeest | connochaetes taurinus | 1 | class |
| baboon species | papio | 1 | class |
| american bison | bison bison | 1 | class |
| dromedary camel | camelus dromedarius | 1 | class |
| common duiker | sylvicapra grimmia | 1 | class |
| pampas deer | ozotoceros bezoarticus | 1 | class |
| eastern fox squirrel | sciurus niger | 1 | class |
| blue sheep | pseudois nayaur | 1 | class |
| puma | puma concolor | 1 | class |
| primate |  | 1 | class |
| olive baboon | papio anubis | 1 | class |
| common fallow deer | dama dama | 1 | class |

## capybara  (189 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 50 | class |
| domestic cattle | bos taurus | 38 | class |
| nutria | myocastor coypus | 30 | order |
| wild boar | sus scrofa | 25 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 23 | order |
| red deer | cervus elaphus | 11 | class |
| american black bear | ursus americanus | 9 | class |
| domestic cat | felis catus | 7 | class |
| eastern gray squirrel | sciurus carolinensis | 6 | order |
| brown bear | ursus arctos | 5 | class |
| domestic horse | equus caballus | 5 | class |
| woodchuck | marmota monax | 5 | order |
| northern raccoon | procyon lotor | 4 | class |
| sika deer | cervus nippon | 4 | class |
| domestic goat | capra aegagrus hircus | 4 | class |
| elk | cervus canadensis | 3 | class |
| mule deer | odocoileus hemionus | 3 | class |
| lowland tapir | tapirus terrestris | 3 | class |
| central american agouti | dasyprocta punctata | 3 | order |
| north american river otter | lontra canadensis | 3 | class |
| white-lipped peccary | tayassu pecari | 3 | class |
| eurasian red squirrel | sciurus vulgaris | 3 | order |
| common warthog | phacochoerus africanus | 2 | class |
| california ground squirrel | otospermophilus beecheyi | 2 | order |
| eastern fox squirrel | sciurus niger | 2 | order |
| american bison | bison bison | 1 | class |
| black-fronted duiker | cephalophus nigrifrons | 1 | class |
| yellow-bellied marmot | marmota flaviventris | 1 | order |
| takin | budorcas taxicolor | 1 | class |
| yellow-throated marten | martes flavigula | 1 | class |
| northern chamois | rupicapra rupicapra | 1 | class |
| fisher | pekania pennanti | 1 | class |
| mongolian marmot | marmota sibirica | 1 | order |
| rodent |  | 1 | order |
| giant otter | pteronura brasiliensis | 1 | class |
| vervet monkey | chlorocebus pygerythrus | 1 | class |
| kinda baboon | papio kindae | 1 | class |
| american mink | neovison vison | 1 | class |
| bushpig | potamochoerus larvatus | 1 | class |

## pronghorn  (155 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 90 | order |
| thomson's gazelle | eudorcas thomsonii | 24 | order |
| impala | aepyceros melampus | 15 | order |
| mule deer | odocoileus hemionus | 13 | order |
| red deer | cervus elaphus | 12 | order |
| springbok | antidorcas marsupialis | 7 | order |
| guenther's dik-dik | madoqua guentheri | 6 | order |
| grant's gazelle | nanger granti | 3 | order |
| domestic cattle | bos taurus | 3 | order |
| domestic dog | canis familiaris | 2 | class |
| european roe deer | capreolus capreolus | 2 | order |
| puku | kobus vardonii | 2 | order |
| dik-dik species | madoqua | 1 | order |
| odocoileus species | odocoileus | 1 | order |
| black-tailed jackrabbit | lepus californicus | 1 | class |
| human | homo sapiens | 1 | class |
| elk | cervus canadensis | 1 | order |
| domestic horse | equus caballus | 1 | class |
| plains zebra | equus quagga | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | order |

## impala  (150 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 167 | order |
| red deer | cervus elaphus | 5 | order |
| plains zebra | equus quagga | 4 | class |
| mule deer | odocoileus hemionus | 4 | order |
| human | homo sapiens | 3 | class |
| pronghorn | antilocapra americana | 3 | order |
| odocoileus species | odocoileus | 1 | order |
| european roe deer | capreolus capreolus | 1 | order |
| vervet monkey | chlorocebus pygerythrus | 1 | class |
| kinda baboon | papio kindae | 1 | class |
| pampas deer | ozotoceros bezoarticus | 1 | order |

## domestic_sheep  (142 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 59 | order |
| mule deer | odocoileus hemionus | 26 | order |
| red deer | cervus elaphus | 19 | order |
| domestic dog | canis familiaris | 18 | class |
| domestic horse | equus caballus | 15 | class |
| wild boar | sus scrofa | 10 | order |
| domestic cat | felis catus | 9 | class |
| coyote | canis latrans | 7 | class |
| eastern gray squirrel | sciurus carolinensis | 5 | class |
| lion | panthera leo | 5 | class |
| sika deer | cervus nippon | 3 | order |
| common warthog | phacochoerus africanus | 2 | order |
| european roe deer | capreolus capreolus | 2 | order |
| elk | cervus canadensis | 2 | order |
| american black bear | ursus americanus | 2 | class |
| hylobatidae family |  | 2 | class |
| odocoileus species | odocoileus | 1 | order |
| eastern cottontail | sylvilagus floridanus | 1 | class |
| african elephant | loxodonta africana | 1 | class |
| human | homo sapiens | 1 | class |
| brush-tailed rock wallaby | petrogale penicillata | 1 | class |
| canis species | canis | 1 | class |
| vicugna species | vicugna | 1 | order |
| red-necked wallaby | macropus rufogriseus | 1 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | class |
| red fox | vulpes vulpes | 1 | class |
| northern raccoon | procyon lotor | 1 | class |

## common_wildebeest  (124 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic horse | equus caballus | 42 | class |
| white-tailed deer | odocoileus virginianus | 34 | order |
| plains zebra | equus quagga | 23 | class |
| red deer | cervus elaphus | 14 | order |
| mule deer | odocoileus hemionus | 10 | order |
| elk | cervus canadensis | 9 | order |
| common warthog | phacochoerus africanus | 5 | order |
| dromedary camel | camelus dromedarius | 4 | order |
| moose | alces alces | 2 | order |
| african elephant | loxodonta africana | 2 | class |
| giraffe | giraffa camelopardalis | 2 | order |
| pronghorn | antilocapra americana | 1 | order |
| przewalski's horse | equus ferus | 1 | class |
| human | homo sapiens | 1 | class |
| domestic donkey | equus asinus | 1 | class |
| mountain zebra | equus zebra | 1 | class |

## greater_kudu  (112 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 88 | order |
| mule deer | odocoileus hemionus | 18 | order |
| red deer | cervus elaphus | 11 | order |
| plains zebra | equus quagga | 8 | class |
| lion | panthera leo | 1 | class |
| domestic horse | equus caballus | 1 | class |
| elk | cervus canadensis | 1 | order |
| odocoileus species | odocoileus | 1 | order |
| dromedary camel | camelus dromedarius | 1 | order |
| european roe deer | capreolus capreolus | 1 | order |
| common fallow deer | dama dama | 1 | order |
| giraffe | giraffa camelopardalis | 1 | order |
| mountain zebra | equus zebra | 1 | class |
| grevy's zebra | equus grevyi | 1 | class |
| sambar | rusa unicolor | 1 | order |
| domestic dog | canis familiaris | 1 | class |

## elephant_seal  (110 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 47 | no_match |
| human | homo sapiens | 20 | no_match |
| lion | panthera leo | 17 | no_match |
| common wildebeest | connochaetes taurinus | 14 | no_match |
| giant otter | pteronura brasiliensis | 9 | no_match |
| lowland tapir | tapirus terrestris | 4 | no_match |
| african elephant | loxodonta africana | 4 | no_match |
| domestic sheep | ovis aries | 3 | no_match |
| common eland | tragelaphus oryx | 2 | no_match |
| sika deer | cervus nippon | 1 | no_match |
| hippopotamus | hippopotamus amphibius | 1 | no_match |
| coyote | canis latrans | 1 | no_match |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | no_match |
| african buffalo | syncerus caffer | 1 | no_match |

## african_wild_dog  (98 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 30 | class |
| spotted hyaena | crocuta crocuta | 21 | order |
| domestic goat | capra aegagrus hircus | 12 | class |
| mule deer | odocoileus hemionus | 7 | class |
| domestic sheep | ovis aries | 7 | class |
| white-tailed deer | odocoileus virginianus | 6 | class |
| wild goat | capra aegagrus | 5 | class |
| wild boar | sus scrofa | 5 | class |
| red deer | cervus elaphus | 4 | class |
| plains zebra | equus quagga | 4 | class |
| domestic horse | equus caballus | 3 | class |
| bobcat | lynx rufus | 3 | order |
| jaguar | panthera onca | 3 | order |
| elk | cervus canadensis | 2 | class |
| human | homo sapiens | 2 | class |
| snow leopard | panthera uncia | 1 | order |
| blue sheep | pseudois nayaur | 1 | class |
| baboon species | papio | 1 | class |
| domestic cat | felis catus | 1 | order |
| puku | kobus vardonii | 1 | class |
| nutria | myocastor coypus | 1 | class |
| common wildebeest | connochaetes taurinus | 1 | class |
| takin | budorcas taxicolor | 1 | class |
| lion | panthera leo | 1 | order |
| african buffalo | syncerus caffer | 1 | class |
| serval | leptailurus serval | 1 | order |

## eared_seals  (98 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 42 | no_match |
| human | homo sapiens | 32 | no_match |
| common wildebeest | connochaetes taurinus | 17 | no_match |
| giant otter | pteronura brasiliensis | 14 | no_match |
| lion | panthera leo | 4 | no_match |
| coyote | canis latrans | 3 | no_match |
| golden mantled ground squirrel | callospermophilus lateralis | 2 | no_match |
| mule deer | odocoileus hemionus | 2 | no_match |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | no_match |
| american beaver | castor canadensis | 1 | no_match |
| mouflon | ovis orientalis | 1 | no_match |
| white-tailed deer | odocoileus virginianus | 1 | no_match |
| domestic sheep | ovis aries | 1 | no_match |
| north american river otter | lontra canadensis | 1 | no_match |
| domestic horse | equus caballus | 1 | no_match |
| bobcat | lynx rufus | 1 | no_match |
| lowland tapir | tapirus terrestris | 1 | no_match |
| common eland | tragelaphus oryx | 1 | no_match |
| domestic dog | canis familiaris | 1 | no_match |
| topi | damaliscus lunatus | 1 | no_match |

## dromedary_camel  (94 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 42 | order |
| domestic horse | equus caballus | 18 | class |
| red deer | cervus elaphus | 17 | order |
| mule deer | odocoileus hemionus | 11 | order |
| white-tailed deer | odocoileus virginianus | 8 | order |
| human | homo sapiens | 7 | class |
| wild goat | capra aegagrus | 5 | order |
| lion | panthera leo | 4 | class |
| plains zebra | equus quagga | 4 | class |
| domestic dog | canis familiaris | 3 | class |
| bighorn sheep | ovis canadensis | 3 | order |
| common wildebeest | connochaetes taurinus | 2 | order |
| impala | aepyceros melampus | 2 | order |
| przewalski's horse | equus ferus | 2 | class |
| moose | alces alces | 2 | order |
| domestic goat | capra aegagrus hircus | 2 | order |
| elk | cervus canadensis | 2 | order |
| domestic donkey | equus asinus | 2 | class |
| hartebeest | alcelaphus buselaphus | 2 | order |
| grevy's zebra | equus grevyi | 1 | class |
| cervidae family |  | 1 | order |
| pronghorn | antilocapra americana | 1 | order |
| olive baboon | papio anubis | 1 | class |
| giraffe | giraffa camelopardalis | 1 | order |

## baboon_genus  (88 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 10 | class |
| domestic cattle | bos taurus | 9 | class |
| domestic cat | felis catus | 7 | class |
| wild boar | sus scrofa | 7 | class |
| human | homo sapiens | 5 | order |
| arizona black-tailed prairie dog | cynomys ludovicianus | 5 | class |
| eastern gray squirrel | sciurus carolinensis | 4 | class |
| domestic dog | canis familiaris | 4 | class |
| coyote | canis latrans | 4 | class |
| giant anteater | myrmecophaga tridactyla | 3 | class |
| brush-tailed rock wallaby | petrogale penicillata | 3 | class |
| impala | aepyceros melampus | 3 | class |
| collared peccary | pecari tajacu | 2 | class |
| bobcat | lynx rufus | 2 | class |
| spotted hyaena | crocuta crocuta | 2 | class |
| american black bear | ursus americanus | 2 | class |
| woodchuck | marmota monax | 2 | class |
| red deer | cervus elaphus | 2 | class |
| yellow-bellied marmot | marmota flaviventris | 2 | class |
| black-and-gold howler monkey | alouatta caraya | 2 | order |
| eastern fox squirrel | sciurus niger | 2 | class |
| kangaroo family |  | 1 | class |
| northern chamois | rupicapra rupicapra | 1 | class |
| fisher | pekania pennanti | 1 | class |
| takin | budorcas taxicolor | 1 | class |
| african buffalo | syncerus caffer | 1 | class |
| northern raccoon | procyon lotor | 1 | class |
| eurasian red squirrel | sciurus vulgaris | 1 | class |
| red squirrel | tamiasciurus hudsonicus | 1 | class |
| capra species | capra | 1 | class |
| golden mantled ground squirrel | callospermophilus lateralis | 1 | class |
| waterbuck | kobus ellipsiprymnus | 1 | class |
| brown bear | ursus arctos | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | class |
| red-rumped agouti | dasyprocta leporina | 1 | class |
| common warthog | phacochoerus africanus | 1 | class |
| tufted deer | elaphodus cephalophus | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | class |
| lion | panthera leo | 1 | class |

## common_eland  (85 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 48 | order |
| red deer | cervus elaphus | 12 | order |
| domestic horse | equus caballus | 11 | class |
| plains zebra | equus quagga | 9 | class |
| pronghorn | antilocapra americana | 9 | order |
| human | homo sapiens | 4 | class |
| przewalski's horse | equus ferus | 2 | class |
| african elephant | loxodonta africana | 2 | class |
| mule deer | odocoileus hemionus | 2 | order |
| common fallow deer | dama dama | 1 | order |
| lion | panthera leo | 1 | class |
| domestic donkey | equus asinus | 1 | class |
| northern plains gray langur | semnopithecus entellus | 1 | class |
| elk | cervus canadensis | 1 | order |
| dromedary camel | camelus dromedarius | 1 | order |

## wild_boar  (84 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 33 | order |
| white-tailed deer | odocoileus virginianus | 17 | order |
| white-lipped peccary | tayassu pecari | 7 | order |
| spotted hyaena | crocuta crocuta | 4 | class |
| capybara | hydrochoerus hydrochaeris | 3 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 3 | class |
| domestic dog | canis familiaris | 3 | class |
| european roe deer | capreolus capreolus | 2 | order |
| collared peccary | pecari tajacu | 2 | order |
| eastern gray squirrel | sciurus carolinensis | 2 | class |
| sika deer | cervus nippon | 2 | order |
| human | homo sapiens | 2 | class |
| common wildebeest | connochaetes taurinus | 2 | order |
| south american coati | nasua nasua | 1 | class |
| common long-tailed macaque | macaca fascicularis | 1 | class |
| coyote | canis latrans | 1 | class |
| vervet monkey | chlorocebus pygerythrus | 1 | class |
| eurasian red squirrel | sciurus vulgaris | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | order |
| nutria | myocastor coypus | 1 | class |
| wild goat | capra aegagrus | 1 | order |
| reeves' muntjac | muntiacus reevesi | 1 | order |
| giant anteater | myrmecophaga tridactyla | 1 | class |
| red fox | vulpes vulpes | 1 | class |
| american bison | bison bison | 1 | order |
| lion | panthera leo | 1 | class |
| elk | cervus canadensis | 1 | order |
| nine-banded armadillo | dasypus novemcinctus | 1 | class |
| domestic sheep | ovis aries | 1 | order |

## common_warthog  (83 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 45 | order |
| common wildebeest | connochaetes taurinus | 5 | order |
| domestic horse | equus caballus | 5 | class |
| white-tailed deer | odocoileus virginianus | 5 | order |
| mule deer | odocoileus hemionus | 3 | order |
| red deer | cervus elaphus | 3 | order |
| impala | aepyceros melampus | 2 | order |
| sika deer | cervus nippon | 2 | order |
| plains zebra | equus quagga | 2 | class |
| california ground squirrel | otospermophilus beecheyi | 2 | class |
| vervet monkey | chlorocebus pygerythrus | 2 | class |
| eastern gray squirrel | sciurus carolinensis | 2 | class |
| coyote | canis latrans | 2 | class |
| domestic cat | felis catus | 2 | class |
| bobcat | lynx rufus | 1 | class |
| dhole | cuon alpinus | 1 | class |
| wild goat | capra aegagrus | 1 | order |
| eastern fox squirrel | sciurus niger | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | order |
| domestic dog | canis familiaris | 1 | class |
| elk | cervus canadensis | 1 | order |
| lion | panthera leo | 1 | class |
| przewalski's horse | equus ferus | 1 | class |
| domestic sheep | ovis aries | 1 | order |
| spotted hyaena | crocuta crocuta | 1 | class |
| african buffalo | syncerus caffer | 1 | order |

## american_bison  (80 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 28 | order |
| red deer | cervus elaphus | 24 | order |
| domestic horse | equus caballus | 11 | class |
| wild boar | sus scrofa | 8 | order |
| domestic dog | canis familiaris | 6 | class |
| mule deer | odocoileus hemionus | 5 | order |
| american black bear | ursus americanus | 3 | class |
| european roe deer | capreolus capreolus | 2 | order |
| coyote | canis latrans | 2 | class |
| hylobatidae family |  | 2 | class |
| sika deer | cervus nippon | 1 | order |
| human | homo sapiens | 1 | class |
| domestic cat | felis catus | 1 | class |
| lion | panthera leo | 1 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | class |

## lion  (80 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 25 | class |
| white-tailed deer | odocoileus virginianus | 19 | class |
| domestic dog | canis familiaris | 6 | order |
| mule deer | odocoileus hemionus | 5 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 5 | class |
| red deer | cervus elaphus | 3 | class |
| coyote | canis latrans | 3 | order |
| spotted hyaena | crocuta crocuta | 3 | order |
| plains zebra | equus quagga | 3 | class |
| common eland | tragelaphus oryx | 2 | class |
| common wildebeest | connochaetes taurinus | 2 | class |
| impala | aepyceros melampus | 2 | class |
| canis species | canis | 2 | order |
| hartebeest | alcelaphus buselaphus | 1 | class |
| giraffe | giraffa camelopardalis | 1 | class |
| kinda baboon | papio kindae | 1 | class |
| dhole | cuon alpinus | 1 | order |
| odocoileus species | odocoileus | 1 | class |
| grey wolf | canis lupus | 1 | order |
| domestic horse | equus caballus | 1 | class |

## sea_otter  (79 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| north american river otter | lontra canadensis | 32 | no_match |
| giant otter | pteronura brasiliensis | 31 | no_match |
| nutria | myocastor coypus | 25 | no_match |
| american beaver | castor canadensis | 5 | no_match |
| domestic cattle | bos taurus | 2 | no_match |
| muskrat | ondatra zibethicus | 2 | no_match |
| capybara | hydrochoerus hydrochaeris | 2 | no_match |
| northern raccoon | procyon lotor | 1 | no_match |
| common wildebeest | connochaetes taurinus | 1 | no_match |
| lion | panthera leo | 1 | no_match |
| western gray kangaroo | macropus fuliginosus | 1 | no_match |
| domestic goat | capra aegagrus hircus | 1 | no_match |
| white-tailed deer | odocoileus virginianus | 1 | no_match |
| common warthog | phacochoerus africanus | 1 | no_match |

## blackbuck  (76 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 25 | no_match |
| springbok | antidorcas marsupialis | 13 | no_match |
| pronghorn | antilocapra americana | 12 | no_match |
| thomson's gazelle | eudorcas thomsonii | 11 | no_match |
| impala | aepyceros melampus | 10 | no_match |
| grant's gazelle | nanger granti | 9 | no_match |
| domestic cattle | bos taurus | 6 | no_match |
| mule deer | odocoileus hemionus | 5 | no_match |
| guenther's dik-dik | madoqua guentheri | 4 | no_match |
| domestic goat | capra aegagrus hircus | 3 | no_match |
| common fallow deer | dama dama | 3 | no_match |
| mountain gazelle | gazella gazella | 2 | no_match |
| red deer | cervus elaphus | 2 | no_match |
| european roe deer | capreolus capreolus | 2 | no_match |
| domestic sheep | ovis aries | 2 | no_match |
| gemsbok | oryx gazella | 2 | no_match |
| domestic horse | equus caballus | 1 | no_match |
| northern red muntjac | muntiacus vaginalis | 1 | no_match |
| blue sheep | pseudois nayaur | 1 | no_match |
| wild goat | capra aegagrus | 1 | no_match |
| steenbok | raphicerus campestris | 1 | no_match |

## reindeer  (75 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 27 | order |
| pronghorn | antilocapra americana | 25 | order |
| bighorn sheep | ovis canadensis | 13 | order |
| capra species | capra | 11 | order |
| blue sheep | pseudois nayaur | 8 | order |
| domestic goat | capra aegagrus hircus | 5 | order |
| domestic sheep | ovis aries | 5 | order |
| alpine ibex | capra ibex | 5 | order |
| domestic dog | canis familiaris | 3 | class |
| domestic horse | equus caballus | 3 | class |
| northern chamois | rupicapra rupicapra | 3 | order |
| coyote | canis latrans | 2 | class |
| domestic donkey | equus asinus | 2 | class |
| common wildebeest | connochaetes taurinus | 1 | order |
| gemsbok | oryx gazella | 1 | order |
| grey wolf | canis lupus | 1 | class |
| guenther's dik-dik | madoqua guentheri | 1 | order |
| wild goat | capra aegagrus | 1 | order |
| przewalski's horse | equus ferus | 1 | class |
| giraffe | giraffa camelopardalis | 1 | order |

## domestic_cattle  (73 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 28 | order |
| domestic horse | equus caballus | 21 | class |
| human | homo sapiens | 14 | class |
| red deer | cervus elaphus | 6 | order |
| domestic dog | canis familiaris | 5 | class |
| mule deer | odocoileus hemionus | 4 | order |
| wild boar | sus scrofa | 2 | order |
| przewalski's horse | equus ferus | 1 | class |
| plains zebra | equus quagga | 1 | class |
| coyote | canis latrans | 1 | class |
| eastern fox squirrel | sciurus niger | 1 | class |
| common warthog | phacochoerus africanus | 1 | order |
| domestic donkey | equus asinus | 1 | class |

## asian_elephant  (72 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 44 | class |
| human | homo sapiens | 14 | class |
| white-tailed deer | odocoileus virginianus | 5 | class |
| red deer | cervus elaphus | 4 | class |
| common warthog | phacochoerus africanus | 3 | class |
| domestic horse | equus caballus | 3 | class |
| domestic dog | canis familiaris | 2 | class |
| elk | cervus canadensis | 1 | class |
| common wildebeest | connochaetes taurinus | 1 | class |
| eastern grey kangaroo | macropus giganteus | 1 | class |
| cheetah | acinonyx jubatus | 1 | class |
| american bison | bison bison | 1 | class |
| lowland tapir | tapirus terrestris | 1 | class |

## common_fallow_deer  (72 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 30 | order |
| pronghorn | antilocapra americana | 15 | order |
| domestic goat | capra aegagrus hircus | 8 | order |
| thomson's gazelle | eudorcas thomsonii | 7 | order |
| impala | aepyceros melampus | 5 | order |
| wild boar | sus scrofa | 3 | order |
| puku | kobus vardonii | 3 | order |
| guenther's dik-dik | madoqua guentheri | 3 | order |
| domestic dog | canis familiaris | 2 | class |
| domestic horse | equus caballus | 2 | class |
| sitatunga | tragelaphus spekii | 1 | order |
| mouflon | ovis orientalis | 1 | order |
| blue sheep | pseudois nayaur | 1 | order |
| wild goat | capra aegagrus | 1 | order |
| coyote | canis latrans | 1 | class |
| lion | panthera leo | 1 | class |

## red_kangaroo  (71 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 19 | class |
| white-tailed deer | odocoileus virginianus | 16 | class |
| lion | panthera leo | 7 | class |
| puma | puma concolor | 6 | class |
| thomson's gazelle | eudorcas thomsonii | 5 | class |
| red deer | cervus elaphus | 5 | class |
| mule deer | odocoileus hemionus | 5 | class |
| coyote | canis latrans | 2 | class |
| black-tailed jackrabbit | lepus californicus | 2 | class |
| sika deer | cervus nippon | 2 | class |
| domestic sheep | ovis aries | 2 | class |
| domestic cat | felis catus | 2 | class |
| vervet monkey | chlorocebus pygerythrus | 1 | class |
| european roe deer | capreolus capreolus | 1 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | class |
| yellow-throated marten | martes flavigula | 1 | class |
| dromedary camel | camelus dromedarius | 1 | class |
| common long-tailed macaque | macaca fascicularis | 1 | class |
| domestic horse | equus caballus | 1 | class |
| snowshoe hare | lepus americanus | 1 | class |
| common duiker | sylvicapra grimmia | 1 | class |
| pampas deer | ozotoceros bezoarticus | 1 | class |
| wild goat | capra aegagrus | 1 | class |
| culpeo | lycalopex culpaeus | 1 | class |
| human | homo sapiens | 1 | class |
| domestic dog | canis familiaris | 1 | class |

## springbok  (70 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 43 | order |
| pronghorn | antilocapra americana | 39 | order |
| red deer | cervus elaphus | 6 | order |
| mule deer | odocoileus hemionus | 4 | order |
| guanaco | lama guanicoe | 2 | order |
| common fallow deer | dama dama | 2 | order |
| domestic dog | canis familiaris | 1 | class |
| giraffe | giraffa camelopardalis | 1 | order |
| human | homo sapiens | 1 | class |

## domestic_dog  (67 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 20 | class |
| domestic cat | felis catus | 17 | order |
| white-tailed deer | odocoileus virginianus | 6 | class |
| domestic goat | capra aegagrus hircus | 5 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 5 | class |
| lion | panthera leo | 5 | order |
| giant otter | pteronura brasiliensis | 2 | order |
| human | homo sapiens | 2 | class |
| wild goat | capra aegagrus | 2 | class |
| wild boar | sus scrofa | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | class |
| eastern gray squirrel | sciurus carolinensis | 1 | class |
| american black bear | ursus americanus | 1 | order |
| thomson's gazelle | eudorcas thomsonii | 1 | class |
| black-tailed jackrabbit | lepus californicus | 1 | class |
| eurasian red squirrel | sciurus vulgaris | 1 | class |
| mule deer | odocoileus hemionus | 1 | class |
| eastern fox squirrel | sciurus niger | 1 | class |
| southern plains gray langur | semnopithecus dussumieri | 1 | class |
| tiger | panthera tigris | 1 | order |
| red deer | cervus elaphus | 1 | class |

## african_elephant  (64 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 48 | class |
| human | homo sapiens | 10 | class |
| common warthog | phacochoerus africanus | 6 | class |
| domestic horse | equus caballus | 5 | class |
| american bison | bison bison | 4 | class |
| plains zebra | equus quagga | 4 | class |
| mountain zebra | equus zebra | 3 | class |
| white-tailed deer | odocoileus virginianus | 2 | class |
| domestic dog | canis familiaris | 2 | class |
| wild boar | sus scrofa | 1 | class |
| lowland tapir | tapirus terrestris | 1 | class |
| red deer | cervus elaphus | 1 | class |
| lion | panthera leo | 1 | class |
| common wildebeest | connochaetes taurinus | 1 | class |
| domestic sheep | ovis aries | 1 | class |

## hartebeest  (60 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 28 | order |
| elk | cervus canadensis | 10 | order |
| red deer | cervus elaphus | 10 | order |
| plains zebra | equus quagga | 9 | class |
| domestic horse | equus caballus | 8 | class |
| mule deer | odocoileus hemionus | 6 | order |
| dromedary camel | camelus dromedarius | 3 | order |
| pronghorn | antilocapra americana | 2 | order |
| domestic donkey | equus asinus | 1 | class |
| european roe deer | capreolus capreolus | 1 | order |
| common warthog | phacochoerus africanus | 1 | order |
| mountain zebra | equus zebra | 1 | class |

## red_deer  (59 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 35 | order |
| puku | kobus vardonii | 10 | order |
| common wildebeest | connochaetes taurinus | 4 | order |
| pronghorn | antilocapra americana | 3 | order |
| impala | aepyceros melampus | 2 | order |
| common warthog | phacochoerus africanus | 2 | order |
| common duiker | sylvicapra grimmia | 2 | order |
| guenther's dik-dik | madoqua guentheri | 1 | order |
| red fox | vulpes vulpes | 1 | class |
| domestic horse | equus caballus | 1 | class |
| domestic sheep | ovis aries | 1 | order |
| thomson's gazelle | eudorcas thomsonii | 1 | order |
| domestic donkey | equus asinus | 1 | class |
| wild boar | sus scrofa | 1 | order |
| topi | damaliscus lunatus | 1 | order |
| grant's gazelle | nanger granti | 1 | order |

## mongoose_family  (57 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| woodchuck | marmota monax | 11 | no_match |
| arizona black-tailed prairie dog | cynomys ludovicianus | 10 | no_match |
| california ground squirrel | otospermophilus beecheyi | 4 | no_match |
| eurasian red squirrel | sciurus vulgaris | 3 | no_match |
| eastern fox squirrel | sciurus niger | 3 | no_match |
| eastern gray squirrel | sciurus carolinensis | 3 | no_match |
| domestic cattle | bos taurus | 3 | no_match |
| wild boar | sus scrofa | 2 | no_match |
| yellow-bellied marmot | marmota flaviventris | 2 | no_match |
| kinda baboon | papio kindae | 2 | no_match |
| collared peccary | pecari tajacu | 2 | no_match |
| american black bear | ursus americanus | 1 | no_match |
| guenther's dik-dik | madoqua guentheri | 1 | no_match |
| white-tailed deer | odocoileus virginianus | 1 | no_match |
| south american coati | nasua nasua | 1 | no_match |
| tayra | eira barbara | 1 | no_match |
| jaguarundi | herpailurus yagouaroundi | 1 | no_match |
| red squirrel | tamiasciurus hudsonicus | 1 | no_match |
| nine-banded armadillo | dasypus novemcinctus | 1 | no_match |
| lion | panthera leo | 1 | no_match |
| domestic horse | equus caballus | 1 | no_match |
| central american agouti | dasyprocta punctata | 1 | no_match |
| grey fox | urocyon cinereoargenteus | 1 | no_match |
| vervet monkey | chlorocebus pygerythrus | 1 | no_match |
| giant anteater | myrmecophaga tridactyla | 1 | no_match |

## african_buffalo  (56 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| common warthog | phacochoerus africanus | 15 | order |
| wild boar | sus scrofa | 13 | order |
| african elephant | loxodonta africana | 8 | class |
| red deer | cervus elaphus | 7 | order |
| white-tailed deer | odocoileus virginianus | 6 | order |
| domestic horse | equus caballus | 4 | class |
| lowland tapir | tapirus terrestris | 2 | class |
| plains zebra | equus quagga | 1 | class |
| common fallow deer | dama dama | 1 | order |
| cervidae family |  | 1 | order |
| mule deer | odocoileus hemionus | 1 | order |
| domestic dog | canis familiaris | 1 | class |
| human | homo sapiens | 1 | class |
| elk | cervus canadensis | 1 | order |

## brown_bear  (56 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 16 | class |
| white-tailed deer | odocoileus virginianus | 5 | class |
| domestic cat | felis catus | 5 | order |
| american bison | bison bison | 5 | class |
| red fox | vulpes vulpes | 4 | order |
| red deer | cervus elaphus | 3 | class |
| capra species | capra | 2 | class |
| takin | budorcas taxicolor | 2 | class |
| yellow-bellied marmot | marmota flaviventris | 2 | class |
| hylobatidae family |  | 2 | class |
| domestic dog | canis familiaris | 2 | order |
| domestic horse | equus caballus | 1 | class |
| european roe deer | capreolus capreolus | 1 | class |
| south american coati | nasua nasua | 1 | order |
| eastern fox squirrel | sciurus niger | 1 | class |
| eastern gray squirrel | sciurus carolinensis | 1 | class |
| fisher | pekania pennanti | 1 | order |
| spotted hyaena | crocuta crocuta | 1 | order |
| sika deer | cervus nippon | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | class |
| elk | cervus canadensis | 1 | class |
| mule deer | odocoileus hemionus | 1 | class |

## ring-tailed_lemur  (51 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| south american coati | nasua nasua | 16 | class |
| vervet monkey | chlorocebus pygerythrus | 16 | order |
| northern raccoon | procyon lotor | 5 | class |
| domestic cat | felis catus | 3 | class |
| domestic cattle | bos taurus | 3 | class |
| red-necked wallaby | macropus rufogriseus | 3 | class |
| plains zebra | equus quagga | 2 | class |
| eastern gray squirrel | sciurus carolinensis | 2 | class |
| wild goat | capra aegagrus | 2 | class |
| lion | panthera leo | 2 | class |
| bobcat | lynx rufus | 1 | class |
| alpine ibex | capra ibex | 1 | class |
| western gray squirrel | sciurus griseus | 1 | class |
| blue sheep | pseudois nayaur | 1 | class |
| argentine gray fox | lycalopex griseus | 1 | class |
| canis species | canis | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| whiptail wallaby | macropus parryi | 1 | class |
| sika deer | cervus nippon | 1 | class |
| golden snub-nosed monkey | rhinopithecus roxellana | 1 | order |
| tiger | panthera tigris | 1 | class |
| common long-tailed macaque | macaca fascicularis | 1 | order |
| large indian civet | viverra zibetha | 1 | class |

## chital  (48 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| impala | aepyceros melampus | 11 | order |
| bushbuck | tragelaphus scriptus | 9 | order |
| domestic cattle | bos taurus | 9 | order |
| thomson's gazelle | eudorcas thomsonii | 8 | order |
| puku | kobus vardonii | 4 | order |
| domestic goat | capra aegagrus hircus | 2 | order |
| springbok | antidorcas marsupialis | 2 | order |
| eastern gray squirrel | sciurus carolinensis | 1 | class |
| domestic dog | canis familiaris | 1 | class |
| dik-dik species | madoqua | 1 | order |
| guenther's dik-dik | madoqua guentheri | 1 | order |
| grant's gazelle | nanger granti | 1 | order |
| macaque species | macaca | 1 | class |

## mountain_goat  (48 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic dog | canis familiaris | 9 | class |
| domestic cat | felis catus | 7 | class |
| coyote | canis latrans | 6 | class |
| mule deer | odocoileus hemionus | 5 | order |
| white-tailed deer | odocoileus virginianus | 5 | order |
| pronghorn | antilocapra americana | 2 | order |
| eastern gray squirrel | sciurus carolinensis | 2 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 2 | class |
| hylobatidae family |  | 2 | class |
| red deer | cervus elaphus | 1 | order |
| grey fox | urocyon cinereoargenteus | 1 | class |
| domestic horse | equus caballus | 1 | class |
| domestic donkey | equus asinus | 1 | class |
| canis species | canis | 1 | class |
| american black bear | ursus americanus | 1 | class |
| eastern fox squirrel | sciurus niger | 1 | class |
| wild boar | sus scrofa | 1 | order |
| artiodactyla order |  | 1 | order |

## domestic_goat  (47 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic dog | canis familiaris | 27 | class |
| domestic horse | equus caballus | 4 | class |
| white-tailed deer | odocoileus virginianus | 4 | order |
| domestic cat | felis catus | 3 | class |
| common fallow deer | dama dama | 3 | order |
| hylobatidae family |  | 2 | class |
| coyote | canis latrans | 2 | class |
| wild boar | sus scrofa | 2 | order |
| pronghorn | antilocapra americana | 2 | order |
| lion | panthera leo | 2 | class |
| eurasian red squirrel | sciurus vulgaris | 1 | class |
| tufted deer | elaphodus cephalophus | 1 | order |
| red deer | cervus elaphus | 1 | order |
| honey badger | mellivora capensis | 1 | class |
| whiptail wallaby | macropus parryi | 1 | class |
| canis species | canis | 1 | class |
| human | homo sapiens | 1 | class |
| grey wolf | canis lupus | 1 | class |
| douglas's squirrel | tamiasciurus douglasii | 1 | class |
| sika deer | cervus nippon | 1 | order |

## plains_zebra  (46 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 16 | class |
| giraffe | giraffa camelopardalis | 10 | class |
| common wildebeest | connochaetes taurinus | 7 | class |
| impala | aepyceros melampus | 3 | class |
| common warthog | phacochoerus africanus | 3 | class |
| springbok | antidorcas marsupialis | 2 | class |
| pronghorn | antilocapra americana | 2 | class |
| blue sheep | pseudois nayaur | 2 | class |
| white-tailed deer | odocoileus virginianus | 2 | class |
| human | homo sapiens | 1 | class |
| hartebeest | alcelaphus buselaphus | 1 | class |
| domestic cat | felis catus | 1 | class |
| wild goat | capra aegagrus | 1 | class |
| thomson's gazelle | eudorcas thomsonii | 1 | class |
| guenther's dik-dik | madoqua guentheri | 1 | class |
| sambar | rusa unicolor | 1 | class |
| wild boar | sus scrofa | 1 | class |

## rock_hyrax  (44 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| arizona black-tailed prairie dog | cynomys ludovicianus | 8 | class |
| yellow-bellied marmot | marmota flaviventris | 7 | class |
| woodchuck | marmota monax | 5 | class |
| domestic cat | felis catus | 3 | class |
| brush-tailed rock wallaby | petrogale penicillata | 2 | class |
| wild boar | sus scrofa | 2 | class |
| belding's ground squirrel | urocitellus beldingi | 2 | class |
| sciuridae family |  | 1 | class |
| mongolian marmot | marmota sibirica | 1 | class |
| common wombat | vombatus ursinus | 1 | class |
| coyote | canis latrans | 1 | class |
| human | homo sapiens | 1 | class |
| eurasian red squirrel | sciurus vulgaris | 1 | class |
| western gray kangaroo | macropus fuliginosus | 1 | class |
| domestic cattle | bos taurus | 1 | class |
| nutria | myocastor coypus | 1 | class |
| culpeo | lycalopex culpaeus | 1 | class |
| california ground squirrel | otospermophilus beecheyi | 1 | class |
| alpine ibex | capra ibex | 1 | class |
| american pika | ochotona princeps | 1 | class |
| himalayan marmot | marmota himalayana | 1 | class |
| mule deer | odocoileus hemionus | 1 | class |
| golden mantled ground squirrel | callospermophilus lateralis | 1 | class |
| white-tailed prairie dog | cynomys leucurus | 1 | class |

## saimiri_species  (43 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| southern pig-tailed macaque | macaca nemestrina | 16 | order |
| eastern gray squirrel | sciurus carolinensis | 8 | class |
| douglas's squirrel | tamiasciurus douglasii | 4 | class |
| eastern fox squirrel | sciurus niger | 4 | class |
| common long-tailed macaque | macaca fascicularis | 3 | order |
| human | homo sapiens | 3 | order |
| vervet monkey | chlorocebus pygerythrus | 2 | order |
| black-and-gold howler monkey | alouatta caraya | 2 | order |
| domestic cattle | bos taurus | 1 | class |
| blue monkey | cercopithecus mitis | 1 | order |
| golden snub-nosed monkey | rhinopithecus roxellana | 1 | order |

## klipspringer  (42 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| mule deer | odocoileus hemionus | 30 | order |
| red deer | cervus elaphus | 4 | order |
| white-tailed deer | odocoileus virginianus | 4 | order |
| european roe deer | capreolus capreolus | 2 | order |
| coyote | canis latrans | 2 | class |
| sika deer | cervus nippon | 1 | order |
| wild boar | sus scrofa | 1 | order |

## elk  (40 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 29 | order |
| domestic horse | equus caballus | 6 | class |
| puku | kobus vardonii | 5 | order |
| pronghorn | antilocapra americana | 2 | order |
| common wildebeest | connochaetes taurinus | 2 | order |
| african elephant | loxodonta africana | 1 | class |
| giraffe | giraffa camelopardalis | 1 | order |

## kangaroo_family  (40 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 13 | no_match |
| domestic cattle | bos taurus | 5 | no_match |
| red deer | cervus elaphus | 2 | no_match |
| mule deer | odocoileus hemionus | 2 | no_match |
| common long-tailed macaque | macaca fascicularis | 2 | no_match |
| european roe deer | capreolus capreolus | 2 | no_match |
| coyote | canis latrans | 2 | no_match |
| puma | puma concolor | 2 | no_match |
| domestic horse | equus caballus | 1 | no_match |
| thomson's gazelle | eudorcas thomsonii | 1 | no_match |
| reedbuck species | redunca | 1 | no_match |
| black-fronted duiker | cephalophus nigrifrons | 1 | no_match |
| domestic cat | felis catus | 1 | no_match |
| blue sheep | pseudois nayaur | 1 | no_match |
| woodchuck | marmota monax | 1 | no_match |
| common wildebeest | connochaetes taurinus | 1 | no_match |
| eastern gray squirrel | sciurus carolinensis | 1 | no_match |
| sika deer | cervus nippon | 1 | no_match |
| vervet monkey | chlorocebus pygerythrus | 1 | no_match |
| white-tailed prairie dog | cynomys leucurus | 1 | no_match |
| eastern fox squirrel | sciurus niger | 1 | no_match |
| guenther's dik-dik | madoqua guentheri | 1 | no_match |
| bobcat | lynx rufus | 1 | no_match |

## european_rabbit  (38 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cat | felis catus | 18 | class |
| white-tailed deer | odocoileus virginianus | 6 | class |
| eastern gray squirrel | sciurus carolinensis | 6 | class |
| guenther's dik-dik | madoqua guentheri | 2 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 2 | class |
| pronghorn | antilocapra americana | 1 | class |
| domestic cattle | bos taurus | 1 | class |
| takin | budorcas taxicolor | 1 | class |
| golden mantled ground squirrel | callospermophilus lateralis | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | class |
| wild boar | sus scrofa | 1 | class |
| human | homo sapiens | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | class |
| nutria | myocastor coypus | 1 | class |
| giant otter | pteronura brasiliensis | 1 | class |
| blue sheep | pseudois nayaur | 1 | class |

## meerkat  (38 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| arizona black-tailed prairie dog | cynomys ludovicianus | 22 | class |
| coyote | canis latrans | 4 | order |
| domestic cattle | bos taurus | 4 | class |
| vervet monkey | chlorocebus pygerythrus | 3 | class |
| olive baboon | papio anubis | 2 | class |
| spotted hyaena | crocuta crocuta | 2 | order |
| american badger | taxidea taxus | 2 | order |
| kinda baboon | papio kindae | 2 | class |
| thomson's gazelle | eudorcas thomsonii | 2 | class |
| white-tailed deer | odocoileus virginianus | 2 | class |
| eastern fox squirrel | sciurus niger | 2 | class |
| yellow-bellied marmot | marmota flaviventris | 2 | class |
| california ground squirrel | otospermophilus beecheyi | 1 | class |
| white-tailed antelope squirrel | ammospermophilus leucurus | 1 | class |
| guenther's dik-dik | madoqua guentheri | 1 | class |
| northern raccoon | procyon lotor | 1 | order |
| takin | budorcas taxicolor | 1 | class |

## white-nosed_coati  (38 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 13 | class |
| eastern gray squirrel | sciurus carolinensis | 7 | class |
| yellow-throated marten | martes flavigula | 4 | order |
| fisher | pekania pennanti | 4 | order |
| domestic cat | felis catus | 3 | order |
| domestic cattle | bos taurus | 2 | class |
| canis species | canis | 2 | order |
| red squirrel | tamiasciurus hudsonicus | 2 | class |
| jaguarundi | herpailurus yagouaroundi | 1 | order |
| mule deer | odocoileus hemionus | 1 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | class |
| american black bear | ursus americanus | 1 | order |
| golden mantled ground squirrel | callospermophilus lateralis | 1 | class |
| red panda | ailurus fulgens | 1 | order |
| brown bear | ursus arctos | 1 | order |
| domestic dog | canis familiaris | 1 | order |
| northern palawan tree squirrel | sundasciurus juvencus | 1 | class |
| eastern fox squirrel | sciurus niger | 1 | class |
| red fox | vulpes vulpes | 1 | order |
| wild boar | sus scrofa | 1 | class |
| american marten | martes americana | 1 | order |

## cebus_species  (37 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| yellow-throated marten | martes flavigula | 11 | class |
| blue monkey | cercopithecus mitis | 8 | order |
| domestic cattle | bos taurus | 3 | class |
| stump-tailed macaque | macaca arctoides | 3 | order |
| domestic cat | felis catus | 2 | class |
| southern pig-tailed macaque | macaca nemestrina | 1 | order |
| takin | budorcas taxicolor | 1 | class |
| giant panda | ailuropoda melanoleuca | 1 | class |
| grey-cheeked mangabey | lophocebus albigena | 1 | order |
| red-shanked douc langur | pygathrix nemaeus | 1 | order |
| macaque species | macaca | 1 | order |
| american black bear | ursus americanus | 1 | class |
| puma | puma concolor | 1 | class |
| black-and-gold howler monkey | alouatta caraya | 1 | order |
| nutria | myocastor coypus | 1 | class |
| angolan colobus | colobus angolensis | 1 | order |
| chacma baboon | papio ursinus | 1 | order |

## nyala  (37 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 33 | order |
| chital | axis axis | 5 | order |
| mule deer | odocoileus hemionus | 2 | order |
| red deer | cervus elaphus | 2 | order |
| sika deer | cervus nippon | 1 | order |
| human | homo sapiens | 1 | class |
| baboon species | papio | 1 | class |

## asiatic_wild_ass  (36 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| pronghorn | antilocapra americana | 17 | class |
| impala | aepyceros melampus | 13 | class |
| domestic cattle | bos taurus | 9 | class |
| springbok | antidorcas marsupialis | 5 | class |
| wild goat | capra aegagrus | 5 | class |
| dromedary camel | camelus dromedarius | 3 | class |
| thomson's gazelle | eudorcas thomsonii | 2 | class |
| giraffe | giraffa camelopardalis | 2 | class |
| domestic sheep | ovis aries | 1 | class |
| domestic dog | canis familiaris | 1 | class |
| common eland | tragelaphus oryx | 1 | class |
| grant's gazelle | nanger granti | 1 | class |
| coyote | canis latrans | 1 | class |
| red deer | cervus elaphus | 1 | class |

## nilgai  (36 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 17 | order |
| red deer | cervus elaphus | 8 | order |
| domestic horse | equus caballus | 3 | class |
| elk | cervus canadensis | 3 | order |
| mule deer | odocoileus hemionus | 2 | order |
| plains zebra | equus quagga | 2 | class |
| pronghorn | antilocapra americana | 2 | order |
| odocoileus species | odocoileus | 1 | order |
| human | homo sapiens | 1 | class |

## vervet_monkey  (36 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| eastern gray squirrel | sciurus carolinensis | 15 | class |
| white-tailed deer | odocoileus virginianus | 7 | class |
| domestic cattle | bos taurus | 2 | class |
| brush-tailed rock wallaby | petrogale penicillata | 2 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 2 | class |
| impala | aepyceros melampus | 2 | class |
| white-tailed antelope squirrel | ammospermophilus leucurus | 1 | class |
| western gray kangaroo | macropus fuliginosus | 1 | class |
| swamp wallaby | wallabia bicolor | 1 | class |
| northern raccoon | procyon lotor | 1 | class |
| guenther's dik-dik | madoqua guentheri | 1 | class |
| douglas's squirrel | tamiasciurus douglasii | 1 | class |
| wild boar | sus scrofa | 1 | class |
| domestic cat | felis catus | 1 | class |
| common warthog | phacochoerus africanus | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | class |
| eastern fox squirrel | sciurus niger | 1 | class |

## northern_chamois  (34 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 10 | order |
| red deer | cervus elaphus | 10 | order |
| elk | cervus canadensis | 3 | order |
| wild boar | sus scrofa | 3 | order |
| mule deer | odocoileus hemionus | 3 | order |
| domestic horse | equus caballus | 2 | class |
| domestic dog | canis familiaris | 2 | class |
| european roe deer | capreolus capreolus | 2 | order |
| red fox | vulpes vulpes | 1 | class |
| common warthog | phacochoerus africanus | 1 | order |

## gorilla_species  (33 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 16 | class |
| american black bear | ursus americanus | 5 | class |
| common wildebeest | connochaetes taurinus | 3 | class |
| hylobatidae family |  | 2 | order |
| american bison | bison bison | 1 | class |
| wild boar | sus scrofa | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| stump-tailed macaque | macaca arctoides | 1 | order |
| red-shanked douc langur | pygathrix nemaeus | 1 | order |
| spectacled bear | tremarctos ornatus | 1 | class |
| lion | panthera leo | 1 | class |
| common warthog | phacochoerus africanus | 1 | class |
| giant otter | pteronura brasiliensis | 1 | class |
| domestic horse | equus caballus | 1 | class |

## eulemur_species  (32 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| black-and-gold howler monkey | alouatta caraya | 7 | order |
| south american coati | nasua nasua | 6 | class |
| eastern fox squirrel | sciurus niger | 4 | class |
| common squirrel monkey | saimiri sciureus | 3 | order |
| swamp wallaby | wallabia bicolor | 2 | class |
| maroon leaf monkey | presbytis rubicunda | 1 | order |
| blue monkey | cercopithecus mitis | 1 | order |
| eastern gray squirrel | sciurus carolinensis | 1 | class |
| koala | phascolarctos cinereus | 1 | class |
| american black bear | ursus americanus | 1 | class |
| lion | panthera leo | 1 | class |
| domestic cattle | bos taurus | 1 | class |
| sciuridae family |  | 1 | class |
| vervet monkey | chlorocebus pygerythrus | 1 | order |
| sambar | rusa unicolor | 1 | class |
| woodchuck | marmota monax | 1 | class |
| tayra | eira barbara | 1 | class |
| douglas's squirrel | tamiasciurus douglasii | 1 | class |
| dhole | cuon alpinus | 1 | class |
| roosevelts' muntjac | muntiacus rooseveltorum | 1 | class |
| yellow-bellied marmot | marmota flaviventris | 1 | class |

## kob  (31 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 30 | order |
| pampas deer | ozotoceros bezoarticus | 3 | order |
| giraffe | giraffa camelopardalis | 1 | order |
| european roe deer | capreolus capreolus | 1 | order |
| red deer | cervus elaphus | 1 | order |

## reedbuck_genus  (31 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 27 | order |
| european roe deer | capreolus capreolus | 3 | order |
| mule deer | odocoileus hemionus | 2 | order |
| red deer | cervus elaphus | 1 | order |
| common fallow deer | dama dama | 1 | order |
| sika deer | cervus nippon | 1 | order |

## moose  (30 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 21 | order |
| domestic dog | canis familiaris | 4 | class |
| common wildebeest | connochaetes taurinus | 2 | order |
| pronghorn | antilocapra americana | 1 | order |
| american bison | bison bison | 1 | order |
| wild goat | capra aegagrus | 1 | order |
| dromedary camel | camelus dromedarius | 1 | order |
| american black bear | ursus americanus | 1 | class |

## sable_antelope  (30 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 12 | order |
| red deer | cervus elaphus | 9 | order |
| domestic horse | equus caballus | 7 | class |
| domestic donkey | equus asinus | 5 | class |
| pronghorn | antilocapra americana | 2 | order |
| common warthog | phacochoerus africanus | 2 | order |
| plains zebra | equus quagga | 2 | class |
| domestic dog | canis familiaris | 1 | class |
| mule deer | odocoileus hemionus | 1 | order |

## arizona_black-tailed_prairie_dog  (28 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 8 | class |
| domestic cattle | bos taurus | 6 | class |
| red deer | cervus elaphus | 3 | class |
| domestic goat | capra aegagrus hircus | 2 | class |
| red fox | vulpes vulpes | 2 | class |
| fossa | cryptoprocta ferox | 1 | class |
| culpeo | lycalopex culpaeus | 1 | class |
| lion | panthera leo | 1 | class |
| mule deer | odocoileus hemionus | 1 | class |
| nine-banded armadillo | dasypus novemcinctus | 1 | class |
| domestic cat | felis catus | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | order |
| puma | puma concolor | 1 | class |

## bornean_orangutan  (27 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 9 | class |
| southern pig-tailed macaque | macaca nemestrina | 7 | order |
| eurasian red squirrel | sciurus vulgaris | 3 | class |
| eastern fox squirrel | sciurus niger | 3 | class |
| domestic horse | equus caballus | 3 | class |
| eastern gray squirrel | sciurus carolinensis | 2 | class |
| domestic goat | capra aegagrus hircus | 2 | class |
| stump-tailed macaque | macaca arctoides | 2 | order |
| american bison | bison bison | 1 | class |
| common wildebeest | connochaetes taurinus | 1 | class |
| domestic dog | canis familiaris | 1 | class |

## saguinus_species  (27 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| common long-tailed macaque | macaca fascicularis | 7 | order |
| blue monkey | cercopithecus mitis | 5 | order |
| domestic cattle | bos taurus | 4 | class |
| vervet monkey | chlorocebus pygerythrus | 3 | order |
| eastern gray squirrel | sciurus carolinensis | 2 | class |
| grey-cheeked mangabey | lophocebus albigena | 1 | order |
| purús red howler monkey | alouatta puruensis | 1 | order |
| golden mantled ground squirrel | callospermophilus lateralis | 1 | class |
| maroon leaf monkey | presbytis rubicunda | 1 | order |
| southern plains gray langur | semnopithecus dussumieri | 1 | order |
| douglas's squirrel | tamiasciurus douglasii | 1 | class |
| american pika | ochotona princeps | 1 | class |
| american black bear | ursus americanus | 1 | class |
| human | homo sapiens | 1 | order |
| domestic goat | capra aegagrus hircus | 1 | class |

## callithrix_species  (26 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| eastern gray squirrel | sciurus carolinensis | 8 | class |
| domestic cat | felis catus | 6 | class |
| blue monkey | cercopithecus mitis | 3 | order |
| carruther's mountain squirrel | funisciurus carruthersi | 1 | class |
| stump-tailed macaque | macaca arctoides | 1 | order |
| coyote | canis latrans | 1 | class |
| black-and-gold howler monkey | alouatta caraya | 1 | order |
| human | homo sapiens | 1 | order |
| southern pig-tailed macaque | macaca nemestrina | 1 | order |
| large-headed capuchin | sapajus macrocephalus | 1 | order |
| yellow-bellied marmot | marmota flaviventris | 1 | class |
| douglas's squirrel | tamiasciurus douglasii | 1 | class |

## domestic_water_buffalo  (26 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 9 | order |
| wild boar | sus scrofa | 6 | order |
| domestic horse | equus caballus | 5 | class |
| red deer | cervus elaphus | 4 | order |
| domestic dog | canis familiaris | 3 | class |
| lion | panthera leo | 1 | class |
| sika deer | cervus nippon | 1 | order |
| lowland tapir | tapirus terrestris | 1 | class |
| human | homo sapiens | 1 | class |

## common_wombat  (25 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| sika deer | cervus nippon | 3 | class |
| eastern gray squirrel | sciurus carolinensis | 3 | class |
| domestic cattle | bos taurus | 2 | class |
| brush-tailed rock wallaby | petrogale penicillata | 2 | order |
| western gray squirrel | sciurus griseus | 2 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 2 | class |
| california ground squirrel | otospermophilus beecheyi | 1 | class |
| eurasian red squirrel | sciurus vulgaris | 1 | class |
| nutria | myocastor coypus | 1 | class |
| pine marten | martes martes | 1 | class |
| american black bear | ursus americanus | 1 | class |
| human | homo sapiens | 1 | class |
| woodchuck | marmota monax | 1 | class |
| grey fox | urocyon cinereoargenteus | 1 | class |
| coyote | canis latrans | 1 | class |
| common wildebeest | connochaetes taurinus | 1 | class |
| north american porcupine | erethizon dorsatum | 1 | class |

## japanese_macaque  (24 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| american black bear | ursus americanus | 11 | class |
| eastern gray squirrel | sciurus carolinensis | 5 | class |
| domestic cat | felis catus | 3 | class |
| domestic cattle | bos taurus | 2 | class |
| mule deer | odocoileus hemionus | 2 | class |
| lion | panthera leo | 1 | class |
| domestic dog | canis familiaris | 1 | class |
| common wombat | vombatus ursinus | 1 | class |
| eastern fox squirrel | sciurus niger | 1 | class |
| yellow-bellied marmot | marmota flaviventris | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| tufted deer | elaphodus cephalophus | 1 | class |

## giant_panda  (23 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 8 | class |
| wolverine | gulo gulo | 4 | order |
| domestic cat | felis catus | 4 | order |
| domestic dog | canis familiaris | 3 | order |
| takin | budorcas taxicolor | 1 | class |
| human | homo sapiens | 1 | class |
| sika deer | cervus nippon | 1 | class |
| blue monkey | cercopithecus mitis | 1 | class |
| yellow-throated marten | martes flavigula | 1 | order |
| honey badger | mellivora capensis | 1 | order |
| striped skunk | mephitis mephitis | 1 | order |
| lion | panthera leo | 1 | order |
| azaras's capuchin | sapajus cay | 1 | class |

## collared_peccary  (22 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| wild boar | sus scrofa | 16 | order |
| domestic cattle | bos taurus | 4 | order |
| white-tailed deer | odocoileus virginianus | 3 | order |
| american black bear | ursus americanus | 1 | class |
| golden jackal | canis aureus | 1 | class |
| domestic cat | felis catus | 1 | class |

## spotted_hyaena  (22 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 8 | class |
| red deer | cervus elaphus | 2 | class |
| bobcat | lynx rufus | 2 | order |
| white-tailed deer | odocoileus virginianus | 2 | class |
| northern raccoon | procyon lotor | 1 | order |
| common warthog | phacochoerus africanus | 1 | class |
| northern chamois | rupicapra rupicapra | 1 | class |
| giraffe | giraffa camelopardalis | 1 | class |
| domestic dog | canis familiaris | 1 | order |
| dhole | cuon alpinus | 1 | order |
| iberian lynx | lynx pardinus | 1 | order |
| lion | panthera leo | 1 | order |
| nutria | myocastor coypus | 1 | class |
| leopard cat | prionailurus bengalensis | 1 | order |

## red_fox  (21 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 7 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 3 | class |
| domestic cattle | bos taurus | 2 | class |
| lion | panthera leo | 2 | order |
| domestic cat | felis catus | 1 | order |
| eastern cottontail | sylvilagus floridanus | 1 | class |
| common wildebeest | connochaetes taurinus | 1 | class |
| western gray kangaroo | macropus fuliginosus | 1 | class |
| wild boar | sus scrofa | 1 | class |
| american bison | bison bison | 1 | class |
| eastern fox squirrel | sciurus niger | 1 | class |

## blesbok  (20 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 8 | order |
| red deer | cervus elaphus | 5 | order |
| pronghorn | antilocapra americana | 2 | order |
| mule deer | odocoileus hemionus | 2 | order |
| common fallow deer | dama dama | 2 | order |
| domestic horse | equus caballus | 2 | class |
| european roe deer | capreolus capreolus | 1 | order |
| equus species | equus | 1 | class |
| sika deer | cervus nippon | 1 | order |

## domestic_cat  (20 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic dog | canis familiaris | 5 | order |
| domestic cattle | bos taurus | 3 | class |
| human | homo sapiens | 3 | class |
| domestic horse | equus caballus | 1 | class |
| przewalski's horse | equus ferus | 1 | class |
| eastern gray squirrel | sciurus carolinensis | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | class |
| coyote | canis latrans | 1 | order |
| white-tailed deer | odocoileus virginianus | 1 | class |
| giant anteater | myrmecophaga tridactyla | 1 | class |
| woodchuck | marmota monax | 1 | class |
| elk | cervus canadensis | 1 | class |

## gemsbok  (20 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| pronghorn | antilocapra americana | 7 | order |
| domestic horse | equus caballus | 6 | class |
| domestic donkey | equus asinus | 2 | class |
| przewalski's horse | equus ferus | 2 | class |
| mule deer | odocoileus hemionus | 1 | order |
| human | homo sapiens | 1 | class |
| elk | cervus canadensis | 1 | order |
| white-tailed deer | odocoileus virginianus | 1 | order |
| dromedary camel | camelus dromedarius | 1 | order |
| red deer | cervus elaphus | 1 | order |
| african elephant | loxodonta africana | 1 | class |

## giant_otter  (20 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 8 | class |
| nutria | myocastor coypus | 4 | class |
| human | homo sapiens | 3 | class |
| domestic dog | canis familiaris | 2 | order |
| american black bear | ursus americanus | 1 | order |
| african elephant | loxodonta africana | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | class |
| american bison | bison bison | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| domestic cat | felis catus | 1 | order |

## macaque_species  (20 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| eastern gray squirrel | sciurus carolinensis | 5 | class |
| coyote | canis latrans | 3 | class |
| domestic cattle | bos taurus | 2 | class |
| eastern fox squirrel | sciurus niger | 2 | class |
| golden mantled ground squirrel | callospermophilus lateralis | 2 | class |
| human | homo sapiens | 2 | order |
| red deer | cervus elaphus | 1 | class |
| common wombat | vombatus ursinus | 1 | class |
| western gray squirrel | sciurus griseus | 1 | class |
| wild boar | sus scrofa | 1 | class |
| domestic dog | canis familiaris | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| domestic cat | felis catus | 1 | class |
| tayra | eira barbara | 1 | class |

## sambar  (20 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 13 | order |
| coyote | canis latrans | 2 | class |
| chinese goral | naemorhedus griseus | 1 | order |
| northern chamois | rupicapra rupicapra | 1 | order |
| golden jackal | canis aureus | 1 | class |
| domestic sheep | ovis aries | 1 | order |
| domestic horse | equus caballus | 1 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | class |
| impala | aepyceros melampus | 1 | order |
| guenther's dik-dik | madoqua guentheri | 1 | order |
| eastern fox squirrel | sciurus niger | 1 | class |
| przewalski's horse | equus ferus | 1 | class |

## waterbuck  (20 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 10 | order |
| red deer | cervus elaphus | 5 | order |
| kinda baboon | papio kindae | 2 | class |
| mule deer | odocoileus hemionus | 1 | order |
| european roe deer | capreolus capreolus | 1 | order |
| baboon species | papio | 1 | class |
| common warthog | phacochoerus africanus | 1 | order |
| sika deer | cervus nippon | 1 | order |

## muridae_family  (20 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| human | homo sapiens | 6 | no_match |
| rodent |  | 4 | no_match |
| kaluta | dasykaluta rosamondae | 2 | no_match |
| capybara | hydrochoerus hydrochaeris | 2 | no_match |
| vervet monkey | chlorocebus pygerythrus | 2 | no_match |
| golden mantled ground squirrel | callospermophilus lateralis | 1 | no_match |
| red fox | vulpes vulpes | 1 | no_match |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | no_match |
| striped skunk | mephitis mephitis | 1 | no_match |
| kha-nyou | laonastes aenigmamus | 1 | no_match |
| domestic cat | felis catus | 1 | no_match |
| greater hog badger | arctonyx collaris | 1 | no_match |
| common long-tailed macaque | macaca fascicularis | 1 | no_match |
| nutria | myocastor coypus | 1 | no_match |
| desert cottontail | sylvilagus audubonii | 1 | no_match |
| brush-tailed rock wallaby | petrogale penicillata | 1 | no_match |
| virginia opossum | didelphis virginiana | 1 | no_match |
| black agouti | dasyprocta fuliginosa | 1 | no_match |
| mammal |  | 1 | no_match |

## chimpanzee  (19 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| stump-tailed macaque | macaca arctoides | 7 | order |
| american black bear | ursus americanus | 3 | class |
| domestic cattle | bos taurus | 3 | class |
| hatinh langur | trachypithecus hatinhensis | 2 | order |
| blue monkey | cercopithecus mitis | 1 | order |
| domestic goat | capra aegagrus hircus | 1 | class |
| eastern gray squirrel | sciurus carolinensis | 1 | class |
| milne-edwards' macaque | macaca thibetana | 1 | order |
| common wildebeest | connochaetes taurinus | 1 | class |
| giant anteater | myrmecophaga tridactyla | 1 | class |
| sumatran serow | capricornis sumatraensis | 1 | class |

## white-tailed_deer  (19 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| impala | aepyceros melampus | 10 | order |
| pronghorn | antilocapra americana | 4 | order |
| giraffe | giraffa camelopardalis | 3 | order |
| blue sheep | pseudois nayaur | 1 | order |
| domestic goat | capra aegagrus hircus | 1 | order |
| coyote | canis latrans | 1 | class |
| guenther's dik-dik | madoqua guentheri | 1 | order |
| thomson's gazelle | eudorcas thomsonii | 1 | order |
| american black bear | ursus americanus | 1 | class |
| springbok | antidorcas marsupialis | 1 | order |
| american marten | martes americana | 1 | class |
| domestic horse | equus caballus | 1 | class |

## ateles_species  (18 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| angolan colobus | colobus angolensis | 2 | order |
| golden snub-nosed monkey | rhinopithecus roxellana | 2 | order |
| grey-cheeked mangabey | lophocebus albigena | 2 | order |
| coyote | canis latrans | 1 | class |
| hatinh langur | trachypithecus hatinhensis | 1 | order |
| blue monkey | cercopithecus mitis | 1 | order |
| domestic cattle | bos taurus | 1 | class |
| human | homo sapiens | 1 | order |
| canis species | canis | 1 | class |
| macaque species | macaca | 1 | order |
| common long-tailed macaque | macaca fascicularis | 1 | order |
| southern plains gray langur | semnopithecus dussumieri | 1 | order |
| blue sheep | pseudois nayaur | 1 | class |
| maroon leaf monkey | presbytis rubicunda | 1 | order |
| vervet monkey | chlorocebus pygerythrus | 1 | order |
| american black bear | ursus americanus | 1 | class |

## walrus  (17 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 10 | no_match |
| african elephant | loxodonta africana | 6 | no_match |
| domestic sheep | ovis aries | 5 | no_match |
| human | homo sapiens | 3 | no_match |
| domestic horse | equus caballus | 1 | no_match |
| lowland tapir | tapirus terrestris | 1 | no_match |

## yak  (17 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| hylobatidae family |  | 10 | class |
| domestic dog | canis familiaris | 5 | class |
| domestic horse | equus caballus | 2 | class |
| domestic cat | felis catus | 2 | class |
| american black bear | ursus americanus | 1 | class |
| mule deer | odocoileus hemionus | 1 | order |

## howler_monkey_genus  (16 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| large-headed capuchin | sapajus macrocephalus | 4 | order |
| blue monkey | cercopithecus mitis | 2 | order |
| american black bear | ursus americanus | 2 | class |
| maroon leaf monkey | presbytis rubicunda | 2 | order |
| domestic cattle | bos taurus | 2 | class |
| angolan colobus | colobus angolensis | 2 | order |
| stump-tailed macaque | macaca arctoides | 1 | order |
| eurasian red squirrel | sciurus vulgaris | 1 | class |
| grey-cheeked mangabey | lophocebus albigena | 1 | order |
| emperor tamarin | saguinus imperator | 1 | order |
| domestic goat | capra aegagrus hircus | 1 | class |

## north_american_river_otter  (16 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| nutria | myocastor coypus | 12 | class |
| american beaver | castor canadensis | 1 | class |
| american black bear | ursus americanus | 1 | order |
| domestic cattle | bos taurus | 1 | class |
| capra species | capra | 1 | class |

## european_bison  (15 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| red deer | cervus elaphus | 4 | order |
| domestic horse | equus caballus | 3 | class |
| domestic cat | felis catus | 2 | class |
| human | homo sapiens | 2 | class |
| domestic dog | canis familiaris | 2 | class |
| wild boar | sus scrofa | 2 | order |
| white-tailed deer | odocoileus virginianus | 2 | order |
| african elephant | loxodonta africana | 1 | class |
| sambar | rusa unicolor | 1 | order |

## weasel_species  (15 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 3 | class |
| dingo | canis lupus dingo | 2 | order |
| domestic goat | capra aegagrus hircus | 1 | class |
| northern raccoon | procyon lotor | 1 | order |
| american black bear | ursus americanus | 1 | order |
| american bison | bison bison | 1 | class |
| domestic cat | felis catus | 1 | order |
| eastern chipmunk | tamias striatus | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| domestic dog | canis familiaris | 1 | order |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | class |
| red fox | vulpes vulpes | 1 | order |
| spotted hyaena | crocuta crocuta | 1 | order |

## european_roe_deer  (13 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 7 | order |
| guenther's dik-dik | madoqua guentheri | 3 | order |
| pronghorn | antilocapra americana | 1 | order |
| wild goat | capra aegagrus | 1 | order |
| puku | kobus vardonii | 1 | order |
| domestic sheep | ovis aries | 1 | order |

## european_hare  (13 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 7 | class |
| mule deer | odocoileus hemionus | 1 | class |
| puma | puma concolor | 1 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | class |
| blue sheep | pseudois nayaur | 1 | class |
| domestic cattle | bos taurus | 1 | class |
| red deer | cervus elaphus | 1 | class |
| coyote | canis latrans | 1 | class |

## american_black_bear  (12 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 8 | class |
| american bison | bison bison | 2 | class |
| red deer | cervus elaphus | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| northern raccoon | procyon lotor | 1 | order |

## cercopithecus_species  (12 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 4 | class |
| eastern gray squirrel | sciurus carolinensis | 3 | class |
| human | homo sapiens | 2 | order |
| south american coati | nasua nasua | 2 | class |
| red-necked wallaby | macropus rufogriseus | 1 | class |
| domestic cat | felis catus | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |

## leaf_monkeys_genus  (12 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| human | homo sapiens | 3 | order |
| takin | budorcas taxicolor | 2 | class |
| domestic cattle | bos taurus | 2 | class |
| domestic cat | felis catus | 1 | class |
| canis species | canis | 1 | class |
| domestic dog | canis familiaris | 1 | class |
| giant anteater | myrmecophaga tridactyla | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | class |

## mule_deer  (12 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 5 | order |
| coyote | canis latrans | 1 | class |
| woodchuck | marmota monax | 1 | class |
| thomson's gazelle | eudorcas thomsonii | 1 | order |
| wild goat | capra aegagrus | 1 | order |
| pronghorn | antilocapra americana | 1 | order |
| guenther's dik-dik | madoqua guentheri | 1 | order |
| sitatunga | tragelaphus spekii | 1 | order |

## quokka  (12 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| european rabbit | oryctolagus cuniculus | 1 | class |
| wild boar | sus scrofa | 1 | class |
| domestic cattle | bos taurus | 1 | class |
| blue monkey | cercopithecus mitis | 1 | class |
| northern raccoon | procyon lotor | 1 | class |
| lion | panthera leo | 1 | class |
| common long-tailed macaque | macaca fascicularis | 1 | class |
| red deer | cervus elaphus | 1 | class |
| woodchuck | marmota monax | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | class |
| golden jackal | canis aureus | 1 | class |
| nutria | myocastor coypus | 1 | class |

## roan_antelope  (12 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| red deer | cervus elaphus | 7 | order |
| white-tailed deer | odocoileus virginianus | 4 | order |
| plains zebra | equus quagga | 2 | class |
| grevy's zebra | equus grevyi | 1 | class |
| dromedary camel | camelus dromedarius | 1 | order |
| elk | cervus canadensis | 1 | order |
| coyote | canis latrans | 1 | class |
| common warthog | phacochoerus africanus | 1 | order |

## saiga  (12 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| pronghorn | antilocapra americana | 11 | no_match |
| impala | aepyceros melampus | 6 | no_match |
| domestic cattle | bos taurus | 6 | no_match |
| white-tailed deer | odocoileus virginianus | 5 | no_match |
| mule deer | odocoileus hemionus | 4 | no_match |
| hartebeest | alcelaphus buselaphus | 1 | no_match |
| domestic goat | capra aegagrus hircus | 1 | no_match |

## red_panda  (12 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 6 | class |
| human | homo sapiens | 2 | class |
| south american coati | nasua nasua | 1 | order |
| northern raccoon | procyon lotor | 1 | order |
| domestic sheep | ovis aries | 1 | class |
| american black bear | ursus americanus | 1 | order |

## dingo  (12 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 5 | class |
| domestic cat | felis catus | 2 | order |
| lion | panthera leo | 2 | order |
| thomson's gazelle | eudorcas thomsonii | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | class |
| mule deer | odocoileus hemionus | 1 | class |

## alpine_marmot  (11 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 2 | class |
| coyote | canis latrans | 1 | class |
| red deer | cervus elaphus | 1 | class |
| rock hyrax | procavia capensis | 1 | class |
| guenther's dik-dik | madoqua guentheri | 1 | class |
| american black bear | ursus americanus | 1 | class |
| domestic cat | felis catus | 1 | class |
| northern raccoon | procyon lotor | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | order |
| common wombat | vombatus ursinus | 1 | class |

## giraffe  (11 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| plains zebra | equus quagga | 3 | class |
| jaguar | panthera onca | 2 | class |
| guenther's dik-dik | madoqua guentheri | 2 | order |
| white-tailed deer | odocoileus virginianus | 1 | order |
| impala | aepyceros melampus | 1 | order |
| sika deer | cervus nippon | 1 | order |
| grant's gazelle | nanger granti | 1 | order |
| dromedary camel | camelus dromedarius | 1 | order |

## grant's_gazelle  (11 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 12 | order |

## nutria  (11 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| wild boar | sus scrofa | 3 | class |
| capybara | hydrochoerus hydrochaeris | 2 | order |
| american beaver | castor canadensis | 2 | order |
| woodchuck | marmota monax | 1 | order |
| red deer | cervus elaphus | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | class |
| northern raccoon | procyon lotor | 1 | class |
| eastern gray squirrel | sciurus carolinensis | 1 | order |

## beaver_genus  (11 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| nutria | myocastor coypus | 8 | order |
| north american river otter | lontra canadensis | 2 | class |
| human | homo sapiens | 1 | class |

## black_wildebeest  (10 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| plains zebra | equus quagga | 4 | class |
| domestic horse | equus caballus | 3 | class |
| elk | cervus canadensis | 3 | order |
| european roe deer | capreolus capreolus | 2 | order |
| red deer | cervus elaphus | 2 | order |
| grevy's zebra | equus grevyi | 1 | class |

## dhole  (10 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 5 | class |
| domestic cattle | bos taurus | 2 | class |
| red deer | cervus elaphus | 2 | class |
| domestic goat | capra aegagrus hircus | 1 | class |
| mule deer | odocoileus hemionus | 1 | class |

## koala  (10 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| red-necked wallaby | macropus rufogriseus | 2 | order |
| eastern gray squirrel | sciurus carolinensis | 2 | class |
| eastern fox squirrel | sciurus niger | 1 | class |
| blue sheep | pseudois nayaur | 1 | class |
| common long-tailed macaque | macaca fascicularis | 1 | class |
| mule deer | odocoileus hemionus | 1 | class |
| dromedary camel | camelus dromedarius | 1 | class |
| lord derby's scaly-tailed squirrel | anomalurus derbianus | 1 | class |

## alpine_ibex  (9 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| red deer | cervus elaphus | 5 | order |
| domestic dog | canis familiaris | 1 | class |
| elk | cervus canadensis | 1 | order |
| wild boar | sus scrofa | 1 | order |
| white-tailed deer | odocoileus virginianus | 1 | order |
| mule deer | odocoileus hemionus | 1 | order |

## agouti_genus  (9 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 3 | class |
| eastern gray squirrel | sciurus carolinensis | 3 | order |
| rodent |  | 1 | order |
| domestic cat | felis catus | 1 | class |
| fossa | cryptoprocta ferox | 1 | class |
| wild boar | sus scrofa | 1 | class |

## hippopotamus  (9 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 7 | order |
| coyote | canis latrans | 1 | class |
| african buffalo | syncerus caffer | 1 | order |
| human | homo sapiens | 1 | class |
| giant otter | pteronura brasiliensis | 1 | class |

## northern_raccoon  (9 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cat | felis catus | 3 | order |
| wild boar | sus scrofa | 1 | class |
| eastern gray squirrel | sciurus carolinensis | 1 | class |
| plains zebra | equus quagga | 1 | class |
| domestic cattle | bos taurus | 1 | class |
| bobcat | lynx rufus | 1 | order |
| brown bear | ursus arctos | 1 | order |

## spectacled_bear  (9 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 6 | class |
| human | homo sapiens | 4 | class |
| fisher | pekania pennanti | 1 | order |

## bat-eared_fox  (8 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| spotted hyaena | crocuta crocuta | 1 | order |
| yellow-bellied marmot | marmota flaviventris | 1 | class |
| guenther's dik-dik | madoqua guentheri | 1 | class |
| brush-tailed rock wallaby | petrogale penicillata | 1 | class |
| common warthog | phacochoerus africanus | 1 | class |
| wild boar | sus scrofa | 1 | class |
| domestic cattle | bos taurus | 1 | class |
| common wildebeest | connochaetes taurinus | 1 | class |

## eurasian_otter  (8 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| nutria | myocastor coypus | 3 | class |
| wild boar | sus scrofa | 2 | class |
| red deer | cervus elaphus | 1 | class |
| northern raccoon | procyon lotor | 1 | order |
| human | homo sapiens | 1 | class |

## golden_jackal  (8 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 4 | class |
| domestic cattle | bos taurus | 3 | class |
| western gray kangaroo | macropus fuliginosus | 2 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | class |

## lowland_tapir  (8 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 3 | class |
| common long-tailed macaque | macaca fascicularis | 1 | class |
| takin | budorcas taxicolor | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| wild boar | sus scrofa | 1 | class |
| sambar | rusa unicolor | 1 | class |

## raccoon_dog  (8 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| wild boar | sus scrofa | 3 | class |
| greater hog badger | arctonyx collaris | 3 | order |
| northern raccoon | procyon lotor | 2 | order |
| domestic cattle | bos taurus | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | class |

## steenbok  (8 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 7 | order |
| domestic dog | canis familiaris | 1 | class |

## thomson's_gazelle  (8 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 6 | order |
| red fox | vulpes vulpes | 1 | class |
| plains zebra | equus quagga | 1 | class |
| coyote | canis latrans | 1 | class |
| mule deer | odocoileus hemionus | 1 | order |
| pronghorn | antilocapra americana | 1 | order |
| common fallow deer | dama dama | 1 | order |

## pinniped_clade  (8 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| human | homo sapiens | 4 | no_match |
| domestic cattle | bos taurus | 3 | no_match |
| red deer | cervus elaphus | 1 | no_match |
| african elephant | loxodonta africana | 1 | no_match |
| lion | panthera leo | 1 | no_match |
| coyote | canis latrans | 1 | no_match |
| domestic sheep | ovis aries | 1 | no_match |

## colobus_species  (7 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 4 | class |
| domestic goat | capra aegagrus hircus | 2 | class |
| wild boar | sus scrofa | 1 | class |
| human | homo sapiens | 1 | order |

## lycalopex_species  (7 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| golden mantled ground squirrel | callospermophilus lateralis | 2 | class |
| chinese goral | naemorhedus griseus | 1 | class |
| black-tailed jackrabbit | lepus californicus | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| domestic cat | felis catus | 1 | order |
| domestic cattle | bos taurus | 1 | class |

## sika_deer  (7 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 3 | order |
| greater kudu | tragelaphus strepsiceros | 1 | order |
| bushbuck | tragelaphus scriptus | 1 | order |
| domestic dog | canis familiaris | 1 | class |
| impala | aepyceros melampus | 1 | order |
| sitatunga | tragelaphus spekii | 1 | order |
| blue sheep | pseudois nayaur | 1 | order |

## baird's_tapir  (6 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 2 | class |
| domestic cattle | bos taurus | 2 | class |
| white-lipped peccary | tayassu pecari | 1 | class |
| sun bear | helarctos malayanus | 1 | class |

## bushbuck  (6 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 4 | order |
| european roe deer | capreolus capreolus | 1 | order |
| wild boar | sus scrofa | 1 | order |

## hares_and_jackrabbits_genus  (6 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 3 | class |
| arizona black-tailed prairie dog | cynomys ludovicianus | 1 | class |
| red fox | vulpes vulpes | 1 | class |
| puku | kobus vardonii | 1 | class |
| guenther's dik-dik | madoqua guentheri | 1 | class |

## rattus_genus  (6 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| rodent |  | 2 | order |
| eastern gray squirrel | sciurus carolinensis | 2 | order |
| greater hog badger | arctonyx collaris | 1 | class |
| capybara | hydrochoerus hydrochaeris | 1 | order |
| american mink | neovison vison | 1 | class |

## red_river_hog  (6 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 2 | order |
| domestic dog | canis familiaris | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | order |
| roosevelts' muntjac | muntiacus rooseveltorum | 1 | order |
| domestic horse | equus caballus | 1 | class |
| domestic goat | capra aegagrus hircus | 1 | order |

## south_american_coati  (5 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 2 | class |
| red fox | vulpes vulpes | 1 | order |
| domestic cat | felis catus | 1 | order |
| domestic goat | capra aegagrus hircus | 1 | class |

## tiger  (5 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| white-tailed deer | odocoileus virginianus | 1 | class |
| human | homo sapiens | 1 | class |
| plains zebra | equus quagga | 1 | class |
| domestic cattle | bos taurus | 1 | class |
| canis species | canis | 1 | order |

## grey_wolf  (5 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| brush-tailed rock wallaby | petrogale penicillata | 2 | class |
| domestic sheep | ovis aries | 1 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| bobcat | lynx rufus | 1 | order |
| eastern fox squirrel | sciurus niger | 1 | class |

## muskrat  (5 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| nutria | myocastor coypus | 3 | order |
| eastern fox squirrel | sciurus niger | 1 | order |
| eurasian red squirrel | sciurus vulgaris | 1 | order |

## sun_bear  (5 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 3 | class |
| domestic horse | equus caballus | 1 | class |
| stump-tailed macaque | macaca arctoides | 1 | class |

## striped_hyaena  (5 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 2 | class |
| domestic dog | canis familiaris | 1 | order |
| domestic goat | capra aegagrus hircus | 1 | class |
| domestic cat | felis catus | 1 | order |

## muntjac_genus  (4 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 2 | order |
| suni | nesotragus moschatus | 1 | order |
| wild boar | sus scrofa | 1 | order |

## sloth_bear  (4 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 2 | class |
| american bison | bison bison | 1 | class |
| hylobatidae family |  | 1 | class |

## asiatic_black_bear  (4 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| domestic cattle | bos taurus | 2 | class |
| domestic cat | felis catus | 1 | order |
| human | homo sapiens | 1 | class |

## grevy's_zebra  (4 rejected multi-animal images)

| Predicted as | Scientific | Count | Match level |
|---|---|---:|---|
| red deer | cervus elaphus | 2 | class |
| white-tailed deer | odocoileus virginianus | 1 | class |
| common warthog | phacochoerus africanus | 1 | class |

