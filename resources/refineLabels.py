"""
Reduces SpeciesNet labels to relevant mammals; diurnal land mammals that are likely to be photographed.
e.g. most rodents are excluded, some missing species were added, some were grouped together.
"""

from pyprojroot import here
import os
from datetime import date

LABELS_INPUT_PATH = here() / "artifacts" / "always_crop_99710272_22x8_v12_epoch_00148.labels.txt"
ARTIFACTS_DIR = here() / "artifacts"
REFINED_LABELS_OUTPUT_FILE = date.today().isoformat() + "_SN_labels_refined_only.txt"

to_remove = [
    " order",  # lines that are just orders - there are 7 of these for mammals and they all exist in more specific labels as well
    " family",  # lines that are just families - reviewed, some added back in.
    # " genus",  # lines that are just genera -> there are none (I think there was a bug here because the species lines are actually genus)
    " species",  # Reviewed below. Most are present at a more specific level, some are added back in, some rodents are just left out.
    "rodentia;muridae;",  # rats/mice
    "chiroptera;",  # bats
    "rodentia;cricetidae;peromyscus;",  # deer mice
    "mammalia;dasyuromorphia;dasyuridae;antechinus;",  # australian mouse
    "rodentia;heteromyidae;",  # kangaroo rats/mice
    "rodentia;nesomyidae;",  # some african rats/mice
    "rodentia;echimyidae;proechimys;",  # spiny rats
    "rodentia;spalacidae;",  # mole rats
    "rodentia;cricetidae;neotoma;",  # woodrats
    "rodentia;cricetidae;tylomys;",  # climbing rats
    "rodentia;cricetidae;scapteromys;",  # swamp rats
    "rodentia;cricetidae;sigmodon;",  # cotton rats
    "mammalia;diprotodontia;potoroidae;",  # marsupial rats
    "afrosoricida",  # tenrecs and golden moles
    "rodentia;bathyergidae;",  # mole rats
    "rodentia;dipodidae;",  # hopping desert mice
    "rodentia;gliridae;",  # dormice
    "rodentia;aplodontiidae;",  # bamboo rats
    "mammalia;diprotodontia;phalangeridae;",  # possums - nocturnal
    "mammalia;dasyuromorphia;dasyuridae;sminthopsis",  # dunnarts - nocturnal, limited range
    "601cf098-9876-4912-84bb-0926834305e9;mammalia;carnivora;ursidae;ursus;u. arctos;grizzly bear",  # Remove subspecies of brown bear
    "4c88622d-efe4-42af-9a54-e3b7a76c3b85;mammalia;rodentia;nesomyidae;cricetomys;gambianus;gambian rat",
    "3764d217-d9e9-4f80-8d77-270474212d48;mammalia;rodentia;cricetidae;euryoryzomys;nitidus;elegant rice rat",
    "e30b280a-7cf3-43dc-b12e-7293acaaabc3;mammalia;rodentia;echimyidae;isothrix;barbarabrownae;barbara brown's brush-tailed rat",
    "52f43264-a987-40ea-a83d-45e0453d4958;mammalia;rodentia;cricetidae;microtus;arvalis;common vole",
    "19fd235e-d808-4e22-bfae-42491045b763;mammalia;perissodactyla;equidae;equus;ferus;przewalski's horse",  # rare and endangered horse
    "fc9bcf48-83b9-48db-8646-063440d6427b;mammalia;lagomorpha;leporidae;pronolagus;randensis;jameson's red rock hare",  # noctural
    "f2d233e3-80e3-433d-9687-e29ecc7a467a;mammalia;;;;;mammal",
    "37a6a0b2-66bc-4846-a4c0-d6acb15eb777;mammalia;primates;;;;primate",
    "eeeb5d26-2a47-4d01-a3de-10b33ec0aee4;mammalia;carnivora;;;;carnivorous mammal",
    "a27d3983-76a1-4d2d-b080-18747cd77dcb;mammalia;artiodactyla;;;;even-toed ungulate",
    "36b9abce-005b-439a-858f-1660e5027ccb;mammalia;chordata;canidae;canis;lupus dingo;dingo",  # fixed order
    "b530f01c-f560-48f8-9770-d7f277ebff9b;mammalia;cetartiodactyla;bovidae;redunca;fulvorufula;mountain reedbuck",  # combined
    "e780d2d3-8822-4c68-b031-1970885663c3;mammalia;cetartiodactyla;bovidae;redunca;redunca;bohor reedbuck",
    "d59da409-557c-46ad-9373-390ee218142f;mammalia;perissodactyla;rhinocerotidae;ceratotherium;simum;white rhinoceros",  # combined rhinos
    "e465aa4e-d6ac-46fe-b990-e4fac9fb08ed;mammalia;perissodactyla;rhinocerotidae;dicerorhinus;sumatrensis;sumatran rhinoceros",
    "fa3a7f4a-912a-4ff9-b82c-95b27f3d39fb;mammalia;perissodactyla;rhinocerotidae;diceros;bicornis;black rhinoceros",
    "b8fcd6ad-a58e-4d68-9661-0a7c4fe537fa;mammalia;rodentia;sciuridae;eutamias;sibiricus;siberian chipmunk",  # technically a sub genus of tamias
    "1db1c6e2-2ea9-45a6-ab69-a730133298eb;mammalia;rodentia;sciuridae;tamias;striatus;eastern chipmunk",  # technically a sub genus of tamias
    "d8a9a77c-c66e-41f6-9134-0ce7d4b7cb3c;mammalia;carnivora;mustelidae;martes;caurina;pacific marten",  # very similar to american marten
    "d620e25c-eed1-40f1-8101-6d3b33f3a723;mammalia;carnivora;felidae;felis;silvestris lybica;african wild cat",  # visually similar
    "mazama;chunyi",  # limited range
    "muntiacus;rooseveltorum",  # limited range
    "muntiacus;vuquangensis",  # limited range
    "catopuma;badia",  # limited range
    "prionailurus;planiceps",  # flat-headed cat, limited range
    "caracal;aurata",  # African golden cat, limited range
    "diplogale;hosei",  # limited range
    "cynogale;bennettii",  # limited range
    "chrotogale;owstoni",  # limited range
    "mydaus;marchei",
    "mydaus;javanensis",
    "spilogale;angustifrons",
    "mammalia;carnivora;felidae;leopardus;",  # visually similar - combined
    "mammalia;primates;cercopithecidae;presbytis;",  # visually similar - all combined
    "mammalia;primates;cercopithecidae;cercocebus;",  # mangabeys, limited ranges - all combined
    "mammalia;carnivora;mustelidae;melogale;",  # ferret badgers, very limited range, Melogale moschata kept
    "mammalia;carnivora;mustelidae;mustela;",  # many visually similar, some added back
    "mammalia;cetartiodactyla;cervidae;mazama;",  # visually similar, americana added back
    "mammalia;primates;atelidae;alouatta;",  # howler monkeys - all combined
    "mammalia;rodentia;dasyproctidae;dasyprocta",  # agoutis - all combined
    "mammalia;cetartiodactyla;cervidae;muntiacus;",  # muntjacs - all combined
    "mammalia;carnivora;mustelidae;arctonyx;"  # hog badgers - all combined
    "mammalia;rodentia;sciuridae;neotamias;",  # chipmunks - all combined
    "mammalia;primates;cercopithecidae;papio;",  # baboons - all combined
    "mammalia;carnivora;viverridae;genetta;",  # genets - all combined
    "mammalia;carnivora;herpestidae;",  # mongooses & meerkat - all combined
    "mammalia;lagomorpha;leporidae;lepus;",  # hares - all combined
    "mammalia;lagomorpha;leporidae;sylvilagus;",  # cottontail rabbits - all combined
    "mammalia;lagomorpha;ochotonidae;",  # pikas - all combined
    "mammalia;artiodactyla;camelidae;lama;",  # llamas and alpacas - all combined
    "mammalia;rodentia;hystricidae;",  # old world porcupines - all combined
    "mammalia;rodentia;castoridae;castor;",  # beavers - all combined
    "mammalia;eulipotyphla;erinaceidae;",  # hedgehogs - all combined
    "mammalia;didelphimorphia;didelphidae;",  # opossums - all combined
    "mammalia;primates;galagidae;",  # bush babies - nocturnal
    # Uncommon squirrels
    "ammospermophilus;harrisii",
    "ammospermophilus;leucurus",
    "callosciurus;caniceps",
    "callosciurus;finlaysonii",
    "callosciurus;notatus",
    "cynomys;leucurus",
    "dremomys;pernyi",
    "dremomys;rufigenis",
    "funisciurus;anerythrus",
    "funisciurus;carruthersi",
    "funisciurus;congicus",
    "funisciurus;lemniscatus",
    "funisciurus;leucogenys",
    "funisciurus;pyrropus",
    "heliosciurus;rufobrachium",
    "heliosciurus;ruwenzorii",
    "hylopetes;alboniger",
    "ictidomys;mexicanus",
    "lariscus;insignis",
    "menetes;berdmorei",
    "microsciurus;flaviventer",
    "notocitellus;adocetus",
    "notocitellus;annulatus",
    "otospermophilus;beecheyi",
    "otospermophilus;variegatus",
    "paraxerus;boehmi",
    "paraxerus;cepapi",
    "paraxerus;lucifer",
    "paraxerus;ochraceus",
    "paraxerus;palliatus",
    "paraxerus;vexillarius",
    "petaurista;petaurista",
    "protoxerus;stangeri",
    "ratufa;affinis",
    "ratufa;bicolor",
    "ratufa;indica",
    "rhinosciurus;laticaudatus",
    "sciurotamias;davidianus",
    "sciurus;aerti",
    "sciurus;aestuans",
    "sciurus;anomalus",
    "sciurus;arizonensis",
    "sciurus;aureogaster",
    "sciurus;colliaei",
    "sciurus;deppei",
    "sciurus;granatensis",
    "sciurus;ignitus",
    "sciurus;igniventris",
    "sciurus;nayaritensis",
    "sciurus;spadiceus",
    "sciurus;stramineus",
    "spermophilus;erythrogenys",
    "sundasciurus;hippurus",
    "sundasciurus;juvencus",
    "sundasciurus;philippinensis",
    "sundasciurus;tenuis",
    "tamiops;swinhoei",
    "trogopterus;xanthipes",
    "urocitellus;armatus",
    "urocitellus;beldingi",
    "urocitellus;columbianus",
    "urocitellus;richardsonii",
    "urocitellus;undulatus",
    "xerospermophilus;tereticaudus",
    "xerus;erythropus",
    "xerus;rutilus",
    "callospermophilus;saturatus",
    "marmota;caligata",
    "marmota;himalayana",
    "marmota;sibirica",
    "neotamias;amoenus",
    "neotamias;dorsalis",
    "neotamias;merriami",
    "neotamias;minimus",
    "neotamias;obscurus",
    "neotamias;quadrivittatus",
    "neotamias;senex",
    "neotamias;siskiyou",
    "neotamias;sonomae",
    "neotamias;speciosus",
    "neotamias;townsendii",
    "neotamias;umbrinus",
    "sciurus;aberti",
    # Other uncommon animals (according to LLM)
    "equus;africanus",
    "equus;mulus",
    "coendou;bicolor",
    "coendou;mexicanus",
    "coendou;quichua",
    "coendou;rufescens",
    "coendou;spinosus",
    "anathana;ellioti",
    "tupaia;dorsalis",
    "tupaia;montana",
    "tupaia;palawanensis",
    "tupaia;tana",
    "moschiola;indica",
    "moschiola;meminna",
    "tragulus;javanicus",
    "tragulus;napu",
    "cabassous;centralis",
    "hylaeamys;megacephalus",
    "lophiomys;imhausi",
    "neacomys;spinosus",
    "onychomys;leucogaster",
    "reithrodontomys;megalotis",
    "cavia;tschudii",
    "galea;spixii",
    "galidia;elegans",
    "galidictis;fasciata",
    "salanoia;concolor",
    "heterohyrax;brucei",
    "brachylagus;idahoensis",
    "bunolagus;monticularis",
    "nesolagus;netscheri",
    "nesolagus;timminsi",
    "pithecia;aequatorialis",
    "plecturocebus;brunneus",
    "plecturocebus;ornatus",
    "tapirus;pinchaque",
    "soricidae;blarina brevicauda",
    "soricidae;sylvisorex vulcanorum",
    "pseudocheirus;occidentalis",
    "tamandua;mexicana",
    "dasypus;hybridus",
    "dasypus;kappleri",
    "isoodon;macrourus",
    "isoodon;obesulus",
    "cuniculus;taczanowskii",
    "choloepus;didactylus",
    "prionodon;linsang",
    "prionodon;pardicolor",
    "moschus;berezovskii",
    "cheirogaleus;major",
    "petaurus;norfolcensis",
    "aotus;nigriceps",
    "aotus;vociferans",
    "propithecus;candidus",
    "perodicticus;ibeanus",
    "perodicticus;potto",
    "hylobates;muelleri",
    "symphalangus;syndactylus",
    "thryonomys;gregorianus",
    "thryonomys;swinderianus",
    "tayassu;pecari",
    "lepilemur;microdon",
    "lagidium;viscacia",
    "galeopterus;variegatus",
    "cyclopes;didactylus",
    "tarsius;bancanus",
    "dinomys;branickii",
    "laonastes;aenigmamus",
    "cercartetus;nanus",
]

to_add = [
    "896f8689-2dca-48f6-9798-23bfda7096ee;mammalia;primates;cercopithecidae;presbytis;;leaf monkeys genus",
    "0f487b30-975d-49bd-bd20-db3dfa0bd931;mammalia;primates;cercopithecidae;cercocebus;;mangabeys genus",
    "70038d11-882c-4c9a-8578-1782a68528c8;mammalia;carnivora;mustelidae;arctonyx;;hog badger genus",
    "a04c522f-8ca1-4c09-85b5-a0f29b8466a8;mammalia;cetartiodactyla;cervidae;muntiacus;;muntjac genus",
    "16698525-37b3-4726-ac28-fc1b0fb5ff9b;mammalia;primates;atelidae;alouatta;;howler monkey genus",
    "4e6eb99a-baf0-4235-a7bc-1dd7b741b7c3;mammalia;rodentia;dasyproctidae;dasyprocta;;agouti genus",
    "05afbf7e-f878-4534-b08f-c97a02435474;mammalia;carnivora;felidae;leopardus;;leopardus species",
    "93bd99ea-193d-4fec-ba55-e2dffd4621d8;mammalia;rodentia;sciuridae;tamias;;chipmunk genus",
    "00049ff0-2ffa-4d82-8cf3-c861fbbfa9d5;mammalia;rodentia;muridae;rattus;;rattus genus",
    "9880b662-dc21-453b-aa2f-dd97338f623b;mammalia;rodentia;muridae;;;muridae family",
    "e4d1e892-0e4b-475a-a8ac-b5c3502e0d55;mammalia;rodentia;sciuridae;;;squirrel family",
    "523439f4-dee3-4f41-afea-edc0c891ef9c;mammalia;rodentia;cricetidae;;;cricetidae family",
    "36b9abce-005b-439a-858f-1660e5027ccb;mammalia;carnivora;canidae;canis;lupus dingo;dingo",  # fixed order
    "34078f5f-1006-4a79-a62d-e5f0477abb53;mammalia;cetartiodactyla;bovidae;redunca;;reedbuck genus",
    "da1c91cc-61eb-4afb-962c-590995ca9beb;mammalia;perissodactyla;rhinocerotidae;;;rhinoceros family",
    "c58da361-68da-477b-b5e9-da668d18ec7b;mammalia;primates;cercopithecidae;papio;;baboon genus",
    "afa5f0f8-3a37-4381-97c9-d3832046ef59;mammalia;carnivora;viverridae;genetta;;genet genus",
    "8ff5f9db-dbd5-4ad3-a834-22ade546c36b;mammalia;carnivora;herpestidae;suricata;suricatta;meerkat",
    "ca79a481-7751-466f-83ac-7799efcd2aca;mammalia;carnivora;herpestidae;;;mongoose family",
    "9a5d6ef5-887d-4060-8ef8-b7e54a7303de;mammalia;lagomorpha;leporidae;lepus;;hares and jackrabbits genus",
    "cacc63d7-b949-4731-abce-a403ba76ee34;mammalia;lagomorpha;leporidae;sylvilagus;;cottontail rabbits genus",
    "ce9a5481-b3f7-4e42-8b8b-382f601fded0;mammalia;lagomorpha;leporidae;lepus;europaeus;european hare",  # add back in because comon
    "667a4650-a141-4c4e-844e-58cdeaeb4ae1;mammalia;lagomorpha;leporidae;sylvilagus;floridanus;eastern cottontail",  # add back in because common
    "9878f262-2131-4259-adb2-fcab40e8238c;mammalia;lagomorpha;ochotonidae;ochotona;;pikas genus",
    "71739522-c635-45d7-a8e9-6115b5615253;mammalia;cetartiodactyla;camelidae;lama;;llama genus",
    "43320a08-bf31-49a5-8213-f032311c5765;mammalia;rodentia;castoridae;castor;;beaver genus",
    "9f332b69-49c6-43a1-aa71-c8888bc58d29;mammalia;rodentia;hystricidae;;;old world porcupine family",
    "0a11c18f-e253-4651-bb1b-4c642e8a03a7;mammalia;eulipotyphla;erinaceidae;;;hedgehog family",
    "87be3a5c-e60a-4e7e-88c7-21544914d067;mammalia;didelphimorphia;didelphidae;;;opossum family",
    "497705f4-99c8-4560-a86d-e2e15905c19b;mammalia;primates;lemuridae;lemur;catta;ring-tailed lemur",  # only species in this genus
    "8fa3d497-ec50-40bd-a61b-b546d7c2ce48;mammalia;primates;pitheciidae;callicebus;;callicebus genus",  # Existed with "species", so were removed
    "58bbfcec-c72c-4e59-a980-214cd62f9bd5;mammalia;carnivora;pinniped;;;pinniped clade",  # seals (3 families combined, only "phocidae family" was present before)
    "696f42f4-7eb9-491c-a1fe-b1d5c78ebcba;mammalia;carnivora;mustelidae;mustela;erminea;stoat",  # more iconic weasels to distinguish from family
    "aadb2862-6e62-4e62-981a-d7c2b25dfbfc;mammalia;carnivora;mustelidae;mustela;nivalis;least weasel",
    "df4c64cf-306d-4bf2-aa86-9e9c8d81fd41;mammalia;carnivora;mustelidae;mustela;frenata;long-tailed weasel",
    "98254337-6ee9-4664-a471-4b1eb0b73c48;mammalia;carnivora;mustelidae;mustela;putorius;western polecat",
    "0dba69b2-dfd7-48af-8e64-e576cbda74c3;mammalia;cetartiodactyla;cervidae;mazama;americana;red brocket",
    "22976d14-d424-4f18-a67a-d8e1689cefcc;mammalia;carnivora;felidae;leopardus;pardalis;ocelot",
    # Some most common rodents (from SpeciesNet training data) added back in
    "d01f67fd-836b-4b25-89fc-2239e59f56b0;mammalia;rodentia;sciuridae;otospermophilus;beecheyi;california ground squirrel",
    "e8490543-5bea-46a7-a326-a02142d2e6e1;mammalia;rodentia;hystricidae;hystrix;brachyura;malayan porcupine",
    "33ada742-d8aa-4402-bc15-f7c2e7d37b5f;mammalia;rodentia;hystricidae;hystrix;indica;indian crested porcupine",
    "60d52271-6958-4cc9-be30-57c83d1dc79b;mammalia;rodentia;hystricidae;atherurus;macrourus;asiatic brush-tailed porcupine",
    "f7fb32b6-1531-44e9-a7e2-a3197edafdb9;mammalia;rodentia;sciuridae;ammospermophilus;leucurus;white-tailed antelope squirrel",
    "b8b9d6cf-88e7-42e8-8c94-669b7ab1486a;mammalia;rodentia;cricetidae;neotoma;cinerea;bushy-tailed woodrat",
    # Added because iconic: bontebok, Cape Clawless Otter, Spotted-necked Otter, striped polecat, cape grysbok
    "8b41b6c7-fb1e-4dc1-93b3-0b3d7fa6c9f5;mammalia;artiodactyla;bovidae;damaliscus;pygargus;bontebok",
    "b18a99c0-13f3-4b1c-8d79-492bf78b33a5;mammalia;carnivora;mustelidae;aonyx;capensis;cape clawless otter",
    "59ac2f2e-35c0-42da-bc8e-98c961f51c9c;mammalia;carnivora;mustelidae;hydrictis;maculicollis;spotted-necked otter",
    "e24e6f46-7b16-4c63-9b4e-7d39e8f08b49;mammalia;carnivora;mustelidae;ictonyx;striatus;striped polecat",
    "7d3baf05-cf29-4f6e-9bb7-ec73935a2c5b;mammalia;artiodactyla;bovidae;raphicerus;melanotis;cape grysbok",
    "d8a9a77c-c66e-41f6-9134-0ce7d4b7cb3c;mammalia;artiodactyla;bovidae;antilope;cervicapra;blackbuck",
    # LLM suggested (and reviewed) additions:
    "9c2c6a6e-7d1b-4e47-9c2a-5d7d5b3e2a11;mammalia;cetartiodactyla;bovidae;bos;grunniens;yak",
    "3f8a2b91-0e64-4a6c-b0e7-2a6c4e5d9f22;mammalia;cetartiodactyla;bovidae;saiga;tatarica;saiga",
    "7a51d4c8-3b6f-4b95-9c14-8c1d2e7f3a33;mammalia;cetartiodactyla;bovidae;kobus;kob;kob",
    "1e6b9f72-5c44-4f0a-8f35-6d2b9a1e4b44;mammalia;cetartiodactyla;bovidae;kobus;leche;lechwe",
    "c4d0a8e3-2f91-4c2b-a6d2-1f7e8b9c5d55;mammalia;cetartiodactyla;bovidae;gazella;subgutturosa;goitered gazelle",
    "5b7e3c19-8a22-4d6f-b3c1-9a4d2e6f7c66;mammalia;cetartiodactyla;bovidae;gazella;bennettii;chinkara",
    "8a3d7c11-2e4f-4b3a-9c8e-1d5f6a7b9011;mammalia;primates;cercopithecidae;macaca;fuscata;japanese macaque",
    "2f6b9e44-7c21-4a0e-b6d3-5a8c1e2d9022;mammalia;primates;cercopithecidae;macaca;sylvanus;barbary macaque",
    "c1d5e8a2-9b34-4f6a-8d1e-3c7b2a4f9033;mammalia;primates;cercopithecidae;cercopithecus;neglectus;de brazza's monkey",
    "7e2a4c90-5d61-4b8f-a3e7-6b1c9d2a9044;mammalia;primates;cercopithecidae;cercopithecus;diana;diana monkey",
    "4b8f1d63-3a72-4e95-b2c4-9e6a1f3d9055;mammalia;primates;cercopithecidae;colobus;polykomos;king colobus",
    "9d7c2e15-6f43-4a1b-9e5c-2b8f3d7a9066;mammalia;primates;cercopithecidae;piliocolobus;badius;western red colobus",
    "1b3f6e92-4a71-4c8d-9e35-2f7a1c6b8011;mammalia;carnivora;mustelidae;melogale;moschata;chinese ferret-badger",
    "7f5c2e63-1b48-4a9d-a6e1-8c3d5f7b8066;mammalia;carnivora;mustelidae;enhydra;lutris;sea otter",
    "3f2a7b91-1c4e-4d6f-9b2a-7e5c8d1f1011;mammalia;cetartiodactyla;cervidae;axis;porcinus;hog deer",
    "6b5d4c82-8f23-4a1e-9c7d-2b6e4f3a1022;mammalia;cetartiodactyla;cervidae;rucervus;duvaucelii;barasingha",
    "9e7c1f43-5d12-4b9a-8e2c-4f6b3a5d1033;mammalia;cetartiodactyla;cervidae;pudu;puda;southern pudu",
    "b1d2c3e4-5678-4abc-9def-1234567890ab;mammalia;cetartiodactyla;bovidae;bos;grunniens;yak",
    "c2e3f4a5-6789-4bcd-8efa-2345678901bc;mammalia;carnivora;phocidae;mirounga;;elephant seal",
    "d3f4a5b6-7890-4cde-9fab-3456789012cd;mammalia;carnivora;otariidae;;;eared seals",
    "e4a5b6c7-8901-4def-abcd-4567890123de;mammalia;carnivora;odobenidae;odobenus;rosmarus;walrus",
]


new_contents = []
with open(LABELS_INPUT_PATH, "r", encoding="utf-8") as infile:
    for _line in infile:
        parts = _line.strip().split(";")
        if len(parts) > 1 and parts[1].lower() == "mammalia":
            if not any(to_remove_item in _line for to_remove_item in to_remove):
                new_contents.append(_line)
    for add_line in to_add:
        new_contents.append(add_line + "\n")

with open(os.path.join(ARTIFACTS_DIR, REFINED_LABELS_OUTPUT_FILE), "w", encoding="utf-8") as outfile:
    outfile.writelines(new_contents)

# Only common name
# output_names_path = os.path.join(ARTIFACTS_DIR, str("names_" + REFINED_LABELS_OUTPUT_FILE))
# with open(os.path.join(ARTIFACTS_DIR, REFINED_LABELS_OUTPUT_FILE), "r", encoding="utf-8") as infile, open(output_names_path, "w", encoding="utf-8") as outfile:
#     outfile.writelines(_line.strip().split(";")[-1] + "\n" for _line in new_contents)

output_sci_names_path = os.path.join(ARTIFACTS_DIR, str("sci_names_" + REFINED_LABELS_OUTPUT_FILE))
with open(os.path.join(ARTIFACTS_DIR, REFINED_LABELS_OUTPUT_FILE), "r", encoding="utf-8") as infile, open(output_sci_names_path, "w", encoding="utf-8") as outfile:
    for _line in new_contents:
        parts = _line.strip().split(";")
        species = parts[-2]
        if species:
            sci_name = f"{parts[-3]} {species}"
        else:
            i = -3
            while not parts[i]:
                i -= 1
            sci_name = parts[i]

        outfile.write(sci_name + "\n")
