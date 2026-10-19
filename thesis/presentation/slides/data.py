"""Numbers and asset paths for the final talk.

Every figure on a slide comes from here, and each one names its source in the thesis
(Chapter 4 table labels) or the repo's reports. Accuracy is mAP@[.5:.95] on the REAL
test set unless the name says otherwise (see results-slides-strategy.md).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from theme import HERE, REPO

INTERIM = HERE / "assets" / "interim"  # extracted from the pptx by extract_assets.py
IMAGES = HERE.parent / "interim-presentation-images"  # gitignored, local only
FIGS = REPO / "thesis" / "manuscript" / "figures" / "images"
GEN = FIGS / "generator_comparison"
DATA = REPO / "data"
REPORTS = REPO / "reports"
EVAL_YOLO26N = (
    REPO / "scripts/training/yolo26n/model_exports/yolo26n-bs32-20260910-212812/evaluation/evaluation_report.json"
)

# --- interim deck assets (by extracted name; see extract_assets.py) -------------
LOGO = INTERIM / "layout_01.png"
PORTRAIT = INTERIM / "slide01_03.png"
SHOP = INTERIM / "slide02_01.png"
AX_VISIO = INTERIM / "slide05_01.jpg"
EYEPIECE = INTERIM / "slide05_02.jpg"
SWALLOW = INTERIM / "slide05_03.png"
NN_ICON = INTERIM / "slide05_04.png"
DIAL = INTERIM / "slide04_02.png"
OCELOT = INTERIM / "slide07_03.jpg"  # 800x600 camera-trap frame
OCELOT_CROP = INTERIM / "slide07_04.jpg"
OCELOT_BBOX = (335, 83, 767, 450)  # MegaDetector box in OCELOT's pixel coordinates
ICON_IMAGE = INTERIM / "slide08_02.png"
ICON_CHIP = INTERIM / "slide08_04.png"
ICON_BRAIN = INTERIM / "slide08_06.png"
HIERARCHY = INTERIM / "slide13_01.png"
BINTURONG_REAL = [INTERIM / f"slide18_0{i}.jpg" for i in (1, 2, 3)]

HEDGEHOGS = sorted((IMAGES / "Hedgehogs").glob("*.*"))
BINTURONG = IMAGES / "Binturong"
GREVY_SYNTH = sorted((IMAGES / "grevy_s_zebra").glob("*.png"))
ZEBRAS_GEMINI = {
    "Grévy's zebra": IMAGES / "Gemini_Generated_Grévy_s zebra.png",
    "Mountain zebra": IMAGES / "Gemini_Generated_mountain zebra.png",
    "Plains zebra": IMAGES / "Gemini_Generated_plains zebra.png",
}
SYNTH_TEST_ZEBRA = IMAGES / "t_plains_zebra_009.png"  # balanced synthetic test set image

PROMPT = HERE / "assets" / "prompt_grevys_zebra.txt"

# --- repo images -----------------------------------------------------------------
REAL_HERD = DATA / "inaturalist/images/plains zebra/inat_527331480.jpg"  # zebras + wildebeest
ASIAN_ELEPHANT = DATA / "inaturalist/images/asian elephant/inat_351716828.jpeg"
AFRICAN_ELEPHANT = DATA / "inaturalist/images/african elephant/inat_544037112.jpg"
SAIGA_REAL = GEN / "appendix/saiga_real.jpg"
HYAENA_SYNTH = DATA / "synthetic/images/band_a/striped_hyaena/a_striped_hyaena_008.png"
MANGABEY_SYNTH_DIR = DATA / "synthetic/images/band_a/mangabeys_genus"
ZEBRA_REAL_FINAL = FIGS / "zebra_species/plains_zebra.jpg"


def mangabey_synth() -> Path:
    files = sorted(MANGABEY_SYNTH_DIR.glob("*.png"))
    return files[7] if len(files) > 7 else MANGABEY_SYNTH_DIR / "missing.png"


# Wikimedia images that MegaDetector rejected as "no animal" (data/wikimedia/filter_results.jsonl).
WIKI = DATA / "wikimedia/images"
REJECTS = [
    WIKI / "cheetah/Opening_of_the_Mouth_-_Tutankhamun_and_Aja.jpg",
    WIKI / "chimpanzee/Monkey-typing.jpg",
    WIKI / "tiger/Mike_VI_the_Tiger_(Louisiana_State_University_mascot).jpg",
    WIKI / "aardvark/Aardvark_mask_and_costume,_Cameroon_grassland,_wood_-_Naturhistorisches_Museum_Nürnberg_-_Nuremberg,_Germany_-_DSC03948.jpg",
    WIKI / "african_buffalo/Syncerus_range_map.png",
    WIKI / "chimpanzee/Bodybuilderchimp.jpg",
    WIKI / "wild_boar/Tennō_Yūryaku_detail.jpg",
    WIKI / "african_elephant/14_Piazzale_delle_Corporazioni_Ostia_Antica_2006-09-08.jpg",
]


def reject_paths() -> list[Path]:
    """Resolve the reject files tolerantly (long Wikimedia names may be stored truncated)."""
    out = []
    for p in REJECTS:
        if p.exists():
            out.append(p)
            continue
        hits = sorted(p.parent.glob(p.name[:40] + "*"))
        out.append(hits[0] if hits else p)
    return out


# Generator-comparison sample images: same prompt, one per generator.
LINEUP = {
    "Real photo": GEN / "appendix/lion_real.jpg",
    "SD 3.5 Large": GEN / "appendix/lion_sd35-large.jpg",
    "SD 3.5 Medium": GEN / "appendix/lion_sd35m.jpg",
    "FLUX.2 klein": GEN / "appendix/lion_flux2-klein-9b.jpg",
    "HiDream-I1": GEN / "appendix/lion_hidream-i1.jpg",
    "RealVisXL": GEN / "appendix/lion_realvisxl-lightning.jpg",
    "Gemini Flash": GEN / "appendix/lion_001_full_gemini.jpg",
    # Not in the repo: pull one lion from the gpt-image-2-low-full cell on the GPU server
    # (data/synthetic_model_comparison/train/) and save it under this name.
    "gpt-image-2": HERE / "assets/extra/lion_gpt-image-2-low.jpg",
}
SAIGA_SYNTH = {
    "SD 3.5 Large": GEN / "appendix/saiga_sd35-large.jpg",
    "FLUX.2 klein": GEN / "appendix/saiga_flux2-klein-9b.jpg",
    "HiDream-I1": GEN / "appendix/saiga_hidream-i1.jpg",
}

# --- data: the 225 classes ---------------------------------------------------------
# Class counts per order, from the interim deck's class table (slide 15).
ORDERS = {
    "Carnivora": 75,
    "Cetartiodactyla": 65,
    "Primates": 24,
    "Rodentia": 24,
    "Other orders": 18,
    "Perissodactyla": 10,
    "Marsupials &\nmonotremes": 9,
}


# The 225 class names by order, as listed on the interim deck's class table (slide 15).
CLASS_TABLE = {
    "Carnivora": [
        "Giant panda", "Sun bear", "Sloth bear", "Spectacled bear", "American black bear",
        "Brown / grizzly bear", "Asiatic black bear", "Red panda", "Grey wolf", "Coyote",
        "Domestic dog", "Dingo", "Black-backed jackal", "Golden jackal", "Red fox",
        "Grey fox", "African wild dog", "Maned wolf", "Dhole", "Raccoon dog",
        "Bat-eared fox", "South American foxes", "Lion", "Leopard", "Tiger", "Jaguar",
        "Snow leopard", "Cheetah", "Puma / cougar", "Bobcat", "Canada lynx",
        "Eurasian lynx", "Domestic cat", "Wildcat", "Serval", "Caracal", "Clouded leopard",
        "Leopard cat", "Leopardus cats", "Ocelot", "Wolverine", "Honey badger",
        "American badger", "Eurasian badger", "N. American river otter", "Eurasian otter",
        "Sea otter", "Giant otter", "American mink", "Fisher", "Tayra", "Martens",
        "Weasels", "Hog badgers", "Spotted hyena", "Striped hyena", "Brown hyena",
        "Aardwolf", "Northern raccoon", "Ringtail", "White-nosed coati",
        "South American coati", "Kinkajou", "Mongooses", "Meerkat", "Striped skunk",
        "Spotted skunks", "Seals and sea lions", "Elephant seals", "Eared seals", "Walrus",
        "African civet", "Binturong", "Genets", "Fossa",
    ],
    "Cetartiodactyla": [
        "White-tailed deer", "Mule deer", "Elk / wapiti", "Red deer", "Sika deer", "Moose",
        "Reindeer / caribou", "European roe deer", "Common fallow deer",
        "Chital / spotted deer", "Water deer", "Sambar", "Muntjacs", "Red brocket",
        "Impala", "Blue wildebeest", "Black wildebeest", "Thomson's gazelle",
        "Grant's gazelle", "Springbok", "Gemsbok", "Gerenuk", "Hartebeest", "Waterbuck",
        "Kob", "Common eland", "Greater kudu", "Nyala", "Bushbuck", "Bongo",
        "Sable antelope", "Roan antelope", "Steenbok", "Kirk's dik-dik", "Common duiker",
        "Klipspringer", "Blesbok / bontebok", "Reedbucks", "Forest duikers",
        "Bighorn sheep", "Domestic sheep", "Mouflon", "Mountain goat", "Alpine ibex",
        "Domestic goat", "Northern chamois", "Blackbuck", "Nilgai", "Saiga",
        "African buffalo", "American bison", "European bison", "Domestic cattle", "Yak",
        "Domestic water buffalo", "Hippopotamus", "Giraffe", "Wild boar", "Domestic pig",
        "Common warthog", "Red river hog", "Collared peccary", "Dromedary camel",
        "Llamas and alpacas", "Pronghorn",
    ],
    "Perissodactyla": [
        "Domestic horse", "Domestic donkey", "Plains zebra", "Grevy's zebra",
        "Mountain zebra", "Asiatic wild ass", "Lowland tapir", "Malay tapir",
        "Baird's tapir", "Rhinoceroses",
    ],
    "Primates": [
        "Human", "Chimpanzee", "Bornean orangutan", "Vervet monkey", "Ring-tailed lemur",
        "Japanese macaque", "Drill", "Patas monkey", "Aye-aye", "Gorillas", "Baboons",
        "Howler monkeys", "Leaf monkeys / langurs", "Mangabeys", "Titi monkeys", "Macaques",
        "Guenons", "Colobus monkeys", "Spider monkeys", "Capuchins", "Tamarins",
        "True lemurs", "Squirrel monkeys", "Marmosets",
    ],
    "Rodentia": [
        "Eastern gray squirrel", "Eastern fox squirrel", "Western gray squirrel",
        "Eurasian red squirrel", "American red squirrel", "Golden-mantled ground squirrel",
        "California ground squirrel", "Groundhog / woodchuck", "Alpine marmot",
        "Yellow-bellied marmot", "Black-tailed prairie dog", "N. American porcupine",
        "Capybara", "Muskrat", "Nutria / coypu", "Chipmunks", "Flying squirrels", "Agoutis",
        "Beavers", "Rats", "Old World mice / rats", "Squirrel family (fallback)",
        "New World mice / voles", "Old World porcupines",
    ],
    "Marsupials and Monotremes": [
        "Red kangaroo", "Eastern grey kangaroo", "Swamp wallaby", "Red-necked wallaby",
        "Quokka", "Common wombat", "Koala", "Kangaroos / / wallabies",
        "Short-beaked echidna",
    ],
    "Other Orders": [
        "African elephant", "Asian elephant", "Giant anteater", "Brown-throated sloth",
        "Hoffmann's two-toed sloth", "Pangolins", "Nine-banded armadillo",
        "Giant armadillo", "European rabbit", "Hares / jackrabbits", "European hare",
        "Cottontail rabbits", "Eastern cottontail", "Pikas", "Aardvark", "Rock hyrax",
        "Hedgehogs", "Opossums",
    ],
}


def gbif_counts() -> list[int]:
    """Images per class in the initial GBIF corpus, sorted ascending (114 < 100, 30 = 0)."""
    with open(REPORTS / "gbif_image_counts_all.csv") as f:
        return sorted(int(r["image_count"]) for r in csv.DictReader(f))


def source_counts() -> list[tuple[int, int]]:
    """(GBIF, all sources) usable images per class after adding four sources; sorted by total."""
    with open(REPORTS / "coverage_analysis.csv") as f:
        rows = [(int(r["gbif_pass"]), int(r["total_pass"])) for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: r[1])


# Usable images per class and source (reports/coverage_analysis.csv), stacked bottom to top.
SOURCES = [
    ("GBIF", "gbif_pass"),
    ("iNaturalist", "inaturalist_pass"),
    ("Wikimedia", "wikimedia_pass"),
    ("OpenImages", "openimages_pass"),
    ("images.cv", "images_cv_pass"),
]


def source_stack() -> list[list[int]]:
    """Per class, usable images per source in SOURCES order; classes sorted by total."""
    with open(REPORTS / "coverage_analysis.csv") as f:
        rows = [[int(r[col]) for _, col in SOURCES] for r in csv.DictReader(f)]
    return sorted(rows, key=sum)


# --- data: generator comparison (tab:generator-ranking) ----------------------------
# (label, mAP mean, std, tier) — real test set, 12 classes, YOLO26n fine-tuned on 1,200 images.
GENERATORS = [
    ("gpt-image-2 low", 0.134, 0.014, "api"),
    ("gpt-image-2 medium", 0.126, 0.011, "api"),
    ("Gemini Flash", 0.105, 0.036, "pick"),
    ("Gemini Flash-Lite", 0.095, 0.006, "api"),
    ("SD 3.5 Large", 0.055, 0.002, "local"),
    ("SD 3.5 Medium", 0.054, 0.001, "local"),
    ("SD 3.5 Large Turbo", 0.043, 0.005, "local"),
    ("FLUX.2 klein", 0.037, 0.001, "local"),
    ("HiDream-I1", 0.029, 0.002, "local"),
    ("RealVisXL Lightning", 0.023, 0.001, "local"),
]
# Cost per image: API cells at the batch list price (tab:generator-api-cost, $ per 1,200
# images / 1,200); local cells as seconds per image on one NVIDIA A40 (tab:generator-throughput).
GENERATOR_COST = {
    "gpt-image-2 low": "$0.017",
    "gpt-image-2 medium": "$0.033",
    "Gemini Flash": "$0.033",
    "Gemini Flash-Lite": "$0.017",
    "SD 3.5 Large": "54 s",
    "SD 3.5 Medium": "24 s",
    "SD 3.5 Large Turbo": "25 s",
    "FLUX.2 klein": "31 s",
    "HiDream-I1": "144 s",
    "RealVisXL Lightning": "0.9 s",
}
# tab:prompt-length-ablation: full vs compressed prompt, same generator.
PROMPT_ABLATION = [("gpt-image-2 low", 0.134, 0.074), ("Gemini Flash-Lite", 0.095, 0.049)]
FULL_PROMPT_TOKENS = 2600  # ~1,300 words (scripts/synthetic_model_comparison/1e-generate_images_openai.py)
COMPRESSED_PROMPT_TOKENS = 55  # reports/model_comparison_compressed_prompt_metadata.jsonl
# sec:results_practicability_verdict, "A note on seeds".
SEED_STEPS = [0.064, 0.090, 0.105]
SEED_SIBLING = 0.095  # Gemini Flash-Lite, 3-seed mean


def generator_heatmap() -> tuple[list[str], list[str], list[list[float]]]:
    """Per-class AP for the 10 full/maxlen cells (rows: classes, rare first)."""
    cells = {
        ("gpt-image-2-low", "full"): "gpt-2 low",
        ("gpt-image-2-medium", "full"): "gpt-2 med",
        ("gemini-3.1-flash-image-preview", "full"): "Gemini",
        ("gemini-3.1-flash-lite-image", "full"): "Gem. Lite",
        ("sd35-large", "maxlen"): "SD3.5 L",
        ("sd35m", "maxlen"): "SD3.5 M",
        ("sd35-large-turbo", "maxlen"): "SD3.5 T",
        ("flux2-klein-9b", "maxlen"): "FLUX.2",
        ("hidream-i1", "maxlen"): "HiDream",
        ("realvisxl-lightning", "maxlen"): "RealVis",
    }
    ap: dict[tuple[str, str], float] = {}
    bands: dict[str, str] = {}
    with open(REPORTS / "model_comparison_full_per_class_ap.csv") as f:  # holds all 12 cells
        for r in csv.DictReader(f):
            k = (r["generator"], r["prompt_regime"])
            if k in cells:
                ap[(cells[k], r["class_name"])] = float(r["ap_mean"])
                bands[r["class_name"]] = r["band"]
    classes = sorted(bands, key=lambda c: (bands[c] != "A", bands[c], c))
    cols = list(cells.values())
    grid = [[ap.get((c, k), 0.0) for c in cols] for k in classes]
    return classes, cols, grid


# --- data: bands (tab:band-grid-real, tab:domain-shift-delta) -----------------------
BAND_CLASSES = {"A": 50, "B": 26, "C": 26, "D": 122}
BAND_REAL = {"A": 0.300, "B": 0.588, "C": 0.591, "D": 0.742}  # YOLO26n direct-FT, fine, real
BAND_SYNTH_TEST = {"A": 0.931, "B": 0.968, "C": 0.857, "D": 0.868}  # mean per-class AP
BAND_REAL_TEST = {"A": 0.202, "B": 0.506, "C": 0.539, "D": 0.664}
FIRST_RUN_BAND_A = 0.0  # first-generation model, before the loader fix (interim slide 28)


def per_class_real_ap() -> dict[str, list[tuple[str, float]]]:
    """Real-test AP per class, grouped by band (tier3_per_class of the YOLO26n report)."""
    rep = json.loads(EVAL_YOLO26N.read_text())
    out: dict[str, list[tuple[str, float]]] = {b: [] for b in "ABCD"}
    for c in rep["tier3_per_class"]:
        v = c["ap_real_fine"]
        if c.get("band") in out and v == v:  # skip NaN
            out[c["band"]].append((c["class_name"], v))
    return out


# --- data: models (tab:headline-cross-model, tab:embedded-latency) ------------------
# (label, real mAP, params M, end-to-end seconds per frame on the Raspberry Pi 400)
MODELS = [
    ("Ensemble, fine-tuned", 0.599, 193, 35.0),
    ("Ensemble, off the shelf", 0.445, 193, 35.0),
    ("YOLOv5s", 0.407, 7.63, 0.949),
    ("YOLO26n", 0.529, 2.71, 0.423),
]
DETECT_REAL = 0.782  # class-agnostic mAP, YOLO26n, real test (tier1.detect_analog.real_map)
FINE_REAL = 0.529
ELEPHANT_FWD, ELEPHANT_REV = 1545, 9  # Asian→African vs African→Asian (sec:results_lookalike_confusion)
# tab:kd-vs-direct-bands, real test: (distilled, direct)
KD_BANDS = {"A": (0.239, 0.300), "B": (0.565, 0.588), "C": (0.572, 0.591), "D": (0.706, 0.742)}
KD_ALL = (0.479, 0.529)


# =================================================================================
# v2 feedback round (talk_v2.py only; talk.py keeps the constants above)
# =================================================================================
def pct(v: float, d: int = 1) -> str:
    """mAP/AP as a percentage string: 0.529 -> '52.9 %'."""
    return f"{v * 100:.{d}f} %"


def source_stack_gbif_order() -> list[list[int]]:
    """Per class, usable images per source in SOURCES order; classes sorted by usable GBIF.

    The usable GBIF count (coverage_analysis.csv) is not a subset of the first GBIF corpus
    (gbif_image_counts_all.csv): 102 classes have more, from the later, wider GBIF pull.
    """
    with open(REPORTS / "coverage_analysis.csv") as f:
        rows = [[int(r[col]) for _, col in SOURCES] for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: r[0])


# tab:generator-ranking, compressed-prompt cells (≤ 75 tokens): (label, mAP, std, insert after)
GENERATORS_COMPRESSED = [
    ("gpt-image-2 low · compressed", 0.074, 0.002, "Gemini Flash-Lite"),
    ("Gemini Flash-Lite · compressed", 0.049, 0.003, "SD 3.5 Medium"),
]
# Prompt length per cell, in tokens. Full: ~5,900 tokens on average over the 12 comparison
# classes (Methods sec. 3.5); maxlen: T5 256 for SD 3.5, 512 for FLUX.2 klein and HiDream
# (GENERATOR_TIER in scripts/synthetic_model_comparison/1i-generate_images_local_maxlen.py);
# RealVisXL reuses the compressed prompts (77 CLIP tokens, 75 usable).
GENERATOR_PROMPT = {
    "gpt-image-2 low": "~5,900",
    "gpt-image-2 medium": "~5,900",
    "Gemini Flash": "~5,900",
    "Gemini Flash-Lite": "~5,900",
    "SD 3.5 Large": "≤ 256",
    "SD 3.5 Medium": "≤ 256",
    "SD 3.5 Large Turbo": "≤ 256",
    "FLUX.2 klein": "≤ 512",
    "HiDream-I1": "≤ 512",
    "RealVisXL Lightning": "≤ 75",
    "gpt-image-2 low · compressed": "≤ 75",
    "Gemini Flash-Lite · compressed": "≤ 75",
}

# tab:headline-cross-model, class-agnostic mAP_detect on the real test set (same order as MODELS)
MODELS_DETECT = {
    "Ensemble, fine-tuned": 0.964,
    "Ensemble, off the shelf": 0.947,
    "YOLOv5s": 0.750,
    "YOLO26n": 0.782,
}

# Look-alikes, YOLO26n direct-FT (EVAL_YOLO26N): real-test AP per zebra class (tier3_per_class).
ZEBRA_AP = {"Grévy's zebra": 0.575, "Mountain zebra": 0.805, "Plains zebra": 0.739}
# Compressed cells are not in tab:generator-api-cost; same output price, ~5,850 fewer input
# tokens, so at most the full-prompt batch price.
GENERATOR_COST_COMPRESSED = {"gpt-image-2 low · compressed": "≤ $0.017", "Gemini Flash-Lite · compressed": "≤ $0.017"}

# Per-species within-group confusion on the REAL test set, YOLO26n direct-FT: of the
# correctly localised detections of a true species that land in its look-alike group, the
# share named as the other species. Same greedy IoU-0.5 matching as
# eval_suite.scoring.within_group_confusion, tallied per true species from the run's
# predictions_real.json (GPU server). Counts: true species -> (predicted, n, matched).
# On the mixed set the same code reproduces the report (zebras 161/1,687, elephants 1,554/3,057).
SPECIES_CONFUSION_REAL = [
    ("Grévy's zebra", "Plains zebra", 112, 280),
    ("Grévy's zebra", "Mountain zebra", 21, 280),
    ("Mountain zebra", "Plains zebra", 12, 148),
    ("Asian elephant", "African elephant", 1545, 1959),
    ("African elephant", "Asian elephant", 2, 998),
]
# tab:kd-vs-direct, Setup A (distilled from the fine-tuned teacher), real test set
MODEL_DISTILLED = ("YOLO26n, distilled", 0.479, 2.71, 0.423)
MODEL_DISTILLED_DETECT = 0.745
# Training band per look-alike species (reports/class_split_counts.csv)
SPECIES_BAND = {"Grévy's zebra": "B", "Mountain zebra": "D", "Plains zebra": "D", "Asian elephant": "B", "African elephant": "D"}

# The twelve species of the generator comparison (reports/model_comparison_classes.csv,
# same order): image folder under data/synthetic_model_comparison/train/<cell>/full/images/
# and the name on the slide.
COMPARISON_CLASSES = [
    ("plains_zebra", "Plains zebra"),
    ("grevy_s_zebra", "Grévy's zebra"),
    ("mountain_zebra", "Mountain zebra"),
    ("red_fox", "Red fox"),
    ("american_black_bear", "American black bear"),
    ("lion", "Lion"),
    ("kinkajou", "Kinkajou"),
    ("water_deer", "Water deer"),
    ("ringtail", "Ringtail"),
    ("saiga", "Saiga"),
    ("aye_aye", "Aye-aye"),
    ("pangolin_family", "Pangolins"),
]
# Within-group confusion over all 30 look-alike groups, YOLO26n direct-FT (EVAL_YOLO26N's
# run), REAL test set only: confused / matched. eval_suite.scoring.within_group_confusion on
# the run's predictions_real.json (GPU server, 2026-10-19); on the mixed set the same call
# reproduces the report's 0.193 (tier2_within_group_confusion.overall_confusion_rate).
WITHIN_GROUP_CONFUSION_REAL = (5528, 28000)  # 19.7 %
# The same, per look-alike group (real test set): confused / matched.
WITHIN_GROUP_CONFUSION_REAL_GROUPS = {"zebras": (147, 1538), "elephants": (1547, 2957)}  # 9.6 %, 52.3 %
