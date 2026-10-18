"""Collect every image the v2 talk uses into thesis/presentation/images/.

Sources are the interim deck (via assets/interim/, see extract_assets.py), the repo's
data/ folder, and the manuscript figures. Copies are downscaled to at most 2,400 px on the
long side, which is enough for a 4K render. Swarovski-internal pictures (the device,
its display and controls) go to images/nda/. The whole folder is gitignored and can be
rebuilt with this script.

Run (from thesis/presentation/slides/, after extract_assets.py):
    uv run python collect_images.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "images"
INTERIM = HERE / "assets" / "interim"
OLD_IMAGES = HERE.parent / "interim-presentation-images"
DATA = REPO / "data"
GEN = REPO / "thesis/manuscript/figures/images/generator_comparison"
FIGS = REPO / "thesis/manuscript/figures/images"
WIKI = DATA / "wikimedia/images"
MAX_PX = 2400

# target path (relative to images/) -> source
SOURCES: dict[str, Path] = {
    # --- Swarovski-internal (NDA) ---
    "nda/ax_visio.jpg": INTERIM / "slide05_01.jpg",
    "nda/eyepiece_view.jpg": INTERIM / "slide05_02.jpg",
    "nda/overlay_display.png": INTERIM / "slide05_05.png",
    "nda/mode_dial.png": INTERIM / "slide06_01.png",
    # --- deck furniture ---
    "deck/inovex_logo.png": INTERIM / "layout_01.png",
    "deck/portrait.png": INTERIM / "slide01_03.png",
    "deck/shop_ax_visio.png": INTERIM / "slide02_01.png",
    "deck/icon_neural_net.png": INTERIM / "slide05_04.png",
    "deck/icon_image.png": INTERIM / "slide08_02.png",
    "deck/icon_chip.png": INTERIM / "slide08_04.png",
    "deck/icon_brain.png": INTERIM / "slide08_06.png",
    # --- use case ---
    "usecase/barn_swallow.png": INTERIM / "slide05_03.png",
    "usecase/speciesnet_blog.png": INTERIM / "slide06_02.png",
    "usecase/camera_trap_armadillo.jpg": INTERIM / "slide06_03.jpg",
    "usecase/ocelot.jpg": INTERIM / "slide07_03.jpg",
    "usecase/ocelot_crop.jpg": INTERIM / "slide07_04.jpg",
    "usecase/rhino.jpg": WIKI / "rhinoceros_family/Northern_White_Rhinoceros_Angalifu.jpg",
    # --- data ---
    "data/taxonomy_hierarchy.png": INTERIM / "slide13_01.png",
    "data/reject_tutankhamun_cheetah.jpg": WIKI / "cheetah/Opening_of_the_Mouth_-_Tutankhamun_and_Aja.jpg",
    "data/reject_typing_monkey.jpg": WIKI / "chimpanzee/Monkey-typing.jpg",
    "data/reject_tiger_mascot.jpg": WIKI / "tiger/Mike_VI_the_Tiger_(Louisiana_State_University_mascot).jpg",
    "data/reject_aardvark_mask.jpg": WIKI / "aardvark/Aardvark_mask_and_costume,_Cameroon_grassland,_wood_-_Naturhistorisches_Museum_Nürnberg_-_Nuremberg,_Germany_-_DSC03948.jpg",
    "data/reject_buffalo_range_map.png": WIKI / "african_buffalo/Syncerus_range_map.png",
    "data/reject_bodybuilder_chimp.jpg": WIKI / "chimpanzee/Bodybuilderchimp.jpg",
    "data/reject_boar_painting.jpg": WIKI / "wild_boar/Tennō_Yūryaku_detail.jpg",
    "data/reject_elephant_mosaic.jpg": WIKI / "african_elephant/14_Piazzale_delle_Corporazioni_Ostia_Antica_2006-09-08.jpg",
    # --- generators ---
    "generators/zebra_grevys_gemini.png": OLD_IMAGES / "Gemini_Generated_Grévy_s zebra.png",
    "generators/zebra_mountain_gemini.png": OLD_IMAGES / "Gemini_Generated_mountain zebra.png",
    "generators/zebra_plains_gemini.png": OLD_IMAGES / "Gemini_Generated_plains zebra.png",
    "generators/lion_real.jpg": GEN / "appendix/lion_real.jpg",
    "generators/lion_sd35-large.jpg": GEN / "appendix/lion_sd35-large.jpg",
    "generators/lion_sd35m.jpg": GEN / "appendix/lion_sd35m.jpg",
    "generators/lion_flux2-klein.jpg": GEN / "appendix/lion_flux2-klein-9b.jpg",
    "generators/lion_hidream-i1.jpg": GEN / "appendix/lion_hidream-i1.jpg",
    "generators/lion_realvisxl.jpg": GEN / "appendix/lion_realvisxl-lightning.jpg",
    "generators/lion_gemini.jpg": GEN / "appendix/lion_001_full_gemini.jpg",
    "generators/lion_gpt-image-2-low.jpg": HERE / "assets/extra/lion_gpt-image-2-low.jpg",  # from the GPU server
    "generators/prompt_compressed_sd35m.jpg": GEN / "appendix/lion_001_compressed_sd35m.jpg",
    "generators/prompt_maxlen256_sd35m.jpg": GEN / "appendix/lion_001_maxlen256_sd35m.jpg",
    "generators/prompt_maxlen512_flux2-klein.jpg": GEN / "appendix/lion_001_maxlen512_flux2-klein-9b.jpg",
    "generators/prompt_full_gemini.jpg": GEN / "appendix/lion_001_full_gemini.jpg",
    "generators/saiga_real.jpg": GEN / "appendix/saiga_real.jpg",
    "generators/saiga_sd35-large.jpg": GEN / "appendix/saiga_sd35-large.jpg",
    "generators/saiga_flux2-klein.jpg": GEN / "appendix/saiga_flux2-klein-9b.jpg",
    "generators/saiga_hidream-i1.jpg": GEN / "appendix/saiga_hidream-i1.jpg",
    # --- bands ---
    "bands/synthetic_test_plains_zebra.png": OLD_IMAGES / "t_plains_zebra_009.png",
    "bands/real_test_zebra_herd.jpg": DATA / "inaturalist/images/plains zebra/inat_527331480.jpg",
    "bands/synthetic_striped_hyaena.png": DATA / "synthetic/images/band_a/striped_hyaena/a_striped_hyaena_008.png",
    # --- models ---
    "models/asian_elephant.jpg": DATA / "inaturalist/images/asian elephant/inat_351716828.jpeg",
    "models/african_elephant.jpg": DATA / "inaturalist/images/african elephant/inat_544037112.jpg",
}

# Folders copied whole: target folder -> (source folder, glob)
FOLDERS = {
    "data/hedgehogs": (OLD_IMAGES / "Hedgehogs", "*.*"),
    "generators/binturong": (OLD_IMAGES / "Binturong", "*.png"),
}
BINTURONG_REAL = {f"generators/binturong/binturong_real_{i}.jpg": INTERIM / f"slide18_0{i}.jpg" for i in (1, 2, 3)}


def _copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"  missing: {dst.relative_to(OUT)}  <- {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if max(im.size) <= MAX_PX and src.suffix.lower() == dst.suffix.lower():
            shutil.copyfile(src, dst)
            return True
        im.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)
        if dst.suffix.lower() in (".jpg", ".jpeg"):
            im.convert("RGB").save(dst, quality=92)
        else:
            im.save(dst)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for rel, src in {**SOURCES, **BINTURONG_REAL}.items():
        ok += _copy(src, OUT / rel)
    for rel, (folder, pattern) in FOLDERS.items():
        for src in sorted(folder.glob(pattern)):
            ok += _copy(src, OUT / rel / src.name)
    # Mangabey: any synthetic training image of the class (file numbering may vary).
    mangabeys = sorted((DATA / "synthetic/images/band_a/mangabeys_genus").glob("*.png"))
    if mangabeys:
        ok += _copy(mangabeys[min(7, len(mangabeys) - 1)], OUT / "bands/synthetic_mangabeys.png")
    # The 96-image Grévy's zebra wall, pre-rendered (96 single files would be ~60 MB).
    grevys = sorted((OLD_IMAGES / "grevy_s_zebra").glob("*.png"))[:96]
    if grevys:
        import sys

        sys.path.insert(0, str(HERE))
        from theme import mosaic  # noqa: E402

        shutil.copyfile(mosaic(grevys, cols=12, tile_px=400), OUT / "generators/grevys_synthetic_wall.png")
        ok += 1
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{ok} images in {OUT} ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
