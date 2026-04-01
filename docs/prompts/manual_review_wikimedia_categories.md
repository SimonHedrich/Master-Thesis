# Prompt: Manual Review of Filtered Wikimedia Category Files

## Context

This is a Master's Thesis project on **wildlife animal species detection** for an embedded binocular device (AX Visio). We are building a training dataset for a lightweight YOLO-nano model that detects **non-bird mammals** in daylight color photographs.

The directory `reports/wikimedia_categories_filtered/` contains 225 `.txt` files — one per target species. Each file lists Wikimedia Commons categories (with file counts) that were automatically filtered to remove obviously unusable content (anatomy, artwork, distribution maps, stamps, taxidermy, dead animals, etc.).

**Your job is to go through every file and manually remove any remaining categories that should NOT be used for downloading training images.** Edit the files in-place.

---

## What makes a category UNUSABLE for training

Remove a category (and indent it out of existence) if it primarily contains:

### Hard removes
- **Artwork of any kind** missed by the filter: watercolors, oil paintings, comics, video game characters, book covers, postage labels, greeting cards, heraldic items, icons for apps/logos
- **Historical black-and-white photos** that are so degraded they'd hurt more than help (look for "historical photographs", "19th century", "pre-1900")
- **Anatomical close-ups** missed by filter: teeth, claws, paws, eyes, tails, ears as standalone categories
- **Non-animal subjects sharing the name**: e.g. a place, ship, aircraft, beer brand, building, person, or sport team named after the animal — these slipped through "things named after" cascades
- **Cultural/ceremonial content**: religious ceremonies, festivals, mascots, dance costumes, folklore characters
- **Single-organ or single-body-part categories** (e.g. "penis", "udder", "teats")
- **Tracks, scat, nests, burrows, dens, wallows, dams** — animal signs rather than photos of the animal
- **Microscopy, histology, proteins, cell biology** content
- **Only video files** (categories labeled "Videos of …" that weren't caught — these contain `.webm`/`.ogv` files, not still images). **Exception:** keep them if the category name suggests they include still frames alongside video.
- **Fur, hide, leather, ivory products and manufacturing**
- **Hunting scenes where the animal is trophy/dead**, poaching evidence, roadkill

### Judgment calls — use your discretion
- **Zoo / captivity** categories: generally **keep** — zoo photos are real photographs showing the full animal
- **"People with X"** categories: **keep** if the animal is the main subject and clearly visible; remove if it reads more like a human-activity photo (e.g. "Elephant rides", "Bear dancing")
- **Subspecies and location categories**: **keep** — geographic and subspecies diversity is valuable
- **Behavior categories** ("eating X", "drinking X", "mating X"): **keep** unless purely about biology not visible in a still image
- **Quantity categories** ("1 Acinonyx jubatus", "2 Acinonyx jubatus"): **keep** — useful for training on different counts
- **Albinism / color morphs**: **keep** — real photographs of unusual color variants
- **Hybrid categories**: **keep** if there are real photos (not reconstructions)
- **"Featured pictures"**, **"Quality images"**, **"Valued images"**: **always keep** — these are the highest-quality photographs on Wikimedia Commons

---

## How to edit the files

- **Delete the entire line** for any category you want to remove.
- When you remove a parent category, **also remove all its indented children** (they are indented with 2 additional spaces per level).
- Do NOT re-indent remaining lines — keep the original indentation structure of what remains.
- Do NOT add any new lines or comments.
- Keep the header line (`# species name | Scientific name`) at the top of each file.

---

## How to work efficiently

1. Process all 225 files in `reports/wikimedia_categories_filtered/`.
2. For ambiguous category names, err on the side of **keeping** — the download script will only fetch images and a human can reject bad images during annotation.
3. Categories with very few files (1–3) in otherwise clean subtrees: **keep** unless clearly off-topic.
4. If a file ends up with only the header + root category line and nothing else, that is fine — it means there are no usable subcategories.

---

## Priority species (check these carefully)

These species had large, complex category trees that are most likely to contain edge-case false-positives after automated filtering:

- `lion.txt` — "Things named after lions" cascade may have left orphaned civic/geographic entries
- `tiger.txt` — same issue; check for martial-arts / dojo references
- `wolf.txt` / `grey_wolf.txt` — check for "Wolf (surname)", bands, ships
- `domestic_dog.txt`, `domestic_cat.txt` — large files; check for fashion/product categories
- `domestic_cattle.txt` — check for cattle-breed heraldry and livestock show content
- `squirrel_family.txt` — large file (500+ kept lines); check for cartoon characters

---

## Output

Edit the files **in-place** in `reports/wikimedia_categories_filtered/`. No new files or directories needed.

When finished, print a short summary of how many lines you removed per file (approximate is fine).
