"""
Convert Wikipedia plain-text files (data/wikipedia/*.txt) to Markdown.

Conversions applied:
  == Title ==       →  ## Title
  === Title ===     →  ### Title
  ==== Title ====   →  #### Title
  * bullet text     →  - bullet text

Output: data/wikipedia/*.md  (same directory, .md extension)
"""
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "wikipedia"

_HEADER_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")
_BULLET_RE = re.compile(r"^\*\s*(.*)")


def convert(text: str) -> str:
    lines = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            level = len(m.group(1))   # 2 → ##, 3 → ###, 4 → ####
            lines.append("#" * level + " " + m.group(2))
            continue

        m = _BULLET_RE.match(line)
        if m:
            lines.append("- " + m.group(1))
            continue

        lines.append(line)

    return "\n".join(lines)


def main() -> None:
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {DATA_DIR}")
        return

    for src in txt_files:
        if src.name == "missing_pages.txt":
            continue
        text = src.read_text(encoding="utf-8")
        md_text = convert(text)
        dst = src.with_suffix(".md")
        dst.write_text(md_text, encoding="utf-8")

    converted = len(txt_files) - 1  # exclude missing_pages.txt
    print(f"Converted {converted} files → {DATA_DIR}/*.md")


if __name__ == "__main__":
    main()
