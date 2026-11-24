"""
verify.py - Audits the generated research paper.
"""
import re
from pathlib import Path

paper_path = Path("report/research_paper.md")
text = paper_path.read_text(encoding="utf-8")

figures = re.findall(r"!\[(.*?)\]\((.*?)\)", text)
print(f"Total figures referenced: {len(figures)}")
missing_figs = []
for alt, fig in figures:
    fig_path = (paper_path.parent / fig).resolve()
    if not fig_path.exists():
        missing_figs.append((fig, str(fig_path)))

if missing_figs:
    print("MISSING FIGURES:", missing_figs)
else:
    print("ALL FIGURES EXIST ON DISK!")

all_sections_found = True
for i in range(1, 24):
    header = f"## Section {i}:"
    found = header in text
    if not found:
        all_sections_found = False
        print(f"MISSING: Section {i}")

print(f"All 23 sections verified: {all_sections_found}")
print(f"References section verified: {'## References' in text}")

words = len(text.split())
print(f"Total word count: {words}")
