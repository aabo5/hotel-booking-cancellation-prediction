"""
build.py - Assembler and Verification Pipeline for the Academic Research Paper.
Assembles all sections into report/research_paper.md and audits structural integrity,
word count, table count, and figure links.
"""

import sys
from pathlib import Path

# Add project root and src/paper to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.paper.part1_intro import (
    get_section_1,
    get_section_2,
    get_section_3,
    get_section_4,
    get_section_5,
)
from src.paper.part2_data_stats import (
    get_section_6,
    get_section_7,
    get_section_8,
    get_section_9,
)
from src.paper.part3_methods_models import (
    get_section_10,
    get_section_11,
)
from src.paper.part4_experiments import (
    get_section_12,
    get_section_13,
    get_section_14,
    get_section_15,
    get_section_16,
    get_section_17,
)
from src.paper.part5_calibration_economics import (
    get_section_18,
    get_section_19,
)
from src.paper.part6_synthesis_conclusion import (
    get_section_20,
    get_section_21,
    get_section_22,
    get_section_23,
)

def build_paper():
    sections = [
        get_section_1(),
        get_section_2(),
        get_section_3(),
        get_section_4(),
        get_section_5(),
        get_section_6(),
        get_section_7(),
        get_section_8(),
        get_section_9(),
        get_section_10(),
        get_section_11(),
        get_section_12(),
        get_section_13(),
        get_section_14(),
        get_section_15(),
        get_section_16(),
        get_section_17(),
        get_section_18(),
        get_section_19(),
        get_section_20(),
        get_section_21(),
        get_section_22(),
        get_section_23(),
    ]

    full_paper = "\n\n".join(sections).strip() + "\n"

    output_path = project_root / "report" / "research_paper.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_paper)

    print(f"Successfully generated academic paper at: {output_path}")

    # Audit statistics
    words = len(full_paper.split())
    lines = len(full_paper.splitlines())
    chars = len(full_paper)
    
    # Check section completeness
    section_checks = [f"## Section {i}:" for i in range(1, 24)]
    missing_sections = [s for s in section_checks if s not in full_paper]

    # Check References
    has_references = "## References" in full_paper

    # Check figures count
    figure_count = full_paper.count("![")

    # Check table count
    table_count = full_paper.count("**Table ")

    print("\n" + "="*60)
    print("ACADEMIC RESEARCH PAPER AUDIT SUMMARY")
    print("="*60)
    print(f"Output File: {output_path}")
    print(f"Total Word Count: {words:,} words")
    print(f"Total Line Count: {lines:,} lines")
    print(f"Total Character Count: {chars:,} characters")
    print(f"All 23 Sections Present: {len(missing_sections) == 0}")
    if missing_sections:
        print(f"  Missing sections: {missing_sections}")
    print(f"References Section Present: {has_references}")
    print(f"Embedded Figures Count: {figure_count}")
    print(f"Embedded Tables Count: {table_count}")
    print("="*60)

    assert len(missing_sections) == 0, f"Missing sections: {missing_sections}"
    assert has_references, "Missing references section!"
    assert words >= 8000, f"Word count ({words}) below expected academic standard!"

if __name__ == "__main__":
    build_paper()
