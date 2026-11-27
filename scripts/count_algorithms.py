#!/usr/bin/env python3
"""Count SurtGIS algorithms from the curated registry and check it is complete.

`docs/algorithms.csv` lists one row per distinct computational operation,
with the public function that implements it. `docs/algorithms_excluded.csv`
lists every other public function of the scanned crates (variants of a
counted operation, utilities, vector geoprocessing) with the reason.

The counting rule is the one of the EMS paper's supplementary Table S1
(paper/supplementary_algorithms.tex): an operation that gives a different
output from the same input counts once, whatever its parameters or
interfaces; the 14 Florinsky curvatures, the spectral indices and the
focal and temporal statistics count one by one; I/O, data structures and
conversions do not count. Vector geoprocessing is reported apart.

The script fails (exit 1) when a registered function no longer exists, or
when a public top-level function of crates/algorithms or crates/relief is
in neither list, so a new operation has to be classified before CI passes.

Usage: python3 scripts/count_algorithms.py [--check]
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "docs" / "algorithms.csv"
EXCLUDED = ROOT / "docs" / "algorithms_excluded.csv"
# Crates whose public top-level functions must all be classified.
SCANNED = ("algorithms", "relief")


def public_fns(crate, top_level):
    """Names of `pub fn` in a crate; methods too unless `top_level`."""
    pattern = re.compile(r"^pub fn (\w+)" if top_level else r"^\s*pub fn (\w+)", re.M)
    names = set()
    for path in (ROOT / "crates" / crate / "src").rglob("*.rs"):
        names.update(pattern.findall(path.read_text(encoding="utf-8")))
    return names


def main():
    quiet = "--check" in sys.argv[1:]
    with REGISTRY.open(encoding="utf-8") as f:
        registry = list(csv.DictReader(f))
    with EXCLUDED.open(encoding="utf-8") as f:
        excluded = list(csv.DictReader(f))

    problems = []
    for row in registry:
        if row["function"] not in public_fns(row["crate"], top_level=False):
            problems.append(f"registered but missing: {row['crate']}::{row['function']} ({row['algorithm']})")
    classified = {(r["crate"], r["function"]) for r in registry + excluded}
    for crate in SCANNED:
        found = public_fns(crate, top_level=True)
        for name in sorted(found):
            if (crate, name) not in classified:
                problems.append(f"unclassified public fn: {crate}::{name} (add it to one of the two CSVs)")
        for r in excluded:
            if r["crate"] == crate and r["function"] not in found:
                problems.append(f"excluded but missing: {crate}::{r['function']}")
    ids = [r["id"] for r in registry]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids in docs/algorithms.csv")

    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)

    total = len(registry)
    vector = sum(1 for r in excluded if r["kind"] == "vector")
    if quiet:
        print(f"{total} algorithms (+{vector} vector operations), registry complete")
        return
    print(f"{total} algorithms, plus {vector} vector geoprocessing operations\n")
    print("By crate:")
    for crate, n in sorted(Counter(r["crate"] for r in registry).items(), key=lambda kv: -kv[1]):
        print(f"  {crate:<12} {n:>4}")
    print("\nBy category:")
    for cat, n in sorted(Counter(r["category"] for r in registry).items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<14} {n:>4}")
    print("\nBy source:")
    for src, n in sorted(Counter(r["source"] for r in registry).items()):
        print(f"  {src:<14} {n:>4}")
    kinds = Counter(r["kind"] for r in excluded)
    print("\nPublic functions not counted: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))


if __name__ == "__main__":
    main()
