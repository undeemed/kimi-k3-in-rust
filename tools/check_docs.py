#!/usr/bin/env python3
"""Fail if the README drifts from the data it quotes.

Every performance number in the README is also in a CSV under `docs/data/`, and the
figures are rendered from those same CSVs. Four ways that has already gone wrong in this
repository, each of which this catches:

  - A speedup transcribed from already-rounded seconds. 0.0550 / 0.0480 is 1.15x, but the
    underlying 0.054969 / 0.048047 is 1.144x, and the README said 1.15x.
  - A table silently compared against the wrong CSV once a second table was added, which
    turned a real mismatch into four passing rows.
  - A figure deleted from `_plots.py` while the README still linked its PNG.
  - An em dash, which this project does not use.

Run from the repository root:  python3 tools/check_docs.py
"""

import csv
import os
import re
import sys

# Both README performance tables share a header line, so they are located by order of
# appearance. Registering them here means adding a third table without updating this
# list fails loudly rather than shifting the indices underneath the existing checks.
TABLES = [
    ("released-checkpoint table", "docs/data/real-checkpoint.csv", "s_per_token_min"),
    ("synthetic table", "docs/data/end-to-end.csv", "s_per_token"),
]

failed = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        failed.append(label)


def main() -> int:
    text = open("README.md").read()

    # GitHub keeps underscores in heading slugs and drops other punctuation.
    anchors = {
        re.sub(r"[^a-z0-9 _-]", "", h.lower()).replace(" ", "-")
        for h in re.findall(r"^#{2,4} (.+)$", text, re.M)
    }
    anchors |= set(re.findall(r'<a id="([^"]+)"', text))
    broken = sorted(
        {a for a in re.findall(r"\]\(#([^)]+)\)", text) if a not in anchors}
    )
    check(not broken, "internal anchors resolve", ", ".join(broken))

    refs = set(re.findall(r"docs/images/([A-Za-z0-9_\-]+\.png)", text))
    disk = {f for f in os.listdir("docs/images") if f.endswith(".png")}
    check(
        not refs - disk,
        "every referenced figure exists",
        ", ".join(sorted(refs - disk)),
    )
    check(not disk - refs, "no orphaned figures", ", ".join(sorted(disk - refs)))

    docs = ["README.md"] + [
        os.path.join(d, f)
        for d in ("docs/data", "docs/images", "docs/patches")
        if os.path.isdir(d)
        for f in os.listdir(d)
        if f.endswith(".md")
    ]
    dashed = [p for p in docs if "\u2014" in open(p).read()]
    check(not dashed, "no em dashes", ", ".join(dashed))

    blocks = re.findall(r"threads\s+C s/tok.*?\n```", text, re.S)
    check(
        len(blocks) == len(TABLES),
        "every s/tok table is registered in TABLES",
        f"README has {len(blocks)}, TABLES has {len(TABLES)}",
    )

    # The full-model result: the README's C/Rust/speedup figures must match the CSV.
    fm = open("docs/data/full-model.csv").readlines()[1].strip().split(",")
    c_s, r_s, sp = float(fm[4]), float(fm[5]), float(fm[6])
    want = [f"C         {c_s:.2f}", f"Rust      {r_s:.2f}", f"speedup: {sp:.2f}x"]
    missing = [w for w in want if w not in text]
    check(not missing, "full-model table matches full-model.csv", ", ".join(missing))

    for i, (label, path, col) in enumerate(TABLES):
        if i >= len(blocks):
            check(False, label, "table missing from README")
            continue
        rows = {
            (r["lang"], int(r["threads"])): float(r[col])
            for r in csv.DictReader(open(path))
        }
        bad, n = [], 0
        for line in blocks[i].splitlines():
            m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x", line)
            if not m:
                continue
            n += 1
            th, c, r, sp = int(m[1]), float(m[2]), float(m[3]), float(m[4])
            true_c, true_r = rows[("C", th)], rows[("Rust", th)]
            # The ratio must come from full precision, never from the rounded columns.
            if (
                round(true_c, 4) != c
                or round(true_r, 4) != r
                or abs(sp - true_c / true_r) >= 0.005
            ):
                bad.append(
                    f"{th}t (csv gives {true_c:.6f}/{true_r:.6f} = {true_c / true_r:.4f}x)"
                )
        check(
            n > 0 and not bad,
            f"{label} matches {os.path.basename(path)}",
            f"{n} rows" + ("; " + "; ".join(bad) if bad else ""),
        )

    print()
    if failed:
        print(f"FAILED {len(failed)} check(s): {', '.join(failed)}")
        return 1
    print("README is consistent with docs/data/ and docs/images/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
