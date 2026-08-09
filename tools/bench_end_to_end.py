#!/usr/bin/env python3
"""Time the WHOLE engine, C against Rust, on one synthetic checkpoint.

The kernel benchmark times two matmuls. This times everything else too: the KDA
recurrence, gated MLA with a KV cache, the attention-residual stack, the router,
MXFP4 expert streaming through the LRU cache, and the per-step plumbing that
holds it together. That is the number a user actually waits on.

WHY A SYNTHETIC CHECKPOINT
    The released one is 1.56 TB. This generates a small model with the same layer
    mix and the same kernel shapes, at a hidden size where per-token arithmetic
    dominates process startup. Weights are random, because what is being compared
    is arithmetic throughput, not model quality.

    Both binaries read the SAME bytes, so the token ids must come out identical.
    This script asserts that before it reports any timing: a speed comparison
    between two engines that computed different things is worthless.

WHAT IS MEASURED
    Per-step seconds taken from the engine's own STEP table, so checkpoint load
    and index time are excluded. Step 0 is the prefill and is dropped; it does
    work proportional to the prompt, not to one token.

USAGE
    tools/gen_bench_model.py /tmp/k3-bench2k 2048
    tools/bench_end_to_end.py /tmp/k3-bench2k [out.csv]
"""

import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUST = HERE.parent / "target" / "release" / "k3"
CBIN = HERE.parent.parent / "kimi-k3-in-c" / "bin" / "k3"

# STEP TOKEN SECONDS CACHE_HIT READ_GB TOK/S. SECONDS is printed to two decimals, which
# at these step times is one significant figure; TOK/S carries three, so invert that.
STEP = re.compile(r"^\s*(\d+)\s+(\d+)\s+([0-9.]+)\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s*$")

THREADS = (1, 2, 4, 10)
REPS = 5
GEN = 12
NPROMPT = 16


def run(binary: Path, model: str, threads: int):
    """One decode run. Returns (per-step seconds, generated ids)."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), RAYON_NUM_THREADS=str(threads))
    ids = ",".join(str(i) for i in range(1, NPROMPT + 1))
    out = f"/tmp/bench_e2e_{binary.name}.json"
    p = subprocess.run(
        [
            str(binary),
            model,
            "--ids",
            ids,
            "--gen",
            str(GEN),
            "--incremental",
            "--trunk-gb",
            "0",
            "--cache-gb",
            "0.1",
            "--out",
            out,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if p.returncode != 0:
        raise SystemExit(f"{binary} failed:\n{p.stderr[-2000:]}")
    rows = [m for m in (STEP.match(line) for line in p.stdout.splitlines()) if m]
    if len(rows) != GEN:
        raise SystemExit(f"{binary}: {len(rows)} step rows, expected {GEN}")
    secs = [1.0 / float(m.group(4)) for m in rows]
    return secs, json.load(open(out))["generated_ids"]


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k3-bench2k"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    for b in (CBIN, RUST):
        if not b.exists():
            raise SystemExit(f"missing {b}; build it first")

    rows, seen = [], set()
    print(f"{'threads':>7} | {'C s/tok':>8} | {'Rust s/tok':>10} | {'speedup':>7}")
    print("-" * 44)
    for t in THREADS:
        c_reps, r_reps = [], []
        for _ in range(REPS):
            s, i = run(CBIN, model, t)
            c_reps.append(statistics.median(s[1:]))
            seen.add(tuple(i))
            s, i = run(RUST, model, t)
            r_reps.append(statistics.median(s[1:]))
            seen.add(tuple(i))
        # The whole comparison rests on this: same bytes in, same tokens out.
        if len(seen) != 1:
            raise SystemExit(f"token divergence at {t} threads: {seen}")
        c, r = statistics.median(c_reps), statistics.median(r_reps)
        rows.append(("C", t, c, min(c_reps), max(c_reps)))
        rows.append(("Rust", t, r, min(r_reps), max(r_reps)))
        print(f"{t:>7} | {c:>8.4f} | {r:>10.4f} | {c / r:>6.2f}x")
    print("-" * 44)
    print(f"identical token ids in every configuration: {list(seen.pop())}")

    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w") as f:
            f.write("lang,threads,s_per_token,s_min,s_max\n")
            for lang, t, med, lo, hi in rows:
                f.write(f"{lang},{t},{med:.6f},{lo:.6f},{hi:.6f}\n")
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
