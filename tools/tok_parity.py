#!/usr/bin/env python
"""tok_parity.py - prove the C tokenizer matches the Python oracle, case by case.

WHY THIS EXISTS
    k3_tok.h populates the Tok struct from tiktoken.model directly, so the whole
    encode path (byte-level keying, rank-BPE merging, the Kimi \\p{Han} pre-tokenizer)
    is new wiring around vendored machinery. "It produced ids" proves nothing -- a
    mis-keyed vocab still produces ids, just one per byte. This compares against
    tiktoken itself on the cases that actually separate a correct tokenizer from a
    plausible one.

    Divergence is reported per case with both id lists, because a single differing id
    in the middle of a long prompt is invisible in aggregate counts.

usage:  tok_parity.py <path-to-test_tok.exe> [--verbose]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


sys.path.insert(0, HERE)
import tok as oracle  # noqa: E402  the reference Python tokenizer

# The tokenizer files are located by tok.find_files(), not by a second copy of the same
# search. Two copies of a path search drift, and when they drift this gate reports a
# tokenizer DIVERGENCE for what is actually a path disagreement between the two halves
# of the comparison, the most misleading failure this script can produce.
FILES = oracle.find_files()

# Each case names the property it probes, so a failure says what broke.
CASES = [
    ("ascii plain",          "Hello world"),
    ("ascii sentence",       "The quick brown fox jumps over the lazy dog."),
    ("leading space",        " leading"),
    ("trailing space",       "trailing "),
    ("double space",         "two  spaces"),
    ("tab",                  "a\tb"),
    ("newlines",             "line1\nline2\n\nline4"),
    ("crlf",                 "dos\r\nline"),
    ("many spaces",          "a" + " " * 17 + "b"),
    ("digits run",           "1234567890 42 007"),
    ("mixed alnum",          "abc123def456"),
    ("punctuation",          "!@#$%^&*()_+-=[]{}|;':\",./<>?"),
    ("contractions",         "don't can't it's I'm we'll they've he'd"),
    ("caps run",             "HTTP HTML JSON XML API"),
    ("camelCase",            "camelCaseIdentifierName"),
    ("snake_case",           "snake_case_identifier_name"),
    # Han is the K3-specific pre-tokenizer rule: a leading Han run splits, and Han is
    # excluded from the letter classes. This is the single most likely thing to break.
    ("han only",             "\u4f60\u597d\u4e16\u754c"),
    ("han + ascii",          "\u4f60\u597d world"),
    ("ascii + han",          "hello \u4f60\u597d"),
    ("han sentence",         "\u6211\u4eec\u5728\u6d4b\u8bd5\u5206\u8bcd\u5668\u3002"),
    ("han + digits",         "\u7b2c123\u9875"),
    ("japanese",             "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c"),
    ("korean",               "\uc548\ub155\ud558\uc138\uc694"),
    ("cyrillic",             "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"),
    ("arabic",               "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"),
    ("greek",                "\u0393\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5"),
    ("accents",              "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fcber"),
    ("emoji",                "\U0001f600\U0001f680\U0001f9e0"),
    ("emoji + text",         "ship it \U0001f680 now"),
    ("emoji zwj",            "\U0001f469\u200d\U0001f4bb"),
    ("code c",               "int main(void){ return 0; }"),
    ("code python",          "def f(x):\n    return x ** 2\n"),
    ("json",                 '{"key": [1, 2, {"n": null}], "b": true}'),
    ("url",                  "https://example.com/a/b?c=d&e=f#g"),
    ("path windows",         "C:\\Users\\dev\\project\\file.c"),
    ("markdown",             "# Title\n\n- item **bold** `code`\n"),
    ("repeated char",        "aaaaaaaaaaaaaaaaaaaa"),
    ("long word",            "supercalifragilisticexpialidocious"),
    ("mixed script",         "hello \u4f60\u597d \u041c\u0438\u0440 \U0001f600 123"),
    ("empty-ish",            " "),
    ("single char",          "x"),
    ("high bytes",           "\u00ff\u00fe\u00fd"),
    ("math symbols",         "\u2211 \u221e \u2260 \u2264 \u03c0 \u00d7 \u00f7"),
    ("quotes typographic",   "\u201cquoted\u201d and \u2018single\u2019"),
    ("dashes",               "em\u2014dash en\u2013dash hyphen-minus"),
]


def c_encode(exe: str, text: str) -> list[int]:
    """Always go through a file.

    argv is re-encoded at the process boundary (on Windows into the active code page),
    so a non-ASCII case sent as an argument reaches the C side as different bytes than
    the oracle saw. That produces confident, entirely bogus 'divergences'. Writing the
    UTF-8 bytes to a file and having C read them verbatim compares the same input.
    """
    fd, tmp = tempfile.mkstemp(prefix="k3_parity_", suffix=".txt")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        r = subprocess.run([exe, FILES, "encodefile", tmp],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"test_tok failed: {r.stderr.strip()}")
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        return [int(x) for x in line.split(",")] if line else []
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    exe = sys.argv[1]
    verbose = "--verbose" in sys.argv

    enc, _cfg, special = oracle.load()
    allowed = set(special)
    npass = nfail = 0
    failures = []

    for name, text in CASES:
        want = enc.encode(text, allowed_special=allowed)
        got = c_encode(exe, text)
        if want == got:
            npass += 1
            if verbose:
                print(f"  PASS  {name:22s} {len(got):3d} ids")
        else:
            nfail += 1
            failures.append((name, text, want, got))
            print(f"  FAIL  {name:22s} oracle={len(want)} c={len(got)}")
            print(f"        oracle: {want}")
            print(f"        c     : {got}")

    print()
    print(f"tokenizer parity: {npass}/{npass + nfail} cases match")
    if failures:
        print("\nFAILING CASES (first divergent index shown):")
        for name, text, want, got in failures:
            i = 0
            while i < min(len(want), len(got)) and want[i] == got[i]:
                i += 1
            print(f"  {name}: diverges at id index {i}  input={text!r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
