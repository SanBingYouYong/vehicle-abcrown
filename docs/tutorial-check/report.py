"""Compare the transcripts written by check.py for Marabou and α,β-CROWN.

    python tools/tutorial-check/report.py > tools/tutorial-check/results/report.md
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import CASES, RESULTS  # noqa: E402

SOLVERS = ["marabou", "abcrown"]


def classify(line):
    if "(trivial)" in line:
        return "trivial"
    if "proved no counterexample" in line:
        return "verified"
    if "counterexample" in line:
        return "falsified"
    if "timed out" in line:
        return "timeout"
    return "errored"


def read(solver, cid):
    path = os.path.join(RESULTS, solver, cid + ".txt")
    if not os.path.exists(path):
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(path).read())
    verdicts = [classify(l.strip()) for l in text.splitlines() if l.strip().startswith("result:")]
    counts = {k: verdicts.count(k) for k in ("verified", "trivial", "falsified", "timeout", "errored")}
    error = ""
    if not verdicts or counts["errored"]:
        m = re.search(r"\n(Error[^\n]*(?:\n  [^\n]*){0,3})", text)
        error = re.sub(r"\s+", " ", m.group(1)).strip()[:160] if m else text.strip()[-160:]
    return verdicts, counts, error


def fmt(r):
    if r is None:
        return "not run"
    verdicts, c, error = r
    if not verdicts:
        return "error: " + error
    parts = [f"{c[k]} {k}" for k in ("verified", "trivial", "falsified", "timeout", "errored") if c[k]]
    return ", ".join(parts) + (f" ({error})" if c["errored"] else "")


def main():
    print("| Case | Expected | Marabou | α,β-CROWN | Agree |")
    print("|---|---|---|---|---|")
    for c in CASES:
        m, a = (read(s, c["id"]) for s in SOLVERS)
        if m and a and m[0] and a[0]:
            pairs = list(zip(m[0], a[0]))
            decided = [(x, y) for x, y in pairs if {x, y} <= {"verified", "falsified", "trivial"}]
            if len(m[0]) != len(a[0]):
                agree = "different numbers of results"
            elif any(x != y for x, y in decided):
                agree = "**NO**"
            elif len(decided) == len(pairs):
                agree = "yes"
            else:
                agree = f"yes on {len(decided)}/{len(pairs)} decided by both"
        else:
            agree = "n/a"
        print(f"| {c['id']} | {c['expected']} | {fmt(m)} | {fmt(a)} | {agree} |")


if __name__ == "__main__":
    main()
