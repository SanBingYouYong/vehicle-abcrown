"""Run the tutorial's `vehicle verify` commands with Marabou and with the
α,β-CROWN backend (tools/vehicle-abcrown) and compare the verdicts.

    python tools/tutorial-check/check.py [--solver marabou|abcrown|both] [--only REGEX]

Writes one transcript per (solver, case) to tools/tutorial-check/results/ and
a summary to tools/tutorial-check/results/summary-<solver>.csv.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
VENV_BIN = os.path.join(ROOT, ".venv", "bin")
VEHICLE = os.path.join(VENV_BIN, "vehicle")
ABCROWN_SOLVER = os.path.join(ROOT, "tools", "vehicle-abcrown", "bin", "vehicle-abcrown")
RES = os.path.join(ROOT, "resources-by-dataset")

QUERY_TIMEOUT = 300   # per solver query, seconds
CALL_TIMEOUT = 3 * 3600  # per vehicle call


def p(*parts):
    return os.path.join(ROOT, *parts)


def case(cid, cwd, spec, networks, datasets=(), params=(), props=(), expected="", cache=False):
    return dict(id=cid, cwd=cwd, spec=spec, networks=networks, datasets=list(datasets),
                params=list(params), props=list(props), expected=expected, cache=cache)


CASES = []

# Chapter 2: ACAS Xu (chapter-2/chapter-code/batch.sh) and the Iris solution.
ACAS17 = os.path.join(RES, "acasXu", "acasXu_1_7.onnx")
for k in range(1, 11):
    CASES.append(case(f"ch2-acas17-p{k}", p("chapter-2", "chapter-code"), "acasXu.vcl",
                      {"acasXu": ACAS17}, props=[f"property{k}"],
                      expected="p3: falsified (website)" if k == 3 else ""))
for k in range(7):
    CASES.append(case(f"ch2-iris-p{k}", p("chapter-2", "solutions", "iris"),
                      "iris-possible-answers.vcl", {"iris": "iris_model.onnx"},
                      props=[f"property{k}"],
                      expected="spec error: input unbounded" if k == 0 else ""))

# Chapter 3: MNIST robustness, exercises 2, 4, 5, 6, the hyper-rectangle
# solution, and exercise 7 (Fashion MNIST).
MNIST = {"classifier": os.path.join(RES, "mnist", "mnist-classifier.onnx")}
T2 = [("trainingImages", os.path.join(RES, "mnist", "t2-images.idx")),
      ("trainingLabels", os.path.join(RES, "mnist", "t2-labels.idx"))]
CH3 = p("chapter-3", "solutions")
for eps in ("0.005", "0.01", "0.05"):
    CASES.append(case(f"ch3-mnist-robust-e{eps}", CH3, "mnist-robustness.vcl", MNIST, T2,
                      [("epsilon", eps), ("eta", "0"), ("delta", "1")], ["robust"],
                      expected="2/2 verified (website)" if eps == "0.005" else ""))
CASES.append(case("ch3-mnist-strong-e0.005", CH3, "mnist-robustness.vcl", MNIST, T2,
                  [("epsilon", "0.005"), ("eta", "0"), ("delta", "1")],
                  ["strongClassificationRobust"]))
CASES.append(case("ch3-mnist-scr-e0.005", CH3, "mnist-robustness-scr.vcl", MNIST, T2,
                  [("epsilon", "0.005")]))
for delta in ("1.0", "5.0"):
    CASES.append(case(f"ch3-mnist-standard-d{delta}", CH3, "mnist-robustness-standard.vcl", MNIST,
                      T2[:1], [("epsilon", "0.005"), ("delta", delta)],
                      expected="Marabou cannot: two network applications"))
FM = p("chapter-3", "exercises", "FMNIST")
FM50 = [("trainingImages", os.path.join(FM, "idxdata", "0-49Images.idx")),
        ("trainingLabels", os.path.join(FM, "idxdata", "0-49Labels.idx"))]
CASES.append(case("ch3-fmnist-euclidean", CH3, "fmnist-robustness-euclidean.vcl",
                  {"classifier": os.path.join(FM, "fashion1l32n.onnx")}, FM50, [("epsilon", "0.005")],
                  expected="non-linear (L2): Vehicle cannot compile"))
HR = p("chapter-3", "solutions", "hyperrec")
for spec in ("spec.vcl", "spec2.vcl"):
    CASES.append(case(f"ch3-hyperrec-{spec[:-4]}", HR, spec, {"classifier": "adv.onnx"},
                      [("inputs", "5inputs.idx")]))
RP = p("chapter-3", "solutions", "repro")
for spec, exp in (("linear", ""), ("reduceadd-linear", ""),
                  ("quadratic", "non-linear: Vehicle cannot compile"),
                  ("reduceadd", "Vehicle crashes (repro/README.md)")):
    CASES.append(case(f"ch3-repro-{spec}", RP, f"{spec}.vcl", {"f": "tiny.onnx"},
                      params=[("epsilon", "0.1")], expected=exp))
CASES.append(case("ch3-fmnist-ex7", p("chapter-3", "exercises"), "FMNIST/fashionRobustness-solution.vcl",
                  {"classifier": "FMNIST/fashion1l32n.onnx"},
                  [(n, os.path.relpath(f, p("chapter-3", "exercises"))) for n, f in FM50],
                  [("epsilon", "0.005")], expected="40/50 verified (expected-output-0-49.txt)"))

# Chapter 4: verification of the vanilla Fashion MNIST network, and the ten
# ACAS Xu properties (one-property specs) on N_{1,8}.
CAP = p("chapter-4", "chapter-code", "capucci-pdt")
CAP50 = [("trainingImages", "0-49Images.idx"), ("trainingLabels", "0-49Labels.idx")]
for eps, exp in (("0.005", "37/50"), ("0.02", "22/50")):
    CASES.append(case(f"ch4-fmnist-vanilla-e{eps}", CAP, "fashionRobustness-solution.vcl",
                      {"classifier": "vanilla_e100.onnx"}, CAP50, [("epsilon", eps)],
                      expected=f"{exp} verified (capucci-pdt/README.md)"))
ACAS2 = p("chapter-4", "solutions", "acas2")
ACAS18_EXPECTED = {1: "verified", 2: "verified", 3: "falsified", 4: "falsified", 5: "falsified",
                   6: "timeout (900 s)", 7: "timeout (1588 s)", 8: "timeout (1284 s)",
                   9: "falsified", 10: "verified"}
for k in range(1, 11):
    CASES.append(case(f"ch4-acas18-p{k}", ACAS2, f"specs-verify/property{k:02d}.vcl",
                      {"acasXu": "acasXu_1_8.onnx"},
                      expected=f"{ACAS18_EXPECTED[k]} (acas2/README.md)"))

# Chapter 5: the wind controller, with caches for export.
CH5 = p("chapter-5", "chapter-code")
CTRL = {"controller": os.path.join(CH5, "controller.onnx")}
CASES.append(case("ch5-wind-safe", CH5, "windController.vcl", CTRL, cache=True,
                  expected="verified (chapter-5/chapter-code/README.md)"))
S5 = p("chapter-5", "solutions")
CASES.append(case("ch5-records", S5, "windController-records-solution.vcl", CTRL, cache=True))
CASES.append(case("ch5-centred", S5, "windController-centred-stability-solution.vcl", CTRL,
                  cache=True))
for speed, exp in (("0.5", "verified"), ("0.1", "falsified, x = [-0.1, -0.3]")):
    CASES.append(case(f"ch5-param-s{speed}", S5, "windController-parameterised-stability-solution.vcl",
                      CTRL, params=[("maxDistance", "0.2"), ("maxSensorChange", "0.2"),
                                    ("maxSpeed", speed)], props=["sensorNearCenter"],
                      expected=f"{exp} (chapter-5/solutions/README.md)"))

# Chapter 6: Vancomycert (files from fetch_ch6.py). The website passes
# pk_mean.idx for both datasets; run that verbatim and with pk_std.idx.
CH6 = os.path.join(HERE, "ch6-data")
PK_PARAMS = [("Ka", "4.5"), ("Ke", "3.5"), ("Vd", "10"), ("C_safe", "30"), ("ttd", "2"),
             ("Ka_over", "0.3228"), ("Ka_under", "0.3227"), ("Ke_over", "0.415"),
             ("Ke_under", "0.4149"), ("eps", "0.001")]
for name, std in (("website", "pk_mean.idx"), ("std", "pk_std.idx")):
    CASES.append(case(f"ch6-pk-{name}", CH6, "pk.vcl", {"pk": "pk.onnx"},
                      [("meanScalingValues", "pk_mean.idx"), ("standardDeviationValues", std)],
                      PK_PARAMS, cache=True,
                      expected="all verified (website, Rocq lemmas validated)"))


def summarise(text):
    """Counts of verdicts in a `vehicle verify` transcript."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    counts = {"verified": 0, "falsified": 0, "timeout": 0, "errored": 0}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("result:"):
            continue
        if "proved no counterexample" in line:
            counts["verified"] += 1
        elif "counterexample" in line:
            counts["falsified"] += 1
        elif "timed out" in line:
            counts["timeout"] += 1
        else:
            counts["errored"] += 1
    error = ""
    if not any(counts.values()):
        m = re.search(r"(Error.*?)(\n\n|$)", text, re.S)
        error = (m.group(1) if m else text[-300:]).strip().replace("\n", " ")[:300]
    return counts, error


def command(c, solver, cache_dir):
    args = [VEHICLE, "verify", "-s", c["spec"]]
    for n, f in c["networks"].items():
        args += ["-n", f"{n}:{f}"]
    for n, f in c["datasets"]:
        args += ["-d", f"{n}:{f}"]
    for n, v in c["params"]:
        args += ["-p", f"{n}:{v}"]
    for prop in c["props"]:
        args += ["-y", prop]
    if solver == "marabou":
        args += ["-v", "Marabou", "-a", f"--timeout={QUERY_TIMEOUT}"]
    else:
        args += ["-v", ABCROWN_SOLVER, "-a", f"--timeout {QUERY_TIMEOUT}"]
    if c["cache"]:
        args += ["-c", cache_dir]
    return args


def run(c, solver):
    os.makedirs(os.path.join(RESULTS, solver), exist_ok=True)
    out_path = os.path.join(RESULTS, solver, c["id"] + ".txt")
    cache_dir = os.path.join(RESULTS, solver, c["id"] + "-cache")
    env = dict(os.environ)
    env["PATH"] = VENV_BIN + os.pathsep + env["PATH"]
    env.setdefault("ABCROWN_ROOT", os.path.expanduser("~/F21ZK/alpha-beta-CROWN"))
    args = command(c, solver, cache_dir)
    start = time.time()
    try:
        proc = subprocess.run(args, cwd=c["cwd"], env=env, capture_output=True, text=True,
                              timeout=CALL_TIMEOUT)
        text, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        text = (e.stdout or b"").decode() + (e.stderr or b"").decode() + "\n[harness] call timed out"
        code = "call-timeout"
    elapsed = time.time() - start
    with open(out_path, "w") as f:
        f.write("$ (cd " + c["cwd"] + " && " + " ".join(args) + ")\n\n" + text)
    counts, error = summarise(text)
    return dict(case=c["id"], solver=solver, exit=code, seconds=round(elapsed, 1),
                expected=c["expected"], error=error, **counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["marabou", "abcrown"], required=True)
    ap.add_argument("--only", default=".*")
    args = ap.parse_args()
    cases = [c for c in CASES if re.search(args.only, c["id"])]
    os.makedirs(RESULTS, exist_ok=True)
    summary = os.path.join(RESULTS, f"summary-{args.solver}.csv")
    fields = ["case", "solver", "verified", "falsified", "timeout", "errored", "exit",
              "seconds", "expected", "error"]
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for c in cases:
            row = run(c, args.solver)
            w.writerow(row)
            f.flush()
            print(f"{row['case']:28} {args.solver:8} v={row['verified']} f={row['falsified']} "
                  f"t={row['timeout']} e={row['errored']} {row['seconds']}s {row['error'][:80]}",
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())
