# Tutorial check: Marabou vs α,β-CROWN

> These scripts were run from `tools/tutorial-check/` inside a checkout of
> [vehicle-lang/tutorial](https://github.com/vehicle-lang/tutorial), with this
> project at `tools/vehicle-abcrown/`. To re-run them, place them there.
> Paths in the transcripts refer to that layout.

Runs the `vehicle verify` commands of the
[Vehicle tutorial](https://vehicle-lang.github.io/tutorial/) (chapters 2–6, chapter
code, exercises and solutions) with Marabou and with the α,β-CROWN backend in
`tools/vehicle-abcrown`, and compares the verdicts.

```bash
python3 tools/tutorial-check/fetch_ch6.py            # Chapter 6 files (from the Zenodo artefact)
python3 tools/tutorial-check/check.py --solver marabou
python3 tools/tutorial-check/check.py --solver abcrown
python3 tools/tutorial-check/report.py > tools/tutorial-check/results/report.md
```

`check.py --only REGEX` runs a subset. Each query has a 300 s limit. The
transcripts are in `results/<solver>/`, and the comparison is in
`results/report.md`. The last full run (2026-10-05) took 1.1 h with Marabou
and 2.4 h with α,β-CROWN. Of the α,β-CROWN time, 1.7 h went on the 150 Fashion
MNIST images (Marabou: 0.35 h), mostly α,β-CROWN start-up, once per query.

## Results of the last run

The two solvers **never disagree** where both decide. That covers all 185
results decided by both, including every one of the 150 Fashion MNIST images
in chapters 3 and 4. The documented results are reproduced: chapter 3 2/2,
exercise 7 40/50, chapter 4 37/50 and 22/50, the chapter 4 ACAS Xu table, and
chapter 5.

Where only one solver decides:

| Case | Marabou | α,β-CROWN | Checked |
|---|---|---|---|
| ACAS Xu N1,7 property 6 | timeout | verified (75 s) | — |
| ACAS Xu N1,7 property 7 | timeout | falsified (6 s) | counterexample re-evaluated with onnxruntime: genuine |
| ACAS Xu N1,8 property 6 | timeout (tutorial: 900 s) | verified (65 s) | — |
| ACAS Xu N1,8 property 7 | timeout (tutorial: 1588 s) | falsified (46 s) | genuine (strong left minimal by 3.9e-5) |
| ACAS Xu N1,8 property 8 | timeout (tutorial: 1284 s) | verified (125 s) | — |
| ACAS Xu N1,7 property 8 | falsified | timeout | **Marabou's counterexample is a near-tie**: re-evaluated, scores 0 and 1 differ by 2e-8 and the property holds there |
| `repro/linear.vcl`, `reduceadd-linear.vcl` | falsified | timeout | **the property is true**: `f(x) = x0 + x1 >= 0` on the box. Marabou's counterexample `x = 0` gives `f = 0`, which is not `< 0` |
| `mnist-robustness-standard.vcl` | cannot run (two network applications) | δ = 5: 2/2 verified, δ = 1: timeout | — |

Errors that are not the solver's:

* **Vehicle cannot compile for either solver:** Iris `property0` (unbounded
  input), `hyperrec/spec2.vcl` (cannot infer `n`), and the Euclidean
  (non-linear) robustness and `repro` non-linear specs.
* **Chapter 6 `pk.vcl` with α,β-CROWN:** Vehicle's VNN-LIB compiler rejects
  `Ka != Ke`, even though this property involves only parameters. Writing it
  as `Ka < Ke or Ka > Ke` works: all properties verify with both solvers, and
  the Rocq exports from the two caches are identical apart from the cache path.

Also checked by hand:

* **Chapter 5 export:** `vehicle export` to Rocq, Agda, Isabelle and Imandra
  works from α,β-CROWN caches. The output is identical to that from Marabou
  caches, except for the embedded cache path.
* **`exists` properties:** with one network application, and with two linked
  ones that Marabou cannot run. Covered by the backend's tests.

Note: the website's Chapter 6 command passes `pk_mean.idx` for both
`meanScalingValues` and `standardDeviationValues`; the artefact also ships
`pk_std.idx`. Both variants verify.
