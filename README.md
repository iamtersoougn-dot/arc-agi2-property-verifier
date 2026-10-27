# ARC-AGI-2 Property Verifier

A calibrated property verifier for the NVARC ARC-AGI-2 solver. It is the code behind the ARC Prize 2026 Paper Track writeup **"Free Rules, Thin Gains: Where Verification Helps an ARC-AGI-2 Solver"** by Tersoo Uganden (THEAFFILMTISER Research).

After NVARC picks its two attempts, the verifier checks them against rules that hold on every training pair of the task. It uses only the 12 rules that were wrong about the true answer at most 1% of the time on the 1,000 public ARC-AGI-2 training tasks. An attempt that breaks a rule is swapped for NVARC's best candidate that breaks none. Attempts that pass are never touched.

## Results (115 public eval tasks)

| Base run | Outputs changed | Without verifier | With verifier | Gained | Lost |
| --- | --- | --- | --- | --- | --- |
| NVARC, harness seed 0 | 13 | 28.70% | 29.13% | 1 | 0 |
| NVARC, harness seed 1 | 16 | 26.67% | 27.10% | 1 | 0 |
| NVARC adaptive pooled (our Kaggle run) | 16 | 29.28% | 30.14% | 1 | 0 |

The verifier never removed a correct answer. It gains little because, in 12 of the 18 cases where NVARC ranked the truth too low, the wrong winner is a near-miss that agrees with the truth on 82–99% of cells. The writeup explains this in full.

Five eval tasks (4a21e3da, abc82100, b6f77b65, f560132c, faa9f03d) are excluded because Kaggle's copy has different test pairs from the public GitHub copy.

## Layout

| Path | What |
| --- | --- |
| `kaggle/verifier_cell.py` | The cell to paste as the **last** cell of an NVARC notebook. It reads `/kaggle/inference_outputs` and rewrites `submission.json`, keeping the original as `submission_nvarc_original.json`. |
| `analysis/props.py` | The 21 candidate properties and the induction logic. |
| `analysis/calibrate.py` | Measures each property's false rate on the training tasks and writes `reliability.json`. |
| `analysis/baseline.py` | Reproduces NVARC's selection score, oracle coverage and lost-in-selection counts. |
| `analysis/evaluate.py` | Re-ranks the harness candidates with the verifier at several thresholds. |
| `analysis/diagnose.py` | Near-miss, rule-pass and view-count analysis of the out-ranked outputs. |
| `analysis/test_kaggle_cell.py` | Runs the exact Kaggle cell on NVARC's real harness submissions. |
| `analysis/score_kaggle_run.py` | Scores our Kaggle run with and without the verifier. |
| `results/` | Outputs of every script, plus both Kaggle submission files. |
| `cover/` | The cover image and its source. |

## Reproduce

Everything runs on a laptop CPU in a few minutes. Python 3.10+ and numpy are required.

```bash
./fetch_data.sh                 # clones ARC-AGI-2 and the NVARC harness, extracts candidates into data/
cd analysis
python3 calibrate.py            # -> reliability.json (training tasks only)
python3 baseline.py
python3 evaluate.py
python3 diagnose.py
python3 test_kaggle_cell.py
python3 score_kaggle_run.py
```

## Credits

- The solver, model and selection score are entirely NVARC's (Ivan Sorokin and Jean-François Puget; [repository](https://github.com/arnuvt/NVARC)).
- The notebook lineage is "ARC AGI2 LB33.89 Minimal Perfpatch" (mikelou1) and "Adaptive pooled TTT passes + full candidate pools + hard coverage fallbacks" (Narisetti Chaitanya naidu).
- The candidate exports and the coverage-vs-selection method come from [arc2026-nvarc-harness](https://github.com/Actualjudgepresident/arc2026-nvarc-harness) (vladee).
- Data: [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2).

## Licence

Apache-2.0 (see `LICENSE`). This covers only the files in this repository. Upstream code, models and data keep their own licences.
