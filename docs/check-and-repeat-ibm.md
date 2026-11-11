# Checking and repeating the IBM jobs

Every hardware number in the paper comes from six jobs on IBM's ibm_kingston, each recorded in
[`ibm_jobs/`](../ibm_jobs/): the circuits as built and as sent, the backend's calibration at the time and the raw
counts ([`ibm_jobs/README.md`](../ibm_jobs/README.md) says what each file holds and which job is which). This page
shows, from the least to the most work, how to check those numbers with no IBM account, how to run the same
circuits again with your own, and how the runs were planned.

Install as in [`run-locally.md`](run-locally.md) (the full `requirements.txt`, which includes Qiskit). Every command
runs from the root of the clone and uses the 16-pass model of the paper, `results/words_born-therm16p2_s0.pt`.

## 1. Check the numbers, no account

These read only the files in the repository and send nothing.

| in the paper | command | prints | time |
|---|---|---|---|
| quantum seconds of every job | `python src/ibm_jobs.py list` | the job list, rewritten into `ibm_jobs/index.md` | seconds |
| the park run (Table 5, Fig. 1b): park 0.948 against 0.951, evidence $r = 0.995$, visibility 0.78 / 0.80 | `python src/hw_word.py --prompt "Lily went to the park . At the" --device ibm_kingston` | the exact and the device's next words, each earlier word's evidence (exact / device), the fitted calibrations | half a minute |
| the kite run: kite 0.723 against 0.670, $r = 0.998$ | `python src/hw_word.py --prompt "Sue had a red kite . She flew her" --device ibm_kingston` | the same | half a minute |
| the deletion: favorite 0.122 against 0.135 | `python src/hw_word.py --prompt "Sue had a red kite . She flew her" --delete 4 --no-evidence --device ibm_kingston` | the answers | seconds |
| the 95 % shot-noise intervals (Section 4.5) | the same commands with `--bootstrap 100` (`--bootstrap 200` for the deletion) | the device's values, their intervals, the share of redraws that keep the word | about an hour a run with evidence |
| evidence from the device's readings (Table 9): park 7 of 7, kite 7 of 8 shares within shot noise | the park and kite commands with `--devnum 100` (`--devnum 0`: no redraws, half a minute) | each earlier word's share with block 1's attention numerators and row sums from the device, its 95 % interval, the completeness residual | about an hour a prompt |
| what the device-readings test would catch (Section 4.5): the plan's systematics 1.8 of 7 and 2.2 of 8 shares outside, shot noise only 0.85 and 1.1 | the park and kite commands with `--devpower 20` | per simulated draw, the exact shares outside the device's interval widths; the ideal simulator's | about forty minutes a prompt |
| the passes as post-selected rounds (Appendix B): $S_{16}=2.9\times10^{-4}$ | `python src/npass_quantum.py` | the rounds' success at n = 1 to 16, the copies' figure, the fixed-mask rounds | minutes, about 4 GB |

Without `--submit`, `hw_word.py` rebuilds every circuit from the model, finds each in `ibm_jobs/` by its key, takes
its counts from there and runs the pass: it logs in nowhere and sends nothing. Block 2's circuits depend on block 1's
measured overlaps, so they are found in the ledger only when block 1's counts are the recorded ones, which is the
check that the pass is the one that ran. The results are written to `results/hw_word_*.json`; the paper's are there
already.

The bootstrap redraws every reading as binomial counts of 2,000 shots around its measured value and runs the pass
again; block 2 keeps its measured circuits, in their order.

## 2. Run the same circuits again, with your own account

```bash
python src/hw_word.py --prompt "Lily went to the park . At the" --device ibm_kingston --account 0 --keys my_keys.txt --submit
```

`my_keys.txt` holds one IBM Quantum key per line (`<name> <key>`), outside the repository; `--account 0,1` puts
block 1 on the first line's account and block 2 on the second's. The ledger checks the account's quota before sending
and records the job before waiting for it, so a lost connection never sends a job twice. A circuit already in the
ledger is not sent again: to measure the same circuits anew, move `ibm_jobs/` aside first, or start from an empty
ledger. The calibration and the visibility will differ from day to day; the calibration circuits in each job measure
them.

## 3. How the runs were planned

Before any job was sent, three things were fixed (results/hw_word_criteria.json):

- the prompts: two whose next word is decided by an earlier word, one confident (park, 0.95) and one less so (kite,
  0.67), and a deletion that changes the answer (deleting "park" was not used: the exact model still answers park);
- a Monte Carlo of the device with the noise Reading Why measured on the same machine (visibility 0.77-0.82, a readout
  offset, each row's own error, shot noise): `python src/hw_word.py --prompt "..." --plan 30` (results in
  `results/hw_word_plan_*.json`);
- the criteria: the same next word, evidence $r \ge 0.95$, and with "kite" deleted an answer other than kite.

All three runs met them. The criteria could hardly fail (the plan expected the same word in every draw), so the
paper adds a test made after the runs from the same counts (`--devnum`, above). The criteria were committed with the
script (5d0a6f9, 04:43:31 UTC) eleven minutes before the first job; the park jobs ran that commit, the other four
ran it with uncommitted changes that did not touch the analysis:
[`ibm_jobs/hw_word_committed_vs_as_run.diff`](../ibm_jobs/hw_word_committed_vs_as_run.diff).
