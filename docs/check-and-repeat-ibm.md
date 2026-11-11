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

All three runs met them.
