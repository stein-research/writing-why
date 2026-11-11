# IBM hardware jobs

The record of every job the paper ran on IBM's **ibm_kingston** (Heron r2), so that every hardware number in the
paper can be checked from the raw counts, without an IBM account. Each folder is one job, named by a key over its
circuits, the backend and the shots; [`index.md`](index.md) lists them with their IBM job ids, times and quantum
seconds. Account details are removed.

## The six jobs

Only the attention overlaps ran on the device: every $\langle q_i|k_j\rangle$ that the word after the prompt needs,
in the 16-pass model (seed 0), read by a five-qubit Hadamard test, 2,000 shots per circuit. Block 1 needs every row
(block 2 reads every earlier position), each with its reference key; block 2 only the last row. Each block is one job,
and block 2's circuits are built from block 1's measured overlaps. Every job also holds 32 calibration circuits as
deep as the overlaps (Hadamard tests of random state pairs with known overlap), from which the block's visibility and
readout offset are fitted.

| folder | run | block | circuits sent | quantum s |
|---|---|---|---|---|
| [`934582274897890b`](934582274897890b/record.json) | "Lily went to the park . At the" (next word: park) | 1 | 208 | 113 |
| [`cef070e6ca396d82`](cef070e6ca396d82/record.json) | the same | 2 | 68 | 38 |
| [`803ffd8a304edae0`](803ffd8a304edae0/record.json) | "Sue had a red kite . She flew her" (kite) | 1 | 244 | 132 |
| [`6dd1df6dc82899da`](6dd1df6dc82899da/record.json) | the same | 2 | 72 | 40 |
| [`7c84fe1b495a6c4b`](7c84fe1b495a6c4b/record.json) | the same with "kite" deleted (favorite) | 1 | 48 | 28 |
| [`fbef784b8019c0a5`](fbef784b8019c0a5/record.json) | the same | 2 | 68 | 38 |

In all, 389 quantum seconds. A circuit already measured is never sent again; its counts are taken from the job that
measured it, and the job's purpose says so:

- the kite run's block 1 has 4 more circuits, the overlaps of "." at position 5 with itself, the same states as in the
  park run (in block 1 a word's query and key depend only on the word and its position); their counts are the park
  run's;
- with "kite" deleted, block 1's overlaps are the kite run's except the reference keys of the rows after the deleted
  word: 160 readings come from earlier jobs (156 from the kite job, measured about 45 minutes earlier at visibility 0.80, and 4 from the park job), and only the 16 changed
  readings and 32 calibrations were sent (visibility 0.73). The run uses the readings as measured, since block 2's
  circuits were built from them so.

The pass criteria were fixed before any job was sent, in [`../results/hw_word_criteria.json`](../results/hw_word_criteria.json),
committed with the script eleven minutes before the first job; what changed in the script before the later jobs is
in [`hw_word_committed_vs_as_run.diff`](hw_word_committed_vs_as_run.diff) (nothing in the analysis).

| run | criterion | exact | ibm_kingston | evidence with the device's row sums |
|---|---|---|---|---|
| park | the same word, evidence $r \ge 0.95$ | park 0.951 | park 0.948 (95 % shot-noise interval 0.913-0.962) | $r = 0.995$ (0.964-0.999) |
| kite | the same word, evidence $r \ge 0.95$ | kite 0.670 | kite 0.723 (0.555-0.812) | $r = 0.998$ (0.990-1.000) |
| kite deleted | not kite | favorite 0.135 | favorite 0.122 (0.073-0.178); kite in none of 200 redraws | answer only |

Made after the runs, from the same counts: with block 1's attention numerators and row sums from the device
(`--devnum`), 7 of 7 (park) and 7 of 8 (kite) earlier words' shares lie within their 95 % shot-noise intervals.

## What each folder holds

| file | |
|---|---|
| `record.json` | job id, backend, status, shots, times, quantum seconds, IBM's metrics, the qubits each circuit ran on, the command, the code version, the software versions, the model file and its sha256, and a note per circuit (block, head, row, key, Re or Im, or calibration and its ideal value) |
| `circuits.qpy`, `circuits.txt` | the circuits as built (QPY, and the text the folder's key is computed from) |
| `transpiled.qpy` | the circuits as sent, transpiled for ibm_kingston |
| `counts.json` | the raw counts of every circuit, in the order of `record.json` |
| `properties.json` | the backend's calibration when the job was sent (T1, T2, gate and readout errors) |

How to check the numbers from these files, or run the jobs again: [`../docs/check-and-repeat-ibm.md`](../docs/check-and-repeat-ibm.md).
