# Running Writing Why on your own computer

## Install

```bash
git clone https://github.com/stein-research/writing-why
cd writing-why
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Without a GPU, install PyTorch's CPU-only build first (`pip install torch --index-url
https://download.pytorch.org/whl/cpu`). Every command below runs from the repository's root. A CPU is enough for
sampling and for a few evidence vectors; training and the evaluations want a GPU.

## The notebooks

```bash
pip install jupyterlab
jupyter lab demo/demo.ipynb        # the paper's story and passage, their evidence, a prompt of your own
jupyter lab demo/train.ipynb       # train one of the paper's models (needs TinyStories in data/ and a GPU)
```

## Write a story, and rerun the paper's samples

The trained models and the vocabularies are in `results/`, so sampling needs no dataset:

```bash
python src/sample.py words_born-therm16p2_s0 --prompt "Once upon a time" --chars 250 --temp 0.7 --top-p 0.9 --n 5 --seed 0
python src/sample.py words_born-therm16p2_s0 --prompt "Once upon a time" --chars 250 --temp 0.7 --top-p 0.9 --n 5 --seed 0 --only 3
```

Each draw prints its token-id checksum (`token-id sha256`); the paper's appendix gives the command and the checksum of
every sample shown, so you can check you have the same story. The draws come from one random generator in turn, so
draw 3 needs draws 1 and 2 first (`--only` still makes them). On another GPU or CPU the last digits of the
probabilities may differ and, rarely, change a sampled word; the checksum tells you at once.

## Reproduce the paper

Download the datasets into `data/tinystories/` and `data/wikitext103/` first ([data/README.md](../data/README.md)).

| command | what it gives |
|---|---|
| `python src/train.py --model "born-x8x2:m4,t16,x1,r16,p2" --task tinywords --epochs 1 --lr 0.032 --batch 64 --n-train 1000000 --seed 0 --warmup 0.02` | the 16-pass model (Table 2); `t1`...`t8` for the pass-count ladder, no `t` for the plain readout, `,s1` for the softmax-limit readout; `born-x8x2:m4,x1,r16,p2,l1` for the linear-head control; `--model softmax-w96 --lr 0.02` with the same `--warmup 0.02` for the softmax transformer (its sweep: Table 13, Appendix G); `--task wikiwords` for WikiText-103 |
| `python src/fresh_test.py <tags>` | the TinyStories test scores of Table 2 and Fig. 4 (windows sharing no story with the validation windows) |
| `python src/word_evidence.py words_born-therm16p2_s0 words_softmax-w96_s0` | Table 3's Born rows (copy positions: flip@1, copied first, bottom-5, ceiling, gaps) |
| `python src/word_evidence.py words_born-p2r16_s0 words_softmax-w96_lr0.02w_s0 --suffix _softmax_lr0.02w` | the softmax baseline at 512 nodes (seed 1 with `_s1`); Table 3's softmax rows: add `--softmax-only --nodes 4096 --suffix _softmax_lr0.02w_n4096` |
| `python src/bew_checks.py words_born-therm16p2_s0 --double 10` | BEW checks: each share with D/2 and D nodes (node doubling), and F, F_q at the 300 copy positions (node doubling at the first 10) |
| `python src/npass_check.py` | Appendix A: the n-pass formula against the model, its distance to the softmax limit, the 1/n term, and the share of 16 copies that would agree (results/npass_check.json) |
| `python src/word_rank_agree.py words_born-therm16p2_s0 words_born-p2r16_s0` | the rank agreement of BEW and gradient times input |
| `python src/held_path.py words_born-therm16p2_s0 words_born-p2r16_s0` | the held function against the real probability along BEW's path (Table 4), and integrated gradients of the real probability (Table 3's "IG of the real $P$" rows) |
| `python src/error_budget.py` | the calibration circuits' scatter against shot noise (Appendix D, error budget), from `ibm_jobs/` |
| `python src/decisive.py` | how often the copy's deciding words decide for the model, the recency rows and the $\rho_\mathrm{LOO}$ column of Table 3 |
| `python src/attention_null.py` | the device prompts with the attention made uninformative: what the answer and the device-readings test can see (Section 4.5) |
| `python src/readout_limit.py words_born-therm16p2_s0` | the distance of the 16-pass readout from its softmax limit |
| `python src/time_evidence.py words_born-therm16p2_s0 words_born-p2r16_s0 words_softmax-w96_s0` | the cost per explained word |
| `python src/compile_optics.py words_born-therm16p2_s0 --bits 8` and `--photons 10 11 12` | the exact optical compile, the light, the SLM precision and shot noise (Table 5) |
| `python src/cell_check.py` | the multipass cell with a pixel-dependent loss and a misregistration between passes (Section 4.4) |
| `python src/mesh_check.py words_born-therm16p2_s0` | the Clements meshes and phase errors (Table 10, Appendix C) |
| `python src/hw_word.py --prompt "Lily went to the park . At the" --device ibm_kingston` | Section 4.5: the device run rebuilt from the recorded counts in `ibm_jobs/` (no account, nothing sent); more in [check-and-repeat-ibm.md](check-and-repeat-ibm.md) |
| `python src/fig_overview.py`, `fig_born_pair.py`, `fig_born_qubits.py`, `fig_ladder.py`, `fig_story.py`, `fig_samples.py` | the figures |

The scripts write into `results/` and `figures/`, replacing the records shipped with the paper when they share a
name; `git checkout results/` restores them.
