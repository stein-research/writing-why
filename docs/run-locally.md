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
| `python src/train.py --model "born-x8x2:m4,t16,x1,r16,p2" --task tinywords --epochs 1 --lr 0.032 --batch 64 --n-train 1000000 --seed 0 --warmup 0.02` | the 16-copy model (Table 1); `t1`...`t8` for the pass-count ladder, no `t` for one copy, `,s1` for the softmax-limit readout; `--model softmax-w96 --lr 0.01` with no warm-up for the softmax transformer; `--task wikiwords` for WikiText-103 |
| `python src/fresh_test.py <tags>` | the TinyStories test scores of Table 1 and Fig. 3 (windows sharing no story with the validation windows) |
| `python src/word_evidence.py words_born-therm16p2_s0 words_softmax-w96_s0` | Table 2 (copy positions: flip@1, copied first, bottom-5, ceiling, gaps) |
| `python src/word_rank_agree.py words_born-therm16p2_s0 words_born-p2r16_s0` | the rank agreement of BEW and gradient times input |
| `python src/readout_limit.py words_born-therm16p2_s0` | the distance of the 16-copy readout from its softmax limit |
| `python src/time_evidence.py words_born-therm16p2_s0 words_born-p2r16_s0 words_softmax-w96_s0` | the cost per explained word |
| `python src/compile_optics.py words_born-therm16p2_s0 --bits 8` and `--photons 10 11 12` | the exact optical compile, the light, the SLM precision and shot noise (Table 4) |
| `python src/mesh_check.py words_born-therm16p2_s0` | the Clements meshes and phase errors (Table 3) |
| `python src/fig_born_pair.py`, `fig_ladder.py`, `fig_story.py`, `fig_samples.py` | the figures |

The scripts write into `results/` and `figures/`, replacing the records shipped with the paper when they share a
name; `git checkout results/` restores them.
