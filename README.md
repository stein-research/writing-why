# Writing Why: Born-Rule Language Models with Exact Word Evidence on Light and Qubits

Companion to the paper by Teerawat Tongloy (KMITL), Jiraporn Pongsopa and Rangsan Jomtarak (Suan Dusit
University).

| | |
|---|---|
| Write a story and see why each word was chosen | [demo/demo.ipynb](demo/demo.ipynb) |
| Train a model, then let it write | [demo/train.ipynb](demo/train.ipynb) |
| Run it on your own computer, and the paper's analyses | [docs/run-locally.md](docs/run-locally.md) |
| Check the IBM runs from their raw counts (no account), or repeat them | [docs/check-and-repeat-ibm.md](docs/check-and-repeat-ibm.md) |
| Paper | arXiv identifier to be added |
| Series | 1. Seeing Why, [arXiv:2611.01993](https://arxiv.org/abs/2611.01993) ([code](https://github.com/stein-research/seeing-why)), pictures<br>2. Reading Why, [arXiv:2611.05355](https://arxiv.org/abs/2611.05355) ([code](https://github.com/stein-research/reading-why)), sentence pairs<br>3. Writing Why (this repository), a model that writes |

![One written word, "park" after "Lily went to the park . At the": (a) on light, (b) on IBM's ibm_kingston](docs/overview.png)

*One written word, "park", (a) on light (simulated) and (b) on IBM's ibm_kingston with the same weights: the
device gives the same word, one of 8,192, and each earlier word's evidence follows the exact one.*

A small causal language model in which the light meets no nonlinearity but quadratic detection, and the next word is
drawn from a Born probability. Passing light $n$ times through one mask sharpens the readout towards a softmax, while
each word's probability stays a ratio of polynomials in the words' amplitudes, so the evidence for every word it
writes, the **Born evidence per word (BEW)**, is computed exactly. The same weights also run with their attention
read on a quantum computer.

## How it works

![The two Born language models, identical up to the readout](docs/born_pair.png)

*The two models, drawn as optics (simulated): the plain readout reads the readout state once, $|\psi|^2$; the
16-pass readout passes light through one mask 16 times, $|1+g\psi/16|^{32}$.*

The readout field $\psi_w$ over the 8,192 words is constructed as a rank-16 factorisation across a $64\times128$ grid
from the last block's measured outputs. The $n$-pass readout gives

```math
P_n(w) = \frac{|1+g\psi_w/n|^{2n}}{\sum_v |1+g\psi_v/n|^{2n}}, \qquad \lim_{n\to\infty} P_n(w) = \frac{e^{2g\,\mathrm{Re}\,\psi_w}}{\sum_v e^{2g\,\mathrm{Re}\,\psi_v}}.
```

A softmax only in the limit, a ratio of polynomials at every finite $n$: for real $x \ge -n$, $(1+x/n)^n$ is the
tempered exponential at $t=1-1/n$ (Amid et al., NeurIPS 2019). Reinhardt and Hauser (arXiv:2608.11173) reach the
softmax exactly with $L$ post-selected rounds; here only the limit is exponential. A causal Born-attention language
model with a plain readout was trained earlier in CNet (Crasmaru, arXiv:2610.08592, Sec. 11). On light no quantum
resource is used: "Born" names the $|\text{amplitude}|^2$ detection law.

With every normaliser held at its value for the actual input, the explained quantity $F(a)$, equal to $P(w)-1/V$ at
that input, is a polynomial of degree at most $D=400n$ in the word amplitudes. Its integrated gradients over the
earlier words,

```math
e_t = a_t \int_0^1 \frac{\partial F}{\partial a_t}(\alpha a_{\lt i},\, a_i)\,d\alpha, \qquad \sum_{t \lt i} e_t = P(w) - \tfrac{1}{V} - F_q,
```

are computed without quadrature error by $D/2$ Gauss–Legendre nodes ($F_q$: the current word alone, which stays at
its amplitude along the path); in double precision the gap is at most $5.2\times10^{-11}$.

![Why the 16-pass model wrote "park"](docs/story.png)

Why the model wrote "park": each earlier word's exact share, deeper red for a larger push; grey words came after.

## What the paper finds

| | 16 passes | plain readout | softmax transformer |
|---|---|---|---|
| TinyStories V2 next-word accuracy (3 seeds) | 49.0 +- 0.1 % | 43.0 +- 0.1 % | 51.2 +- 0.3 % |
| WikiText-103 (3 seeds) | 27.8 +- 0.2 % | | 30.2 +- 0.04 % |
| evidence sums to the explained score | to 5.2e-11 (no quadrature error) | to 1e-13 | integrated gradients, approximate |

- On TinyStories, under the setup tested, replacing the readout does not close the gap: the same body with the
  readout's softmax limit scores the same (49.0 %), with a linear softmax head less (48.1 %). The softmax transformer was tuned over
  nine settings.
- The exact evidence ranks words as gradient times input does: it adds exact accounting, not a better ranking.
- In simulation the weights compile exactly to an opto-electronic system (meshes, balanced detectors, modulators, a
  controller and the 16-pass mask); the paper lists what the simulation idealises.

### On a quantum computer

![The 16-pass model on qubits and light](docs/born_qubits.png)

The attention overlaps read by Hadamard tests on IBM's ibm_kingston, the rest as simulated light; pass criteria fixed
before any job was sent:

| prompt | exact | ibm_kingston (95 % shot-noise interval) | evidence with the device's row sums |
|---|---|---|---|
| Lily went to the park . At the | park 0.951 | park 0.948 (0.913-0.962) | r = 0.995 (0.964-0.999) |
| Sue had a red kite . She flew her | kite 0.670 | kite 0.723 (0.555-0.812) | r = 0.998 (0.990-1.000) |
| the same, "kite" deleted | favorite 0.135 | favorite 0.122; kite in none of 200 redraws | answer only |

Six jobs, 389 quantum seconds, all recorded in [`ibm_jobs/`](ibm_jobs/). A demonstration, not statistics, and no
quantum advantage is claimed; the paper also says what the evidence test and the deletion cannot show.

## Contents

| | |
|---|---|
| [`demo/`](demo/) | the two notebooks and their engine, [`writing_why.py`](demo/writing_why.py) |
| [`src/`](src/) | the models, the trainer, sampling, the exact evidence and the evaluations, the optical compile and its mesh check, and the scripts that draw the figures |
| [`results/`](results/) | the trained models (three seeds of each model of the paper, the pass-count ladder, the mesh fine-tune), the evaluation records and checks, the device runs and their plans, and the two vocabularies |
| [`data/`](data/) | where TinyStories V2 and WikiText-103 go (downloaded and checked, not stored) |
| [`docs/`](docs/) | [running it locally and reproducing the paper](docs/run-locally.md); [checking and repeating the IBM runs](docs/check-and-repeat-ibm.md); the figures of this page |
| [`ibm_jobs/`](ibm_jobs/) | the record of every job run on IBM's ibm_kingston: job id, times, calibration, qubits, circuits as built and as sent, raw counts (account details removed); see [`ibm_jobs/README.md`](ibm_jobs/README.md) |

## A century of the Born rule

> In 1926, Max Born gave the rule that a measurement finds an outcome with probability given by the squared modulus of
> its amplitude. Decades earlier, Ludwig Boltzmann and J. Willard Gibbs had developed the statistical foundations of
> the weight $e^{-E/(k_BT)}$. The same exponential normalisation appears today as softmax in the readout of most
> language models.
>
> In this work, the two meet. Passing light through a suitably chosen mask $n$ times gives weights
> $|1+g\psi/n|^{2n}$, which approach $e^{2g\mathrm{Re}\psi}$ through the limit $(1+z/n)^n\to e^z$, with $2g$
> playing the role of inverse temperature. At every finite $n$, the readout remains a Born probability, while its word
> evidence—with the normalisers held at their input values—admits exact finite quadrature. We complete this work in
> the rule’s hundredth year, in gratitude to Born, Boltzmann, Gibbs
> and the many others whose work this one rests on.
>
> If this work stands, it stands on their shoulders. What we add is small: another way to use Born’s rule, to write
> a word and show how each earlier word contributed to its predicted probability.
>
> *The authors, 10 November 2026*

## Acknowledgements

We acknowledge the use of IBM Quantum services for this work. The views expressed are those of the authors, and do
not reflect the official policy or position of IBM or the IBM Quantum team.

## Use of AI tools

An AI assistant (Claude, Anthropic) was used under the authors' direction to write and run simulation and analysis
code, to prepare and check the hardware jobs, to write the code that draws the figures, to search the literature and check the references against their
sources, and to draft and revise the text; the authors chose what to cite. The authors retained responsibility for
the study design, the scientific judgments and the claims, checked the reported results against the saved data and
the code, and take full responsibility for the content.

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)). TinyStories: CDLA-Sharing-1.0; WikiText-103: CC BY-SA and GFDL; the vocabularies
derived from them keep those licences (see [`data/README.md`](data/README.md)). Please cite the paper
([`CITATION.cff`](CITATION.cff)).
