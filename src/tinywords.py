"""TinyStories at the word level: a vocabulary of thousands of classes for the Born readout.

The same official split and stream as tinystories.py; only the token changes.
  tokens      a word (letters, with an apostrophe ending such as "don't"), or any other single non-space
              character (punctuation, digits), or a line break; other whitespace only separates. Case is kept.
              Curly quotes and apostrophes are first made straight.
  vocabulary  0 unused (absent), <unk> = 1, <eos> = 2, then the 8,189 most common tokens of the first 500,000 training
              stories; V = 8,192. On those stories it covers 99.85 % of the tokens.
  windows     129 tokens from a uniformly random start in the stream of every story followed by <eos>
There is no copy-position truth at this level yet.
The training file is encoded once (tens of minutes) to <data>/cache/words_eos.i16, two bytes per token.
"""
import collections
import json
import os
import re

import numpy as np

import stories
import tinystories
from tinystories import CACHE, TRAIN, _stories_of

UNK, EOS, C0 = 1, 2, 3
V = 8192
L = stories.L
N_VOCAB = 500_000
STREAM = os.path.join(CACHE, "words_eos.i16")
META = os.path.join(CACHE, "words_vocab.json")
_TOKEN = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\n|[^\sA-Za-z]")
_STRAIGHT = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_cache = {}


def tokens(s):
    return _TOKEN.findall(s.translate(_STRAIGHT))


def detokenize(toks):
    """Text from tokens: a space between words, none before closing punctuation, quotes paired as they come."""
    out, open_quote, glue = [], False, True                      # glue: no space before the next token
    for t in toks:
        if t == '"':
            out.append(('' if glue else ' ') + t if not open_quote else t)
            glue = not open_quote                                # after an opening quote, none
            open_quote = not open_quote
        elif t.startswith("\n") or t in ".,!?;:)":
            out.append(t)
            glue = t.startswith("\n")
            if glue:
                open_quote = False                               # a quote never runs past a line break or <eos>
        else:
            out.append(('' if glue else ' ') + t)
            glue = False
    return "".join(out)


def _build():
    os.makedirs(CACHE, exist_ok=True)
    count = collections.Counter()
    for k, s in enumerate(_stories_of(TRAIN)):
        if k == N_VOCAB:
            break
        count.update(tokens(s))
    words = [w for w, _ in count.most_common(V - C0)]
    index = {w: C0 + i for i, w in enumerate(words)}
    n = unk = 0
    with open(STREAM + ".part", "wb") as out:
        for s in _stories_of(TRAIN):
            ids = [index.get(t, UNK) for t in tokens(s)]
            ids.append(EOS)
            a = np.array(ids, np.int16)
            out.write(a.tobytes())
            n += len(a)
            unk += int((a == UNK).sum())
    os.replace(STREAM + ".part", STREAM)
    json.dump(dict(words=words, source=os.path.basename(TRAIN), size=os.path.getsize(TRAIN), tokens=n, unk=unk),
              open(META, "w"))


def vocab():
    """(words by id, index): id - C0 into words; <unk> and <eos> have no entry."""
    path = META if os.path.exists(META) else os.path.join(stories.ROOT, "results", "vocab_tinystories.json")   # shipped copy
    words = json.load(open(path))["words"]
    return words, {w: C0 + i for i, w in enumerate(words)}


def decode(ids, words):
    return detokenize(["<unk>" if i == UNK else "\n\n" if i == EOS else words[i - C0] for i in ids if i > 0])


def _load():
    if "train" not in _cache:
        if not os.path.exists(META):
            _build()
        meta = json.load(open(META))
        assert meta["size"] == os.path.getsize(TRAIN), "the training file changed since the cache was built"
        words, index = vocab()
        codes = np.memmap(STREAM, np.int16, "r")
        test = []
        for s in _stories_of(stories.PATH):
            test += [index.get(t, UNK) for t in tokens(s)] + [EOS]
        _cache["train"] = (codes, np.array(test, np.int64), words)
    return _cache["train"]


def get(n_train=3_000_000, n_test=2_000, seed=0):
    """As tasks.get: x/y [n, L]; train as int16 (train.py widens a batch)."""
    codes, test, words = _load()
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(codes) - L, n_train)
    w = np.empty((n_train, L + 1), np.int16)
    for lo in range(0, n_train, 1 << 18):
        hi = min(n_train, lo + (1 << 18))
        w[lo:hi] = codes[a[lo:hi, None] + np.arange(L + 1)]
    b = np.random.default_rng(seed + 1_000_003).integers(0, len(test) - L, n_test)
    wt = test[b[:, None] + np.arange(L + 1)]
    return dict(x_train=w[:, :L], y_train=w[:, 1:], x_test=wt[:, :L], y_test=wt[:, 1:], words=words, vocab=V)
