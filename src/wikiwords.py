"""WikiText-103 at the word level: the TinyStories word task (tinywords.py) on encyclopedic text,
a second domain for the paper's main comparison.

Same tokens, vocabulary size, window and <eos> as tinywords.py:
  text        the raw split (wikitext-103-raw-v1). Its tokenizer's marks are undone first (" @-@ " -> "-",
              " @,@ " -> ",", " @.@ " -> ".") and its split endings are joined again ("Story 's" -> "Story's",
              "don 't" -> "don't"), as tinywords.py reads them; the tokens of tinywords.py then split the text again, so the
              spaces it put around punctuation do not matter.
  articles    split at the top-level heading " = Title = "; heading lines (title and sections) are dropped, the
              paragraphs of an article are joined by a line break, <eos> after every article.
  vocabulary  0 unused, <unk> = 1, <eos> = 2, then the 8,189 most common tokens of the training articles; V = 8,192.
              Wikipedia's vocabulary is far larger than the stories', so <unk> is far more common (printed by
              `python wikiwords.py`).
  windows     129 tokens from a uniformly random start in the training stream; test windows from the official
              test split.
Data: <datasets>/wikitext103/*.parquet (README there: licence CC BY-SA and GFDL, sha256, revision). The training
split is encoded once to <datasets>/wikitext103/cache/words_eos.i16, two bytes per token.
"""
import collections
import glob
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq

import stories
from tinywords import C0, EOS, UNK, V, decode, detokenize, tokens  # noqa: F401  (same tokens and text)

L = stories.L
DATA = os.environ.get("WRITING_WHY_WIKITEXT", os.path.join(stories.ROOT, "data", "wikitext103"))
TRAIN = sorted(glob.glob(os.path.join(DATA, "train-*.parquet")))
TEST = os.path.join(DATA, "test-00000-of-00001.parquet")
CACHE = os.path.join(DATA, "cache")
STREAM = os.path.join(CACHE, "words_eos.i16")
META = os.path.join(CACHE, "words_vocab.json")
_TITLE = re.compile(r"^ = [^=].* = $")
_HEADING = re.compile(r"^ (= )+.* (= )+$")
_MARKS = ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."))
_CLITIC = re.compile(r" '(s|t|re|ve|m|d|ll)\b")                  # "Story 's" -> "Story's", "don 't" -> "don't"
_cache = {}


def _lines(files):
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=8192, columns=["text"]):
            yield from batch.column(0).to_pylist()


def _articles(files):
    """Each article's body: paragraphs joined by a line break, headings dropped."""
    body = []
    for s in _lines(files):
        s = s.rstrip("\n")
        if _TITLE.match(s):
            if body:
                yield "\n".join(body)
            body = []
        elif s.strip() and not _HEADING.match(s):
            for a, b in _MARKS:
                s = s.replace(a, b)
            body.append(_CLITIC.sub(r"'\1", s).strip())
    if body:
        yield "\n".join(body)


def _build():
    os.makedirs(CACHE, exist_ok=True)
    count = collections.Counter()
    for s in _articles(TRAIN):
        count.update(tokens(s))
    words = [w for w, _ in count.most_common(V - C0)]
    index = {w: C0 + i for i, w in enumerate(words)}
    n = unk = n_art = 0
    with open(STREAM + ".part", "wb") as out:
        for s in _articles(TRAIN):
            a = np.array([index.get(t, UNK) for t in tokens(s)] + [EOS], np.int16)
            out.write(a.tobytes())
            n, unk, n_art = n + len(a), unk + int((a == UNK).sum()), n_art + 1
    os.replace(STREAM + ".part", STREAM)
    json.dump(dict(words=words, sources=[os.path.basename(f) for f in TRAIN],
                   sizes=[os.path.getsize(f) for f in TRAIN], articles=n_art, tokens=n, unk=unk), open(META, "w"))


def vocab():
    """(words by id, index): id - C0 into words; <unk> and <eos> have no entry."""
    path = META if os.path.exists(META) else os.path.join(stories.ROOT, "results", "vocab_wikitext103.json")   # shipped copy
    words = json.load(open(path))["words"]
    return words, {w: C0 + i for i, w in enumerate(words)}


def _load():
    if "train" not in _cache:
        if not os.path.exists(META):
            _build()
        meta = json.load(open(META))
        assert meta["sizes"] == [os.path.getsize(f) for f in TRAIN], "the training files changed since the cache"
        words, index = vocab()
        codes = np.memmap(STREAM, np.int16, "r")
        test = []
        for s in _articles([TEST]):
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


if __name__ == "__main__":
    codes, test, words = _load()
    meta = json.load(open(META))
    print(f"train {meta['articles']} articles, {meta['tokens']} tokens, unk {meta['unk'] / meta['tokens']:.4%}; "
          f"test {len(test)} tokens, unk {(test == UNK).mean():.4%}")
    print("most common:", words[:30])
    print(decode(test[:200].tolist(), words))
