"""TinyStories on its official split, read the standard way.

Train on TinyStoriesV2-GPT4-train.txt (2.7 M stories), test on -valid.txt (27,630). stories.py (the validation
stories split 90 / 10, windows inside one story) stays as it was, so its runs reproduce. Here, as language models
are usually fed:
  stream      every story, stripped, followed by an end-of-story token <eos>, back to back
  windows     129 characters from a uniformly random start in the stream (so every character is equally likely,
              and a window may run over <eos> into the next story); x the first 128, y the next character at each
  vocabulary  the 60 most common characters of the training stories at 2..61, <eos> = 62, <unk> = 1, 0 unused
              (absent); V = 63
Copy-position truth as in stories.py (<eos> is not a letter, so it ends a word); its word-prefix statistics come from
the first 200,000 training stories (a dictionary over all of them does not fit).
The training file never sits in memory as text: a one-time pass (a few minutes) encodes it to
<data>/cache/train_eos.u8, one byte per character; windows are drawn from that memory map.
"""
import collections
import json
import os

import numpy as np

import stories
from stories import C0, L, N_CHAR, UNK

EOS = C0 + N_CHAR
V = EOS + 1
EOS_CHAR = "\x03"                                                # <eos> inside raw text; never in a story
DATA = os.path.dirname(stories.PATH)
TRAIN = os.path.join(DATA, "TinyStoriesV2-GPT4-train.txt")
CACHE = os.path.join(DATA, "cache")
STREAM = os.path.join(CACHE, "train_eos.u8")
SEP = "<|endoftext|>"
N_PREFIX = 200_000
_cache = {}


def _stories_of(path, size=1 << 26):
    """The stories of a file, stripped, empty ones skipped, read size characters at a time."""
    rest = ""
    with open(path, encoding="utf-8") as f:
        while True:
            block = f.read(size)
            parts = (rest + block).split(SEP)
            rest = parts.pop() if block else ""
            for s in parts:
                if s.strip():
                    yield s.strip()
            if not block:
                return


def _table(chars):
    table = np.full(0x110000, UNK, np.uint8)
    for i, c in enumerate(chars):
        table[ord(c)] = C0 + i
    table[ord(EOS_CHAR)] = EOS
    return table


def _build():
    os.makedirs(CACHE, exist_ok=True)
    count = collections.Counter()
    for s in _stories_of(TRAIN):
        count.update(s)
    chars = sorted(c for c, _ in count.most_common(N_CHAR))
    assert EOS_CHAR not in chars
    table = _table(chars)
    with open(STREAM + ".part", "wb") as out:
        for s in _stories_of(TRAIN):
            out.write(table[np.frombuffer((s + EOS_CHAR).encode("utf-32-le"), np.uint32)].tobytes())
    os.replace(STREAM + ".part", STREAM)
    json.dump(dict(chars=chars, source=os.path.basename(TRAIN), size=os.path.getsize(TRAIN),
                   unk=int(sum(n for c, n in count.items() if c not in chars))),
              open(os.path.join(CACHE, "vocab_eos.json"), "w"))


def _load():
    if "train" not in _cache:
        if not os.path.exists(os.path.join(CACHE, "vocab_eos.json")):
            _build()
        meta = json.load(open(os.path.join(CACHE, "vocab_eos.json")))
        assert meta["size"] == os.path.getsize(TRAIN), "the training file changed since the cache was built"
        chars = meta["chars"]
        codes = np.memmap(STREAM, np.uint8, "r")
        test = "".join(s + EOS_CHAR for s in _stories_of(stories.PATH))
        prefix = collections.defaultdict(collections.Counter)
        for k, s in enumerate(_stories_of(TRAIN)):
            if k == N_PREFIX:
                break
            for w in stories._words(s):
                for m in range(2, len(w) + 1):
                    prefix[w[:m]][w[m] if m < len(w) else ""] += 1
        _cache["train"] = (codes, test, chars, _table(chars), prefix)
    return _cache["train"]


def vocab():
    """(chars with <eos> last, table) from the cache alone: for sampling, without loading the data."""
    chars = json.load(open(os.path.join(CACHE, "vocab_eos.json")))["chars"]
    return chars + [EOS_CHAR], _table(chars)


def train_stories(n):
    """The first n training stories as text (for the word checks; the file is read only that far)."""
    out = []
    for s in _stories_of(TRAIN):
        if len(out) == n:
            break
        out.append(s)
    return out


def get(n_train=3_000_000, n_test=2_000, seed=0):
    """As stories.get. x_train / y_train are uint8 (3 M windows in int64 would not fit); train.py widens a batch."""
    codes, test, chars, table, prefix = _load()
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(codes) - L, n_train)
    w = np.empty((n_train, L + 1), np.uint8)
    for lo in range(0, n_train, 1 << 18):                          # gather in slices: the index array stays small
        hi = min(n_train, lo + (1 << 18))
        w[lo:hi] = codes[a[lo:hi, None] + np.arange(L + 1)]
    b = np.random.default_rng(seed + 1_000_003).integers(0, len(test) - L, n_test)
    raw = [test[i:i + L + 1] for i in b]
    t = table.astype(np.int64)
    wt = np.array([t[[ord(c) for c in r]] for r in raw])
    ste, gte = stories._copy_truth(raw, prefix)
    return dict(x_train=w[:, :L], y_train=w[:, 1:], x_test=wt[:, :L], y_test=wt[:, 1:], src_test=ste, gt_test=gte,
                copy_test=ste >= 0, raw_test=raw, chars=chars + [EOS_CHAR], vocab=V)
