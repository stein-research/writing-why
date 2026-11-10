"""TinyStories at the character level: the first real-text corpus.

Data: TinyStoriesV2-GPT4-valid.txt (27,630 stories by GPT-4, CDLA-Sharing-1.0), downloaded to
data/tinystories/ (data/README.md: download, sha256, licence); $WRITING_WHY_TINYSTORIES overrides.
  vocabulary  the 60 most common characters of the training stories (99.99 % of the text) at 2..61, every other
              character <unk> = 1, 0 unused (absent), as in tasks.py; V = 62
  split       by story, 90 / 10 (seed 0): no story is in both
  windows     129 characters from inside one story (random start): x the first 128, y the next character at each
              of the 128 positions, every position scored
Ground truth where the text has one: a copy position. Position t (current character x_t, predicting x_(t+1)) is one
when the current word's prefix p = the letters from the word's start up to x_t, |p| >= 2, also starts an earlier
word in the window, and
  - x_(t+1) is a letter (the word goes on; its end is no copy), every earlier word starting with p continues with
    the same letter, and that letter is x_(t+1) (the copy is unambiguous inside the context), and
  - language alone does not decide it: over the training stories' words, the probability of that character after p
    is below 0.5 (a word-prefix model would guess something else).
Deciding tokens: x_t, the earlier occurrence's last prefix character j and the copied character j + 1 (induction's A,
B and second A); the source is j + 1, taken from the latest earlier occurrence.
"""
import collections
import os

import numpy as np

UNK, C0, N_CHAR = 1, 2, 60
V = C0 + N_CHAR
L = 128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the repository
PATH = os.environ.get("WRITING_WHY_TINYSTORIES",
                      os.path.join(ROOT, "data", "tinystories", "TinyStoriesV2-GPT4-valid.txt"))
_cache = {}


def _stories():
    if "split" not in _cache:
        text = open(PATH, encoding="utf-8").read()
        st = [s.strip() for s in text.split("<|endoftext|>")]
        st = [s for s in st if len(s) > L]                          # a window must fit
        order = np.random.default_rng(0).permutation(len(st))
        n_test = len(st) // 10
        test = [st[i] for i in order[:n_test]]
        train = [st[i] for i in order[n_test:]]
        count = collections.Counter("".join(train))
        chars = sorted(c for c, _ in count.most_common(N_CHAR))
        table = np.full(0x110000, UNK, np.int8)
        for i, c in enumerate(chars):
            table[ord(c)] = C0 + i
        prefix = collections.defaultdict(collections.Counter)      # word prefix -> next character (or end, "")
        for s in train:
            for w in _words(s):
                for m in range(2, len(w) + 1):
                    prefix[w[:m]][w[m] if m < len(w) else ""] += 1
        _cache["split"] = (train, test, chars, table, prefix)
    return _cache["split"]


def _words(s):
    w = []
    for c in s:
        if c.isalpha():
            w.append(c)
        elif w:
            yield "".join(w)
            w = []
    if w:
        yield "".join(w)


def _windows(stories, table, rng, n):
    pick = rng.integers(0, len(stories), n)
    out = np.zeros((n, L + 1), np.int64)
    raw = []
    for r, k in enumerate(pick):
        s = stories[k]
        a = rng.integers(0, len(s) - L)
        w = s[a:a + L + 1]
        out[r] = table[[ord(c) for c in w]]
        raw.append(w)
    return out[:, :L], out[:, 1:], raw


def _copy_truth(raw, prefix):
    n = len(raw)
    src = np.full((n, L), -1)
    gt = np.zeros((n, L, L), bool)
    for r, w in enumerate(raw):
        starts = [i for i in range(L) if w[i].isalpha() and (i == 0 or not w[i - 1].isalpha())]
        start_of = np.full(L + 1, -1)
        for i in range(L):
            if w[i].isalpha():
                start_of[i] = i if (i == 0 or not w[i - 1].isalpha()) else start_of[i - 1]
        for t in range(L):
            b = start_of[t]
            if b < 0 or b == 0 or t - b + 1 < 2:                   # a word cut by the window start is not known
                continue
            p, nxt = w[b:t + 1], w[t + 1]
            if not nxt.isalpha():                                  # the word's end is not a copy of anything
                continue
            m = len(p)
            later = [s for s in starts if s < b and w[s:s + m] == p and s + m <= t]
            if not later:
                continue
            cont = {w[s + m] if w[s + m].isalpha() else "" for s in later}
            if cont != {nxt}:
                continue
            seen = prefix.get(p)
            if seen is not None and seen[nxt] / sum(seen.values()) >= 0.5:
                continue
            j = later[-1] + m - 1
            src[r, t] = j + 1
            gt[r, t, [t, j, j + 1]] = True
    return src, gt


def get(n_train=300_000, n_test=2_000, seed=0):
    """As tasks.get, plus copy_test [n, L] (the copy positions, where gt_test and src_test hold) and chars."""
    train, test, chars, table, prefix = _stories()
    xtr, ytr, _ = _windows(train, table, np.random.default_rng(seed), n_train)
    xte, yte, raw = _windows(test, table, np.random.default_rng(seed + 1_000_003), n_test)
    ste, gte = _copy_truth(raw, prefix)
    return dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte, src_test=ste, gt_test=gte,
                copy_test=ste >= 0, raw_test=raw, chars=chars, vocab=V)
