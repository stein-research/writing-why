"""The paper's two word tasks: TinyStories V2 (tinywords.py) and WikiText-103 (wikiwords.py). Both share the
tokeniser, the vocabulary size V = 8,192, the window of 128 words and <eos>.

    get(task, n_train, n_test, seed) -> dict of x/y [n, L] for train and test, words, vocab
"""
TASKS = ("tinywords", "wikiwords")


def get(task, n_train=100_000, n_test=2_000, seed=0):
    if task == "tinywords":
        import tinywords
        return tinywords.get(n_train=n_train, n_test=n_test, seed=seed)
    if task == "wikiwords":
        import wikiwords
        return wikiwords.get(n_train=n_train, n_test=n_test, seed=seed)
    raise ValueError(f"unknown task {task!r}")
