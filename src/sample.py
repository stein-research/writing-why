"""Let a stories model write: continue a prompt one character at a time (the last 128 characters as context).

Stories and tinystories runs alike (a tinystories sample stops at <eos>; a <tag>_ckpt checkpoint works too).
Born samples from its own P_v, softmax from its softmax; both sharpened the same way, p^(1 / T) renormalised.
  python sample.py stories_born_s0 --prompt "Once upon a time" --chars 400 --temp 0.8
"""
import argparse
import json

import numpy as np
import torch

import stories
from models import build

p = argparse.ArgumentParser()
p.add_argument("tags", nargs="+")
p.add_argument("--prompt", default="Once upon a time, there was a little")
p.add_argument("--chars", type=int, default=400)
p.add_argument("--temp", type=float, default=0.8)
p.add_argument("--n", type=int, default=2, help="samples per model")
p.add_argument("--top-p", type=float, default=1.0, help="sample from the smallest set of characters holding this share")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--only", type=int, help="print only draw number ONLY (1-based; the earlier draws are still made, as the generator needs them)")
p.add_argument("--dump", help="append every draw's token ids to this json file ({tag: [ids, ...]})")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"

drawn = {}
for tag in args.tags:
    run = json.load(open(f"results/{tag}.json"))
    if run["task"] in ("tinywords", "wikiwords"):   # --chars counts words then
        import importlib
        tinywords = importlib.import_module(run["task"])   # wikiwords: the same interface
        words, index = tinywords.vocab()
        vocab, eos, banned = tinywords.V, tinywords.EOS, [0, tinywords.UNK]
        encode = lambda s: [index.get(t, tinywords.UNK) for t in tinywords.tokens(s)]
        decode = lambda ids: tinywords.decode(ids, words).replace("\n\n", " <eos>")
    else:
        if run["task"] == "tinystories":
            import tinystories
            chars, table = tinystories.vocab()
            vocab, eos = tinystories.V, tinystories.EOS
        else:
            _, _, chars, table, _ = stories._stories()
            vocab, eos = stories.V, None
        banned = list(range(stories.C0))
        encode = lambda s: [int(table[ord(c)]) for c in s]
        decode = lambda ids: "".join(chars[i - stories.C0] for i in ids).replace("\x03", " <eos>")
    model = build(run["model"], vocab, stories.L)
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu"))
    model = model.to(dev).eval()
    if model.family == "B":                                # float64: a plain readout passes float32's range
        model = model.double()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    at = f", step {run['step']}" if "step" in run else ""
    print(f"=== {tag} (test acc {run['test_acc']:.3f}, ppl {run['test_ppl']:.2f}{at}), T = {args.temp}, top-p {args.top_p}")
    for _ in range(args.n):
        ids = encode(args.prompt)
        for _ in range(args.chars):
            x = torch.tensor([ids[-stories.L:]], device=dev)
            with torch.no_grad():
                pv = model.probs(x)[0, -1] if model.family == "B" else torch.softmax(model(x)[0, -1], -1)
            pv = pv.double().clamp(min=0).cpu()
            pv[banned] = 0                                         # never absent or <unk>
            pv = pv ** (1 / args.temp)
            if args.top_p < 1:
                srt, idx = pv.sort(descending=True)
                cum = srt.cumsum(0) / srt.sum()
                srt[(cum - srt / srt.sum()) >= args.top_p] = 0             # keep through the one that crosses p
                pv = torch.zeros_like(pv).scatter_(0, idx, srt)
            ids.append(int(torch.multinomial(pv / pv.sum(), 1, generator=gen)))
            if ids[-1] == eos:
                break
        text = decode(ids)
        drawn.setdefault(tag, []).append(ids)
        k = len(drawn[tag])
        if args.only is None or args.only == k:
            import hashlib
            digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]
            ended = "ends" if ids[-1] == eos else "cut"
            print(f"[draw {k}, {ended}, token-id sha256 {digest}]", text.replace("\n", " / "), "\n")

if args.dump:
    import os
    old = json.load(open(args.dump)) if os.path.exists(args.dump) else {}
    old.update({f"{t}|{args.prompt}|T{args.temp}|p{args.top_p}|s{args.seed}": v for t, v in drawn.items()})
    json.dump(old, open(args.dump, "w"))
