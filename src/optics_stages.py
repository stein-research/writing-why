"""The light at each detection stage of the compiled 16-copy model (compile_optics.inner's stage report), saved:
the text's photons per balanced pair come from here (pair light x photons per slot).

    python optics_stages.py [--n 8]        # writes results/optics_stages.json
"""
import json, sys, torch
import tasks
from models import build
from compile_optics import compiled
tag = "words_born-therm16p2_s0"
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8
run = json.load(open(f"results/{tag}.json")); d = tasks.get(run["task"], n_train=10)
model = build(run["model"], d["vocab"], d["x_test"].shape[1])
model.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu")); model = model.double().eval()
st = []
with torch.no_grad():
    compiled(model, torch.tensor(d["x_test"][:n]), stages=st)
out = [dict(stage=a, log10_light=b if isinstance(b, float) else str(b)) for a, b in st]
for o in out: print(o)
pair = {o["stage"]: o["log10_light"] for o in out}
summary = {}
for k, v in pair.items():
    if "balanced pair" in k: summary[k] = {f"photons at 1e{p}": round(10 ** (float(v) + p)) for p in (10, 11, 12)}
    if "overlap pair" in k:
        lp = float(str(v).split("/")[-1]); summary[k] = {f"photons at 1e{p}": round(10 ** (lp + p)) for p in (10, 11, 12)}
print(summary)
json.dump(dict(tag=tag, windows=n, stages=out, photons_per_pair=summary), open("results/optics_stages.json", "w"), indent=1)
