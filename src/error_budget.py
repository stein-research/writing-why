"""Error budget of the ibm_kingston runs (Section 4.5): for each job that ran calibration circuits, the scatter of
the 32 calibration readings about their fitted line (slope and offset, the fit hw_word.py makes), with the two fitted
parameters taken out (ddof = 2), against the scatter shot noise alone gives at the measured values,
sqrt(mean((1 - p^2) / shots)).

    python src/error_budget.py        Reads ibm_jobs/*/record.json and counts.json; writes results/error_budget.json.
"""
import glob
import json
import os

import numpy as np

out = []
for rec_f in sorted(glob.glob("ibm_jobs/*/record.json")):
    rec = json.load(open(rec_f))
    counts = json.load(open(os.path.join(os.path.dirname(rec_f), "counts.json")))
    shots = int(rec["shots"])
    cal = [(float(c["note"].split("ideal ")[1]), (n.get("0", 0) - n.get("1", 0)) / shots)
           for c, n in zip(rec["circuits"], counts) if "calibration" in c["note"]]
    if not cal:
        continue
    x, p = np.array(cal).T
    slope, off = np.polyfit(x, p, 1)
    res = p - slope * x - off
    sd, shot = float(np.sqrt((res ** 2).sum() / (len(p) - 2))), float(np.sqrt(np.mean((1 - p ** 2) / shots)))
    out.append(dict(job=rec["job_id"], purpose=rec["purpose"], n=len(p), slope=float(slope), offset=float(off),
                    resid_sd_ddof2=sd, resid_sd_ddof0=float(res.std()), shot_sd=shot, ratio=sd / shot))
    print(f"{rec['created'][:16]}  {rec['purpose'][-60:]:60s}  slope {slope:.3f}  scatter {sd:.4f}  "
          f"shot {shot:.4f}  ratio {sd / shot:.2f}")
json.dump(out, open("results/error_budget.json", "w"), indent=1)
