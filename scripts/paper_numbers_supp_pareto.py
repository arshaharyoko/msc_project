import glob
import json

arms = {"raw": [], "anch": [], "prot": []}
for arm in arms:
    for f in sorted(glob.glob(f"artifacts/optimizer/latent_{arm}__s*.json")):
        for r in json.load(open(f))["runs"]:
            rt = r["round_trip"]
            arms[arm].append((rt["R"], rt["realism_95"], rt["feasible"], r["step_norm"]))

allruns = [x for v in arms.values() for x in v]
print("runs", len(allruns))
print("max R/R95", max(r / r95 for r, r95, _, _ in allruns))
print("max R", max(r for r, *_ in allruns), "R95 range", min(x[1] for x in allruns), max(x[1] for x in allruns))
for arm, v in arms.items():
    fe = [x for x in v if x[2]]
    print(arm, "n", len(v), "feasible", len(fe), "max R/R95", max(r / r95 for r, r95, *_ in v))
