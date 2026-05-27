import sys
import wandb

PROJECT = "cil-project"

tags = sys.argv[1:]
api = wandb.Api()
runs = api.runs(PROJECT, filters={"tags": {"$all": tags}})

rows = []
for run in runs:
    h = run.history(keys=["eval/score", "eval/mae", "eval/accuracy"])
    if h.empty:
        continue
    rows.append((run.name, h["eval/score"].max(), h["eval/mae"].min(), h["eval/accuracy"].max()))

rows.sort(key=lambda r: -r[1])

print(f"Tags: {tags}  ({len(rows)} runs)")
for name, score, mae, acc in rows:
    print(f"  {name:<60}  score={score:.4f}  mae={mae:.4f}  acc={acc:.4f}")
