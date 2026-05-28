"""Download deepset/gbert-large to scratch and patch the missing model_type."""

import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "deepset/gbert-large"
TARGET = Path(f"/work/scratch/{os.environ['USER']}/text_classifier_cache/gbert-large")

TARGET.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=REPO_ID, local_dir=str(TARGET))

cfg_path = TARGET / "config.json"
cfg = json.loads(cfg_path.read_text())
if "model_type" not in cfg:
    cfg["model_type"] = "bert"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"Patched {cfg_path}: added model_type=bert")
else:
    print(f"No patch needed (model_type={cfg['model_type']})")

print(f"Ready: {TARGET}")
