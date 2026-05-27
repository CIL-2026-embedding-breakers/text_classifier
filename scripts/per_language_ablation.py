import numpy as np
import pandas as pd
from pathlib import Path

LR = "5.00e-04"
EP = 5
LANG_DIR = Path("data/lang")


def run_dir(model, tag):
    short = model.split("/")[-1]
    return Path(f"fine_tuning/{short}_lora_light_review_only_{tag}_lr{LR}_ep{EP}")


MULTI_DIR      = run_dir("xlm-roberta-base", "multilingual")
XLMR_EN_DIR    = run_dir("xlm-roberta-base", "xlmr_en")
XLMR_DE_DIR    = run_dir("xlm-roberta-base", "xlmr_de")
DEBERTA_EN_DIR = run_dir("microsoft/deberta-v3-base", "deberta_en")
GBERT_DE_DIR   = run_dir("deepset/gbert-base", "gbert_de")


def score(true, pred):
    mae = np.abs(np.asarray(true) - np.asarray(pred)).mean()
    return 1 - mae / 4, mae, (np.asarray(true) == np.asarray(pred)).mean()


def load(run_dir):
    return pd.read_csv(run_dir / "validation_results.csv").set_index("id")


def show(name, true, pred):
    s, mae, acc = score(true, pred)
    print(f"  {name:<48}  score={s:.4f}  MAE={mae:.4f}  acc={acc:.4f}  (n={len(pred)})")


multi      = load(MULTI_DIR)
xlmr_en    = load(XLMR_EN_DIR)
xlmr_de    = load(XLMR_DE_DIR)
deberta_en = load(DEBERTA_EN_DIR)
gbert_de   = load(GBERT_DE_DIR)

en_ids  = set(pd.read_csv(LANG_DIR / "EN" / "val.csv")["id"])
de_ids  = set(pd.read_csv(LANG_DIR / "DE" / "val.csv")["id"])
unc_ids = set(pd.read_csv(LANG_DIR / "UNCERTAIN" / "val.csv")["id"])

print("Validation split:")
print(f"  EN rows:        {len(en_ids):>5}")
print(f"  DE rows:        {len(de_ids):>5}")
print(f"  UNCERTAIN rows: {len(unc_ids):>5}")
print()

print("Per-bucket scores from the multilingual baseline:")
mb_en  = multi.loc[multi.index.isin(en_ids)]
mb_de  = multi.loc[multi.index.isin(de_ids)]
mb_unc = multi.loc[multi.index.isin(unc_ids)]
show("multilingual on EN bucket",        mb_en["true_label"],  mb_en["predicted_label"])
show("multilingual on DE bucket",        mb_de["true_label"],  mb_de["predicted_label"])
show("multilingual on UNCERTAIN bucket", mb_unc["true_label"], mb_unc["predicted_label"])
print()

print("EN specialists on EN bucket:")
show("xlmr_en   (same-backbone control)", xlmr_en["true_label"],    xlmr_en["predicted_label"])
show("deberta_en (monolingual)",          deberta_en["true_label"], deberta_en["predicted_label"])
print()

print("DE specialists on DE bucket:")
show("xlmr_de  (same-backbone control)", xlmr_de["true_label"],  xlmr_de["predicted_label"])
show("gbert_de (monolingual)",           gbert_de["true_label"], gbert_de["predicted_label"])
print()

print("=== Strategy A: drop UNCERTAIN ===")
mb_union = multi.loc[multi.index.isin(en_ids | de_ids)]
show("multilingual on EN+DE rows", mb_union["true_label"], mb_union["predicted_label"])

true_xlmr = pd.concat([xlmr_en["true_label"],    xlmr_de["true_label"]])
pred_xlmr = pd.concat([xlmr_en["predicted_label"], xlmr_de["predicted_label"]])
show("xlmr_en + xlmr_de  (data-only specialization)", true_xlmr, pred_xlmr)

true_mono = pd.concat([deberta_en["true_label"],    gbert_de["true_label"]])
pred_mono = pd.concat([deberta_en["predicted_label"], gbert_de["predicted_label"]])
show("deberta_en + gbert_de  (monolingual)", true_mono, pred_mono)
print()

print("=== Strategy B: specialists + multilingual fallback for UNCERTAIN ===")
fallback = multi.loc[multi.index.isin(unc_ids)]
show("multilingual on all rows", multi["true_label"], multi["predicted_label"])

true_xlmr_b = pd.concat([xlmr_en["true_label"],    xlmr_de["true_label"],    fallback["true_label"]])
pred_xlmr_b = pd.concat([xlmr_en["predicted_label"], xlmr_de["predicted_label"], fallback["predicted_label"]])
show("xlmr specialists + fallback", true_xlmr_b, pred_xlmr_b)

true_mono_b = pd.concat([deberta_en["true_label"],    gbert_de["true_label"],    fallback["true_label"]])
pred_mono_b = pd.concat([deberta_en["predicted_label"], gbert_de["predicted_label"], fallback["predicted_label"]])
show("monolingual specialists + fallback", true_mono_b, pred_mono_b)
