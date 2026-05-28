import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC_DIR = Path("data")
OUT_DIR = Path("data/split")
VAL_FRACTION = 0.10
SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC_DIR / "train.csv")
train_df, val_df = train_test_split(df, test_size=VAL_FRACTION, random_state=SEED)
train_df.to_csv(OUT_DIR / "train.csv", index=False)
val_df.to_csv(OUT_DIR / "val.csv", index=False)
shutil.copy(SRC_DIR / "test.csv", OUT_DIR / "test.csv")

print(f"train: {len(train_df)} rows -> {OUT_DIR / 'train.csv'}")
print(f"val:   {len(val_df)} rows -> {OUT_DIR / 'val.csv'}")
print(f"test:  copied  -> {OUT_DIR / 'test.csv'}")
