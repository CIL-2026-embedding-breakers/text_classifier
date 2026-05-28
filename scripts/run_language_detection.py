from data_analysis.utils.language_detection import split_languages_in_csv

SRC = "data/split"
OUT = "data/lang"

for name in ("train.csv", "val.csv", "test.csv"):
    print(f"=== {name} ===")
    split_languages_in_csv(
        input_filepath=f"{SRC}/{name}",
        text_column="sentence",
        output_dir=OUT,
        confidence_threshold=0.63,
    )
