from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# Preprocessing sanity-check notebook\n\n"
        "This notebook is used to validate the preprocessing logic before launching the XLM-R runs.\n\n"
        "It checks:\n"
        "- title/review splitting\n"
        "- review_only vs title_plus_review\n"
        "- light / medium / heavy preprocessing\n"
        "- URL token preservation\n"
        "- tokenization behavior"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import pandas as pd"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "ROOT = Path.cwd()\n"
        "\n"
        "# Climb up until we find a folder that looks like the repo root\n"
        "while ROOT != ROOT.parent and not ((ROOT / 'data').exists() and (ROOT / 'deep_learning').exists()):\n"
        "    ROOT = ROOT.parent\n"
        "\n"
        "os.chdir(ROOT)\n"
        "print('Working directory:', Path.cwd())\n"
        "\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "from transformers import AutoTokenizer\n"
        "from deep_learning.preprocessing.preprocessor import Preprocessor\n"
        "from deep_learning.code_configs import Preprocessor_Configs"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "DATA_PATH = Path('data/train.csv')\n"
        "SAMPLE_N = 30\n"
        "\n"
        "if not DATA_PATH.exists():\n"
        "    raise FileNotFoundError(f'Could not find {DATA_PATH}')\n"
        "\n"
        "df = pd.read_csv(DATA_PATH)\n"
        "df_sample = df.sample(n=min(SAMPLE_N, len(df)), random_state=42).reset_index(drop=True)\n"
        "\n"
        "print('Loaded sample shape:', df_sample.shape)\n"
        "df_sample.head()"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "print(df.columns.tolist())"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "def display_example(title, text):\n"
        "    print('=' * 100)\n"
        "    print(title)\n"
        "    print('=' * 100)\n"
        "    print(text)\n"
        "    print()"
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## Manual examples")
)

cells.append(
    nbf.v4.new_code_cell(
        "manual_examples = [\n"
        "    'Great product\\n\\nI absolutely loved it!!!!!!! 😊 Check https://example.com for details.',\n"
        "    \"Not bad at all\\n\\nI didn’t expect much, but it works surprisingly well!!!\",\n"
        "    'Terrible quality\\n\\nDo not buy this product!!!!! It broke after 2 days... www.badshop.com',\n"
        "    'Simple title only without body',\n"
        "    'Amazing deal\\n\\nSO GOOD!!! REALLY REALLY GOOD!!! Visit http://shop.com now!!! 😍'\n"
        "]\n"
        "\n"
        "tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')\n"
        "\n"
        "for input_format in ['review_only', 'title_plus_review']:\n"
        "    for level in ['light', 'medium', 'heavy']:\n"
        "        cfg = Preprocessor_Configs(\n"
        "            preprocessing_level=level,\n"
        "            input_format=input_format,\n"
        "            truncation=True,\n"
        "            max_length=128,\n"
        "            padding=False,\n"
        "        )\n"
        "        preproc = Preprocessor(cfg, tokenizer, return_special_tokens_mask=False)\n"
        "\n"
        "        print('#' * 120)\n"
        "        print(f'INPUT FORMAT = {input_format} | PREPROCESSING LEVEL = {level}')\n"
        "        print('#' * 120)\n"
        "\n"
        "        for i, example in enumerate(manual_examples, start=1):\n"
        "            processed = preproc.preprocess_one_text(example)\n"
        "            display_example(f'Example {i}', processed)"
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## Real dataset examples")
)

cells.append(
    nbf.v4.new_code_cell(
        "tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')\n"
        "\n"
        "for idx in range(5):\n"
        "    raw_text = df_sample.loc[idx, 'sentence']\n"
        "\n"
        "    print('#' * 120)\n"
        "    print(f'RAW EXAMPLE {idx}')\n"
        "    print('#' * 120)\n"
        "    print(raw_text)\n"
        "    print()\n"
        "\n"
        "    for input_format in ['review_only', 'title_plus_review']:\n"
        "        for level in ['light', 'medium', 'heavy']:\n"
        "            cfg = Preprocessor_Configs(\n"
        "                preprocessing_level=level,\n"
        "                input_format=input_format,\n"
        "                truncation=True,\n"
        "                max_length=128,\n"
        "                padding=False,\n"
        "            )\n"
        "            preproc = Preprocessor(cfg, tokenizer, return_special_tokens_mask=False)\n"
        "            processed = preproc.preprocess_one_text(raw_text)\n"
        "\n"
        "            display_example(f'{input_format} | {level}', processed)"
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## Tabular preview")
)

cells.append(
    nbf.v4.new_code_cell(
        "tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')\n"
        "\n"
        "rows = []\n"
        "\n"
        "for _, row in df_sample.head(10).iterrows():\n"
        "    raw_text = row['sentence']\n"
        "    entry = {'raw_text': raw_text}\n"
        "\n"
        "    for input_format in ['review_only', 'title_plus_review']:\n"
        "        for level in ['light', 'medium', 'heavy']:\n"
        "            cfg = Preprocessor_Configs(\n"
        "                preprocessing_level=level,\n"
        "                input_format=input_format,\n"
        "                truncation=True,\n"
        "                max_length=128,\n"
        "                padding=False,\n"
        "            )\n"
        "            preproc = Preprocessor(cfg, tokenizer, return_special_tokens_mask=False)\n"
        "            entry[f'{level}_{input_format}'] = preproc.preprocess_one_text(raw_text)\n"
        "\n"
        "    rows.append(entry)\n"
        "\n"
        "preview_df = pd.DataFrame(rows)\n"
        "pd.set_option('display.max_colwidth', 300)\n"
        "preview_df"
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## URL preservation check")
)

cells.append(
    nbf.v4.new_code_cell(
        "tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')\n"
        "\n"
        "cfg = Preprocessor_Configs(\n"
        "    preprocessing_level='light',\n"
        "    input_format='review_only',\n"
        "    truncation=True,\n"
        "    max_length=128,\n"
        "    padding=False,\n"
        ")\n"
        "\n"
        "preproc = Preprocessor(cfg, tokenizer, return_special_tokens_mask=False)\n"
        "\n"
        "url_examples = df[df['sentence'].astype(str).str.contains(r'https?://|www\\.', regex=True, na=False)].head(10)\n"
        "\n"
        "for i, text in enumerate(url_examples['sentence'].tolist(), start=1):\n"
        "    processed = preproc.preprocess_one_text(text)\n"
        "    print('=' * 100)\n"
        "    print(f'URL EXAMPLE {i}')\n"
        "    print('=' * 100)\n"
        "    print('RAW:')\n"
        "    print(text)\n"
        "    print()\n"
        "    print('PROCESSED:')\n"
        "    print(processed)\n"
        "    print()"
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## Length comparison")
)

cells.append(
    nbf.v4.new_code_cell(
        "tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')\n"
        "\n"
        "stats = []\n"
        "\n"
        "for _, row in df_sample.iterrows():\n"
        "    raw_text = row['sentence']\n"
        "\n"
        "    for input_format in ['review_only', 'title_plus_review']:\n"
        "        for level in ['light', 'medium', 'heavy']:\n"
        "            cfg = Preprocessor_Configs(\n"
        "                preprocessing_level=level,\n"
        "                input_format=input_format,\n"
        "                truncation=True,\n"
        "                max_length=128,\n"
        "                padding=False,\n"
        "            )\n"
        "            preproc = Preprocessor(cfg, tokenizer, return_special_tokens_mask=False)\n"
        "            processed = preproc.preprocess_one_text(raw_text)\n"
        "\n"
        "            stats.append({\n"
        "                'input_format': input_format,\n"
        "                'level': level,\n"
        "                'char_len': len(processed),\n"
        "                'word_len': len(processed.split())\n"
        "            })\n"
        "\n"
        "stats_df = pd.DataFrame(stats)\n"
        "stats_df.groupby(['input_format', 'level'])[['char_len', 'word_len']].mean().round(2)"
    )
)

cells.append(
    nbf.v4.new_markdown_cell("## Tokenization sanity check")
)

cells.append(
    nbf.v4.new_code_cell(
        "tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')\n"
        "\n"
        "example = manual_examples[0]\n"
        "\n"
        "for level in ['light', 'medium', 'heavy']:\n"
        "    cfg = Preprocessor_Configs(\n"
        "        preprocessing_level=level,\n"
        "        input_format='title_plus_review',\n"
        "        truncation=True,\n"
        "        max_length=128,\n"
        "        padding=False,\n"
        "    )\n"
        "    preproc = Preprocessor(cfg, tokenizer, return_special_tokens_mask=True)\n"
        "    processed = preproc.preprocess_one_text(example)\n"
        "\n"
        "    encoded = tokenizer(\n"
        "        processed,\n"
        "        truncation=True,\n"
        "        max_length=128,\n"
        "        padding=False,\n"
        "        return_special_tokens_mask=True,\n"
        "    )\n"
        "\n"
        "    print('=' * 100)\n"
        "    print(level.upper())\n"
        "    print('=' * 100)\n"
        "    print('Processed text:')\n"
        "    print(processed)\n"
        "    print()\n"
        "    print('Tokens:')\n"
        "    print(tokenizer.convert_ids_to_tokens(encoded['input_ids'][:40]))\n"
        "    print()"
    )
)

nb["cells"] = cells

output_path = Path("preprocessing_sanity_check.ipynb")
with output_path.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook created at: {output_path.resolve()}")