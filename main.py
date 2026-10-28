from datasets import Dataset, load_dataset, DatasetDict
import pandas as pd
import numpy as np
from collections import Counter
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


# =====================================================
# 1. DATA LOADING & ANALYSIS
# =====================================================
def loading_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset


def dataset_first_look(dataset, sample_size: int = 2, text_columns=None):
    """Perform a quick initial analysis of the DPO dataset (DatasetDict format)."""
    line = "=" * 70
    print(f"\n{line}\nDATASET FIRST LOOK (DPO FORMAT)\n{line}")

    # Helper function to extract raw text from chat format: [{"content": "...", "role": "..."}]
    def extract_text(val):
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            return " ".join([str(item.get("content", "")) for item in val])
        elif isinstance(val, dict):
            return str(val.get("content", ""))
        return str(val) if val is not None else ""

    # ---------- 1) Splits and Shape ----------
    print("\n[1] SPLITS AND SHAPE")
    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        for split in splits:
            print(f"    - split '{split}': {len(dataset[split]):,} rows")
        main_split = "train" if "train" in splits else splits[0]
        data = dataset[main_split]
        print(f"    * Analyzing main split: '{main_split}'")
    else:
        data = dataset
        print(f"    - single dataset: {len(data):,} rows")

    # ---------- 2) Columns ----------
    print("\n[2] COLUMNS")
    for i, col in enumerate(data.column_names):
        print(f"    [{i}] {col!r}  type={data.features[col]}")

    # Get data as lists for analysis without Pandas
    col_data = {col: data[col] for col in data.column_names}

    if text_columns is None:
        default_dpo_cols = ["prompt", "chosen", "rejected"]
        text_columns = [c for c in default_dpo_cols if c in data.column_names]

    # ---------- 3) Null / empty ----------
    print("\n[3] NULL / EMPTY VALUES")
    for col in data.column_names:
        n_empty = sum(1 for x in col_data[col] if len(extract_text(x).strip()) == 0)
        pct = 100 * n_empty / len(data) if len(data) else 0
        status = "OK  " if n_empty == 0 else "WARN"
        print(f"    [{status}] {col}: {n_empty:,} empty ({pct:.2f}%)")

    # ---------- 4) Unique values ----------
    print("\n[4] UNIQUE VALUES PER COLUMN")
    for col in data.column_names:
        extracted_list = [extract_text(x) for x in col_data[col]]
        n_unique = len(set(extracted_list))
        if n_unique <= 20:
            print(f"    - {col}: {n_unique} unique values")
            for val, count in Counter(extracted_list).most_common():
                pct = 100 * count / len(data)
                print(f"        * {val!r}: {count:,} ({pct:.1f}%)")
        else:
            print(f"    - {col}: {n_unique:,} unique values")

    # ---------- 5) DPO Specific: Chosen vs Rejected Length ----------
    if "chosen" in text_columns and "rejected" in text_columns:
        print("\n[5] DPO SPECIFIC: CHOSEN vs REJECTED LENGTHS")
        len_chosen = [len(extract_text(x)) for x in col_data["chosen"]]
        len_rejected = [len(extract_text(x)) for x in col_data["rejected"]]

        diff = np.array(len_chosen) - np.array(len_rejected)
        print(f"    - Mean Chosen length:   {np.mean(len_chosen):.1f} chars")
        print(f"    - Mean Rejected length: {np.mean(len_rejected):.1f} chars")
        print(
            f"    - Chosen longer in:     {(diff > 0).sum():,} rows ({(diff > 0).mean()*100:.1f}%)"
        )
        print(
            f"    - Rejected longer in:   {(diff < 0).sum():,} rows ({(diff < 0).mean()*100:.1f}%)"
        )
        print(
            f"    - Equal length:         {(diff == 0).sum():,} rows ({(diff == 0).mean()*100:.1f}%)"
        )

    # ---------- 6) Text length distribution ----------
    print("\n[6] TEXT LENGTH DISTRIBUTION")
    print(f"    (columns: {text_columns})")

    for col in text_columns:
        lengths = np.array([len(extract_text(x)) for x in col_data[col]])
        if lengths.size == 0:
            continue
        print(f"    - {col}:")
        print(f"        * min:    {lengths.min()}")
        print(f"        * max:    {lengths.max()}")
        print(f"        * mean:   {lengths.mean():.1f}")
        print(f"        * median: {np.median(lengths):.1f}")
        print(f"        * std:    {lengths.std():.1f}")

        buckets = [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
        labels = [
            "0-20",
            "20-50",
            "50-100",
            "100-200",
            "200-500",
            "500-1k",
            "1k-2k",
            "2k-5k",
            "5k+",
        ]
        hist = np.histogram(lengths, bins=buckets)[0]
        total = hist.sum() or 1
        for label, count in zip(labels, hist):
            pct = 100 * count / total
            bar = "#" * int(pct / 2)
            print(f"          {label:>7}: {count:>7,} ({pct:5.1f}%) {bar}")

    # ---------- 7) Roles Distribution ----------
    print("\n[7] ROLES DISTRIBUTION")
    for col in text_columns:
        roles = []
        for val in col_data[col]:
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "role" in item:
                        roles.append(item["role"])
            elif isinstance(val, dict) and "role" in val:
                roles.append(val["role"])

        if roles:
            print(f"    - {col}:")
            for role, count in Counter(roles).most_common():
                pct = 100 * count / len(roles)
                print(f"        * {role!r}: {count:,} ({pct:.1f}%)")

    # ---------- 8) Duplicates ----------
    print("\n[8] DUPLICATES")
    print(
        f"    - exact duplicate rows (all cols): {len(data) - len(set(tuple(str(val) for val in row.values()) for row in data)):,}"
    )
    for col in text_columns:
        extracted_list = [extract_text(x) for x in col_data[col]]
        n = len(extracted_list) - len(set(extracted_list))
        pct = 100 * n / len(data) if len(data) else 0
        print(f"    - duplicate values in {col!r}: {n:,} ({pct:.2f}%)")

    # ---------- 9) Samples ----------
    print("\n[9] SAMPLES")
    for i in range(min(sample_size, len(data))):
        print(f"\n    --- Sample {i + 1} ---")
        for col in data.column_names:
            val = col_data[col][i]
            if col in text_columns:
                val_str = extract_text(val).replace("\n", " ")
                if len(val_str) > 250:
                    val_str = val_str[:250] + "..."
                print(f"        [{col}] {val_str!r}")
            else:
                val_str = str(val)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"        {col}: {val_str!r}")

    print(f"\n{line}\nAnalysis finished.\n{line}")


# =====================================================
# MODEL LOADING & QLORA SETUP
# =====================================================
def load_model_and_tokenizer(model_name):
    """Load a standard (non-quantized) model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


if __name__ == "__main__":
    # ==================== CONFIG ====================
    DATASET_NAME = "shawhin/youtube-titles-dpo"
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    # ==================== STEP 1: Load Dataset ====================
    raw_data = loading_dataset(DATASET_NAME)
    # dataset_first_look(raw_data)

    # ==================== STEP ====================
    base_model, base_tokenizer = load_model_and_tokenizer(MODEL_NAME)
