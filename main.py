from datasets import Dataset, load_dataset, DatasetDict
import pandas as pd
import numpy as np
from collections import Counter
from trl import SFTTrainer, SFTConfig, DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
from transformers.trainer_utils import get_last_checkpoint
from tqdm import tqdm
from evaluate import load
import nltk
from nltk.translate.meteor_score import meteor_score
import json


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
# 2. DATA PREPROCESSING
# ======================================================
def prepare_sft_dataset(example):
    # For SFT, we only need the prompt and the chosen (winning) response.
    # The dataset already provides them as a list of dictionaries (chat format).
    # We concatenate them to create a full conversation.
    messages = example["prompt"] + example["chosen"]
    return {"messages": messages}


def sft_train_valid(dataset):
    train_dataset = dataset["train"].map(
        prepare_sft_dataset, remove_columns=dataset["train"].column_names
    )
    valid_dataset = dataset["valid"].map(
        prepare_sft_dataset, remove_columns=dataset["valid"].column_names
    )

    return train_dataset, valid_dataset


# =====================================================
# 3. MODEL & TOKENIZER LOADING
# =====================================================
def load_model_and_tokenizer(model_name):
    """Load a standard (non-quantized) model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# =====================================================
# 4. UTILITIES & INSPECTION
# =====================================================
def qwen_chat_format(user_input, system_message="You are a helpful assistant."):
    prompt = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def inspect_formatted_sample(dataset, tokenizer, index=0):
    """Inspects exactly what the model sees during SFT training."""
    print("\n" + "=" * 70)
    print(f"INSPECTING DATASET SAMPLE AT INDEX {index}")
    print("=" * 70)

    example = dataset[index]
    messages = example["messages"]

    print("\n[1] RAW MESSAGES (List of dicts):")
    for msg in messages:
        print(f"    Role: {msg['role']}")
        print(
            f"    Content: {msg['content'][:150]}..."
            if len(msg["content"]) > 150
            else f"    Content: {msg['content']}"
        )

    # 2. Apply Chat Template (What the text looks like before tokenization)
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    print("\n[2] FORMATTED TEXT (What the model actually reads as a string):")
    print(f"    {repr(formatted_text)}")

    # 3. Tokenize (What the model actually processes)
    tokenized = tokenizer(
        formatted_text,
        padding="max_length",
        max_length=128,  # Using a small length just to show padding clearly
        truncation=True,
        return_tensors="pt",
    )


def signle_prompt_test(
    trained_model,
    trained_tokenizer,
    test_prompt="This video is about how to focus hard without any distractions.",
):
    # 1. Format prompt exactly as it was during training (with Qwen chat template)
    messages = [
        {
            "role": "user",
            "content": f"Given the YouTube video idea write an engaging title.\n\n**Video Idea**: {test_prompt}",
        }
    ]
    formatted_prompt = trained_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2. Tokenize and convert to PyTorch Tensors, then move to GPU
    inputs = trained_tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    # 3. Generate text (not just a forward pass)
    with torch.no_grad():
        outputs = trained_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=trained_tokenizer.eos_token_id,
        )

    # 4. Extract only the newly generated tokens (ignore the prompt part)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]

    # 5. Decode back to text
    result_test = trained_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n--- Generated Title ---")
    print(f"user prompt:\n{test_prompt}")
    print("\nmodel result: ")
    print(result_test)
    print("-----------------------")


# =====================================================
# 5. QLORA CONFIGURATION
# =====================================================
def set_bnb():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    return bnb_config


def set_qlora(MODEL_NAME):
    bnb_config = set_bnb()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    return model, tokenizer, peft_config


# =====================================================
# 6. SFT TRAINER
# =====================================================
def sft_trainer(MODEL_NAME, train_dataset, valid_dataset, OUTPUT_DIR):
    # Define a separate directory for checkpoints
    CHECKPOINT_DIR = "data/model/checkpoints"

    # 1. Check if the FINAL model already exists
    final_adapter_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
    if os.path.exists(final_adapter_path):
        print(f"Existing final SFT model found at {OUTPUT_DIR}. Skipping training...")
        bnb_config = set_bnb()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
        return model, tokenizer

    # 2. Check for existing CHECKPOINTS to resume from
    last_checkpoint = None
    if os.path.isdir(CHECKPOINT_DIR):
        last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)

    if last_checkpoint is not None:
        print(f"Checkpoint found at {last_checkpoint}. Resuming training...")
    else:
        print("No checkpoint found. Starting training from scratch...")

    # 3. Setup Model and Trainer
    print("Setting up SFT Trainer...")
    model, tokenizer, peft_config = set_qlora(MODEL_NAME)

    training_args = SFTConfig(
        output_dir=CHECKPOINT_DIR,  # Save checkpoints in data/model/checkpoints
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        packing=False,
        max_seq_length=512,
    )

    # Define a formatting function to apply Qwen chat template to the messages
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            text = tokenizer.apply_chat_template(
                example["messages"][i], tokenize=False, add_generation_prompt=False
            )
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    # 4. Start training (with resume capability)
    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 5. Save the final model to the main OUTPUT_DIR
    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training finished and final model saved!")

    return trainer.model, tokenizer


# =====================================================
# 6. EVALUATION
# =====================================================
def make_predictions(model, tokenizer, raw_dataset):
    # Define the save directory and file path
    eval_dir = "data/evaluation"
    file_path = os.path.join(eval_dir, "predictions_cache.json")

    # Check if the cached file already exists
    if os.path.exists(file_path):
        print(f"Cached predictions found at {file_path}. Loading from file...")
        with open(file_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)

        predictions = cached_data["predictions"]
        references = cached_data["references"]

        print(f"Loaded {len(predictions)} cached titles.")
        return predictions, references

    # If file does not exist, start generating
    print("No cache found. Starting generation from scratch...")
    os.makedirs(eval_dir, exist_ok=True)  # Create directory if it doesn't exist

    model.eval()
    predictions = []
    references = []

    valid_dataset = raw_dataset["valid"]

    for example in tqdm(valid_dataset, desc="Generating titles"):
        messages = example["prompt"]
        # Format the prompt with Qwen chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,  # Use greedy decoding for deterministic evaluation
                pad_token_id=tokenizer.eos_token_id,
            )

        # Extract generated title
        generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
        generated_title = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Extract reference title (the "chosen" response)
        reference_title = example["chosen"][0]["content"].strip()

        predictions.append(generated_title)
        references.append(reference_title)

    print(f"Generated {len(predictions)} titles.")

    # Save the generated results to the JSON file
    print(f"Saving generated titles to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            {"predictions": predictions, "references": references},
            f,
            indent=4,
            ensure_ascii=False,
        )
    print("Saved successfully!")

    return predictions, references


def bert_score(predictions, references):
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="distilbert-base-uncased",
    )

    print(f"BERTScore Precision: {sum(results['precision']) / len(results['precision']):.4f}")
    print(f"BERTScore Recall:    {sum(results['recall']) / len(results['recall']):.4f}")
    print(f"BERTScore F1:        {sum(results['f1']) / len(results['f1']):.4f}")


def calculate_meteor(predictions, references):
    nltk.download("wordnet")
    nltk.download("punkt")
    nltk.download("omw-1.4")
    nltk.download("punkt_tab")

    meteor_scores = []
    for pred, ref in zip(predictions, references):
        # Tokenize both texts
        pred_tokens = nltk.word_tokenize(pred)
        ref_tokens = nltk.word_tokenize(ref)

        # Calculate METEOR (requires a list of references)
        score = meteor_score([ref_tokens], pred_tokens)
        meteor_scores.append(score)

    mean_meteor = sum(meteor_scores) / len(meteor_scores)
    print(f"Mean METEOR score: {mean_meteor:.4f}")


# =====================================================
# 7. DPO
# =====================================================
def prepare_dpo_dataset(example):
    """Extracts plain text strings from the conversational format for DPO."""
    return {
        "prompt": example["prompt"][0]["content"],
        "chosen": example["chosen"][0]["content"],
        "rejected": example["rejected"][0]["content"],
    }


def dpo_trainer(DPO_OUT_DIR, model, tokenizer, raw_dataset):
    # Define a separate directory for DPO checkpoints
    DPO_CHECKPOINT_DIR = "data/model/dpo-checkpoints"

    dpo_train_dataset = raw_data["train"].map(prepare_dpo_dataset)
    dpo_valid_dataset = raw_data["valid"].map(prepare_dpo_dataset)

    # 1. Check if the FINAL DPO model already exists
    dpo_adapter_path = os.path.join(DPO_OUT_DIR, "adapter_config.json")
    if os.path.exists(dpo_adapter_path):
        print(f"Existing DPO model found at {DPO_OUT_DIR}. Skipping training...")
        from peft import PeftModel

        # The 'model' passed to this function is the SFT model (a PeftModel).
        # We stack the DPO adapter on top of it.
        dpo_model = PeftModel.from_pretrained(model, DPO_OUT_DIR)
        return dpo_model, tokenizer

    # 2. Check for existing CHECKPOINTS to resume from
    last_checkpoint = None
    if os.path.isdir(DPO_CHECKPOINT_DIR):
        last_checkpoint = get_last_checkpoint(DPO_CHECKPOINT_DIR)

    if last_checkpoint is not None:
        print(f"Checkpoint found at {last_checkpoint}. Resuming DPO training...")
    else:
        print("No DPO checkpoint found. Starting DPO training from scratch...")

    # 3. Setup DPO Trainer
    training_args = DPOConfig(
        output_dir=DPO_CHECKPOINT_DIR,
        beta=0.1,
        learning_rate=7e-7,
        num_train_epochs=1,
        loss_type="sigmoid",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        bf16=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25,
        report_to="none",
        max_length=512,
        max_prompt_length=384,
    )

    print("Enabling input gradients for DPO training...")
    model.enable_input_require_grads()

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dpo_train_dataset,
        eval_dataset=dpo_valid_dataset,
    )

    # 4. Start training (with resume capability)
    print("Starting DPO training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("DPO training finished!")

    # 5. Save the final DPO model
    print(f"Saving final DPO model to {DPO_OUT_DIR}...")
    trainer.save_model(DPO_OUT_DIR)
    tokenizer.save_pretrained(DPO_OUT_DIR)
    print("DPO model saved successfully!")

    # Return the trained model and tokenizer for immediate use
    return trainer.model, tokenizer


if __name__ == "__main__":
    # ==================== STEP 1: CONFIGURATION ====================
    DATASET_NAME = "shawhin/youtube-titles-dpo"
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    OUTPUT_DIR = "data/model/final-qwen-sft-youtube-titles"
    DPO_OUT_DIR = "data/model/final-DPO"

    # ==================== STEP 2: LOAD RAW DATA ====================
    raw_data = loading_dataset(DATASET_NAME)

    # Optional: Inspect dataset structure
    # dataset_first_look(raw_data)

    # ==================== STEP 3: PREPARE DATA ====================
    train_dataset, valid_dataset = sft_train_valid(raw_data)

    # ==================== STEP 4: (OPTIONAL) INSPECT FORMATTED DATA ====================
    # inspect_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # inspect_formatted_sample(train_dataset, inspect_tokenizer)

    # ==================== STEP 5: START TRAINING (OR LOAD IF EXISTS) ====================
    trained_model, trained_tokenizer = sft_trainer(
        MODEL_NAME, train_dataset, valid_dataset, OUTPUT_DIR
    )

    # ==================== STEP 6: TEST TRAINED MODEL ====================
    # signle_prompt_test(
    #     trained_model,
    #     trained_tokenizer,
    #     test_prompt="battery- phone - apple - protect",
    # )

    # ==================== STEP 7: EVALUATION ====================
    # predictions, references = make_predictions(trained_model, trained_tokenizer, raw_data)
    # bert_score(predictions, references)
    # calculate_meteor(predictions, references)

    # ==================== STEP 8: DPO ====================
    dpo_model, dpo_tokenizer = dpo_trainer(
        DPO_OUT_DIR, trained_model, trained_tokenizer, raw_data
    )

    # ==================== STEP 9: TEST DPO MODEL ====================
