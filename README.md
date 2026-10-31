# 🎬 YouTube Title Generator (Qwen2.5-3B + SFT + DPO)

![Python](https://img.shields.io/badge/Python-3.12.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.45.2-yellow.svg)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-orange.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4.svg)
![Hardware](https://img.shields.io/badge/Hardware-RTX%204060%20(8GB)-green.svg)
![Task](https://img.shields.io/badge/Task-Title%20Generation-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🚀 Live Demo & Models

*   **🎮 Interactive Web App:** [Try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/mdanialFarokhi/youtube-title-generator)
*   **📥 Download Models:** [Download the SFT & DPO LoRA adapters (v1.0.0)](https://github.com/mohammaddanial-farokhi/15-youtube-title/releases/tag/v1.0.0)

---

## 🧠 About The Project

A complete end-to-end pipeline to fine-tune the **Qwen2.5-3B-Instruct** language model for generating highly engaging, click-worthy YouTube titles. 

This project demonstrates the full LLM alignment process: starting from a raw base model, applying **Supervised Fine-Tuning (SFT)** to learn the format, and finally using **Direct Preference Optimization (DPO)** to align the model with human preferences (preferring engaging titles over generic ones). The entire training pipeline is optimized to run on consumer hardware with an **8GB VRAM GPU (RTX 4060 Laptop)**.

**Result:** The DPO model successfully learned to avoid generic clickbait phrases and generate more intriguing titles, outperforming the SFT model in both BERTScore and METEOR metrics.

---

## 🎯 Key Features

1. **Two-Stage Fine-Tuning:** Combines standard SFT (to learn the format) with DPO (to learn human preferences).
2. **QLoRA Integration:** Uses 4-bit quantization and LoRA adapters to fit the 3B model into limited VRAM.
3. **ChatML Formatting:** Auto-formats data into Qwen's specific chat template (`<|im_start|>user...`) for optimal instruction-following.
4. **Automated Evaluation:** Includes a comprehensive evaluation script using `BERTScore` (semantic similarity) and `METEOR` (lexical overlap).
5. **Checkpoint Resume:** Robust training scripts that automatically detect and resume from the latest checkpoint safely.
6. **Side-by-Side Demo:** A custom Gradio UI that generates titles using the Base, SFT, and DPO models simultaneously for direct comparison.

---

## 📂 Project Structure

```text
.
├── data/
│   ├── evaluation/              # JSON files containing SFT & DPO predictions
│   │   ├── predictions_sft.json
│   │   └── predictions_dpo.json
│   │
│   └── model/                   # (Ignored in Git) Model weights & checkpoints
│       ├── checkpoints/         # SFT training checkpoints
│       ├── dpo-checkpoints/     # DPO training checkpoints
│       ├── final-DPO/           # Final DPO LoRA adapter
│       └── final-qwen-sft-youtube-titles/ # Final SFT LoRA adapter
│
├── main.py                      # Main training pipeline (SFT & DPO)
├── app.py                       # Gradio web UI (Base vs SFT vs DPO)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files and folders ignored by Git
└── README.md
```

---

## 📊 Dataset

**Source:** `shawhin/youtube-titles-dpo` (Hugging Face Hub)
*   **Total Examples:** 1,140
*   **Train Split:** 1,026
*   **Validation Split:** 114
*   **Format:** Conversational DPO format (Prompt, Chosen, Rejected).

---

## ⚙️ Training Configuration

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-3B-Instruct` | Multilingual instruction-tuned model |
| **Quantization** | 4-bit (nf4) | To fit into 8GB VRAM |
| **LoRA `r`** | 16 | Rank of LoRA matrices |
| **LoRA `alpha`** | 32 | Scaling factor |
| **SFT Learning Rate** | 2e-4 | Cosine scheduler |
| **DPO Learning Rate** | 2e-6 | Much lower than SFT to prevent catastrophic forgetting |
| **DPO `beta`** | 0.1 | KL penalty coefficient |
| **Epochs** | 2 (SFT) / 2 (DPO) | Validated to prevent overfitting |

---

## 📈 Evaluation Results

### Quantitative Metrics
Models were evaluated on the hold-out validation set (114 samples) using **BERTScore** (semantic meaning) and **METEOR** (vocabulary & synonyms).

| Metric | SFT Model | DPO Model | Improvement |
| :--- | :--- | :--- | :--- |
| **BERTScore Precision** | 0.8074 | **0.8108** | 📈 +0.0034 |
| **BERTScore Recall** | 0.8394 | 0.8383 | - |
| **BERTScore F1** | 0.8227 | **0.8240** | 📈 +0.0013 |
| **METEOR** | 0.3832 | **0.3865** | 📈 +0.0033 |
| **Final Train Loss** | 0.3321 | 0.6750 | *(DPO loss calculation differs)* |
| **Final Eval Loss** | 0.1358 | 0.7091 | *(Stable without overfitting)* |

*Note: The DPO model achieved a **58.7% Reward Accuracy** (eval_rewards/accuracies), indicating it successfully learned to distinguish between chosen (engaging) and rejected (generic) titles.*

### 🔍 Qualitative Analysis (The DPO Effect)
While the numerical improvements seem small, the qualitative difference in generation is striking:

*   **Base Model:** `Trying Hard: The Plan That Failed` (Off-topic)
*   **SFT Model:** `How to Make Your Written Plan Count (VIDEO)` (Robotic/Generic)
*   **DPO Model:** `How Much Hard Work Is Enough? (Plan Example)` (Highly engaging & click-worthy)

---

## ⚠️ Known Limitations

*   **Dataset Domain:** The training dataset is heavily focused on Data Science and Machine Learning topics. While the model generalizes well, it performs best on tech-related ideas.
*   **Reward Hacking:** Because the dataset is small (1k examples), aggressive DPO settings might occasionally cause the model to use overly dramatic words (e.g., "Secrets", "Unwrapped").
*   **Length Constraint:** The model is trained to strictly output titles between 30 and 75 characters.

---

## 🚀 Usage

### 1. Install Dependencies
Ensure you have Python 3.12 and CUDA installed. Then:
```bash
pip install -r requirements.txt
```

### 2. Training (SFT & DPO)
To start training from scratch or resume from a checkpoint:
```bash
python main.py
```

### 3. Web Application (Local)
To run the Gradio UI locally:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser to compare the Base, SFT, and DPO models side-by-side.

---

## 📜 License
This project is licensed under the MIT License - free to use, modify, and distribute.