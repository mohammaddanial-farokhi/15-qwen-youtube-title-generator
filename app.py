import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==================== CONFIG ====================
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SFT_ADAPTER_PATH = "data/model/final-qwen-sft-youtube-titles"
DPO_ADAPTER_PATH = "data/model/final-DPO"

# ==================== LOAD MODELS (4-BIT) ====================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading Base model in 4-bit...")
# Use 4-bit quantization to fit perfectly in VRAM and avoid meta-device errors
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True,
)

print("Loading adapters...")
# Load both adapters into the SAME model
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, adapter_name="sft")
model.load_adapter(DPO_ADAPTER_PATH, adapter_name="dpo")
model.eval()

print("All models loaded successfully!")


# ==================== GENERATION FUNCTION ====================
def generate_titles(video_idea):
    if not video_idea.strip():
        return (
            "Please enter a video idea.",
            "Please enter a video idea.",
            "Please enter a video idea.",
        )

    # Fixed prompt (exact same as training)
    messages = [
        {
            "role": "user",
            "content": (
                f"Given the YouTube video idea write an engaging title.\n\n"
                f"**Video Idea**: {video_idea}\n\n"
                f"**Additional Guidance**:\n"
                f"- Title should be between 30 and 75 characters long\n"
                f"- Only return the title idea, nothing else!"
            ),
        }
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    def _generate(model_instance):
        with torch.no_grad():
            outputs = model_instance.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

    # 1. Base Output (Disable all adapters)
    with model.disable_adapter():
        base_result = _generate(model)

    # 2. SFT Output (Activate SFT adapter)
    model.set_adapter("sft")
    sft_result = _generate(model)

    # 3. DPO Output (Activate DPO adapter)
    model.set_adapter("dpo")
    dpo_result = _generate(model)

    return base_result, sft_result, dpo_result


# ==================== GRADIO INTERFACE ====================
with gr.Blocks(title="YouTube Title Generator: Base → SFT → DPO") as demo:
    gr.Markdown("""
    # 🎬 YouTube Title Generator: The Evolution of an LLM

    This demo shows how a base language model progressively improves through fine-tuning stages:
    - 🟦 **Base Model:** Raw Qwen2.5-3B-Instruct (no YouTube-specific training)
    - 🟨 **+ SFT:** The model learns the format and style from 1,026 high-quality examples
    - 🟩 **+ DPO:** The model refines its style by learning to prefer engaging (chosen) titles over generic (rejected) ones

    👉 **Notice how titles become more engaging and click-worthy at each stage!**
    """)

    with gr.Row():
        video_idea_input = gr.Textbox(
            label="💡 Enter your Video Idea",
            placeholder="e.g., how to focus hard without any distractions",
            lines=2,
        )

    generate_btn = gr.Button("🚀 Generate Titles", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🟦 Base Model\n*No fine-tuning*")
            base_output = gr.Textbox(label="", interactive=False, lines=3)

        with gr.Column():
            gr.Markdown("### 🟨 + SFT\n*Learned format & style*")
            sft_output = gr.Textbox(label="", interactive=False, lines=3)

        with gr.Column():
            gr.Markdown("### 🟩 + DPO\n*Refined for engagement*")
            dpo_output = gr.Textbox(label="", interactive=False, lines=3)

    generate_btn.click(
        fn=generate_titles,
        inputs=video_idea_input,
        outputs=[base_output, sft_output, dpo_output],
    )

    gr.Examples(
        examples=[
            ["battery- phone - apple - protect"],
            ["This video is about how to focus hard without any distractions."],
            ["p-values. definition, examples, and misconceptions"],
        ],
        inputs=video_idea_input,
    )

if __name__ == "__main__":
    demo.launch()
