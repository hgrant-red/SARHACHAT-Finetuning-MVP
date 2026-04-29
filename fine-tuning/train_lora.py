# train_lora.py
import unsloth
import os
import torch
from training_hub import lora_sft

MODEL_NAME = "mistralai/Mistral-Small-24B-Instruct-2501" 
DATA_PATH = "data/sarhachat_training_data.jsonl"
OUTPUT_DIR = "./sarhachat-lora-vA-10epochs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"🚀 Launching Version A: High Repetition...")

lora_sft(
    model_path=MODEL_NAME, data_path=DATA_PATH, ckpt_output_dir=OUTPUT_DIR,
    lora_r=16, lora_alpha=32, lora_dropout=0.0, load_in_4bit=True,
    num_epochs=10,  
    learning_rate=2e-4, max_seq_len=2048, micro_batch_size=2, gradient_accumulation_steps=4,
    bf16=True, sample_packing=True, dataset_type="chat_template", field_messages="messages",
    logging_steps=2, save_steps=100,
)
print(f"✅ Version A saved to {OUTPUT_DIR}")