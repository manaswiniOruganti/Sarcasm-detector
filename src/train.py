from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("tweet_eval", "irony")

# Convert to pandas
train_df = dataset['train'].to_pandas()
val_df = dataset['validation'].to_pandas()
test_df = dataset['test'].to_pandas()

print(train_df.head())
print(train_df['label'].value_counts())
from datasets import Dataset

label_names = ['non_sarcastic', 'sarcastic']

def format_example(row):
    label = label_names[row['label']]
    return f"""<|im_start|>system
Detect sarcasm. Answer ONLY: sarcastic or non_sarcastic<|im_end|>
<|im_start|>user
{row['text']}<|im_end|>
<|im_start|>assistant
{label}<|im_end|>"""

# Convert dataframe → formatted text
train_texts = train_df.apply(format_example, axis=1).tolist()

# Convert to HuggingFace dataset
train_dataset = Dataset.from_dict({"text": train_texts})

print("\nSample formatted data:\n")
print(train_texts[0])
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

print("\nLoading Qwen model...")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Training config
training_args = SFTConfig(
    output_dir="models/",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=20,
    save_steps=200,
    max_length=256,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    processing_class=tokenizer
)

print("\n🚀 Training started...\n")
trainer.train()

# Save model
model.save_pretrained("models/final")
tokenizer.save_pretrained("models/final")

print("\n✅ Training complete. Model saved in models/final")