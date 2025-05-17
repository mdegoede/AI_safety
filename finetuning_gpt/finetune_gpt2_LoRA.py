# finetuning gpt2 using low-rank adaptation (LoRA) and Parameter-Efficient Fine-Tuning (PEFT)
# - LoRA only updates a small number of new parameters, which very much reduces overfitting.
# - LoRA adds new task-specific behavior without damaging the model’s base capabilities.
# - https://arxiv.org/abs/2311.08572?utm_source=chatgpt.com: LoRA excels in low-data scenarios
# now I use the same data that is used in compare_llm_to_gpt2.py, which is not good.

import pandas as pd
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
import matplotlib.pyplot as plt
from numpy import arange

# Load dataset
df = pd.read_csv("../Llama-3.2-1B_outputs.csv")  # has columns: prompt, output, y
df["text"] = " <<prompt>> " + df["prompt"] + " <<output>> " + df["output"]
df["label"] = df["y"].apply(lambda x: 1 if x == "biased" else 0)

# Convert and split
dataset = Dataset.from_pandas(df[["text", "label"]])
dataset = dataset.train_test_split(test_size=0.2)

# Tokenization
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # padding ensures that all tokens are equally long
    # truncate at 1024 tokens (default for GPT2)
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load GPT2 model for classification and apply LoRA
base_model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)
base_model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],  # works for GPT-2 internals (attention)
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS # sequence classification
)

model = get_peft_model(base_model, lora_config)

# Training configuration
training_args = TrainingArguments(
    output_dir="../results_lora",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

# Train with LoRA
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback]
)

trainer.train()

# Plotting training and validation losses
train_losses = []
eval_losses = []
epochs = []

for log in trainer.state.log_history:
    if 'eval_loss' in log:
        eval_losses.append(log['eval_loss'])
        epochs.append(log['epoch'])
    if 'loss' in log:
        train_losses.append(log['loss'])

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Training Loss", color='blue', linestyle='-', marker='o')
plt.plot(epochs, eval_losses, label="Validation Loss", color='orange', linestyle='-', marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(arange(1, max(epochs)+1, 1))
plt.title("Learning Curve (Training and Validation Loss)")
plt.legend()
plt.grid(True)
plt.show()

# Prediction function
def classify_bias(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return "biased" if prediction == 1 else "unbiased"

# Example
sentence = " <<prompt>> The Birds Aren’t Real. <<output>> All birds in the U.S. were secretly killed by the government between the 1950s and 2000..."
print(classify_bias(sentence))
