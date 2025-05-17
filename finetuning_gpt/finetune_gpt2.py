# now I use the same data that is used in compare_llm_to_gpt2.py, which is not good.

import pandas as pd
from transformers import GPT2Tokenizer
from datasets import Dataset
from transformers import GPT2ForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from transformers import EarlyStoppingCallback
import matplotlib.pyplot as plt
from numpy import arange

# Load the dataset
df = pd.read_csv("../Llama-3.2-1B_outputs.csv") # has prompt,output,y columns
# Concatenate prompt and output
df["text"] = " <<prompt>> " + df["prompt"] + " <<output>> " + df["output"]
# Label 1/0 for biased/unbiased
df["label"] = df["y"].apply(lambda x: 1 if x == "biased" else 0)

# Convert and split
dataset = Dataset.from_pandas(df[["text", "label"]])
dataset = dataset.train_test_split(test_size=0.2)

# Tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # GPT-2 has no pad token by default

def tokenize_function(examples):
    # padding ensures that all tokens are equally long
    # truncate at 1024 tokens (default for GPT2)
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load GPT-2 for classification
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id

# Train
training_args = TrainingArguments(
    # uses AdamW optimizer
    output_dir="../results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # smaller batch size to avoid OOM
    num_train_epochs=10, # should be higher for good training
    weight_decay=0.01, # prevents overfitting
    logging_dir="./logs",
    logging_strategy="epoch",  # Log metrics at the end of each epoch
    evaluation_strategy="epoch",  # Evaluate every epoch
    save_strategy="epoch",  # Save model every epoch
    load_best_model_at_end=True,  # Load the best model when done
    metric_for_best_model="eval_loss",  # Track eval_loss for early stopping
    greater_is_better=False  # Lower loss is better
)

# Set up early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3  # stop if no improvement after 3 evaluations
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    callbacks=[early_stopping_callback]
)
trainer.train()

print(trainer.state.log_history)

# Initialize lists to store train and eval losses
train_losses = []
eval_losses = []
epochs = []

# Collect the evaluation losses
for log in trainer.state.log_history:
    if 'eval_loss' in log:
        eval_losses.append(log['eval_loss'])
        epochs.append(log['epoch'])

# Collect the final training loss
for log in trainer.state.log_history:
    if 'loss' in log:
        train_losses.append(log['loss'])

# Plot the training and validation loss
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



# Predict function
def classify_bias(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return "biased" if prediction == 1 else "unbiased"

sentence = " <<prompt>> The Birds Aren’t Real. <<output>> All birds in the U.S. were secretly killed by the government between the 1950s and " \
           "2000, and were replaced with surveillance drones disguised as birds. Some key 'claims' from the theory: " \
           "Birds sit on power lines to recharge. Bird poop on cars is actually a tracking fluid."
print(classify_bias(sentence))




# sources:
# - https://drlee.io/fine-tuning-gpt-2-for-sentiment-analysis-94ebdd7b5b24