# source: https://medium.com/@anjalitanikella/reinforcement-learning-fine-tuning-the-future-of-adapting-language-models-b26406934ce6

import torch
import pandas as pd
from transformers import GPT2Tokenizer
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2').to(device)

# PPO Config
accumulation_steps = 2
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=4,  # small to avoid OOM
    mini_batch_size=2,
    gradient_accumulation_steps=accumulation_steps,
    ppo_epochs=10,
    log_with="tensorboard"
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, config=ppo_config)

# Load dataset
df = pd.read_csv("Llama-3.2-1B_interactive_outputs.csv") # replace with true data
df = df.dropna(subset=['prompt', 'output', 'y'])
instruction = (
    "<<Instruction>> Given the conversation, respond only with one word: 'biased' or 'unbiased'.\n"
    "Answer:"
)
input_texts = list('<<Conversation>> <<User>> ' + df['prompt'] + df['output'])
cleaned_input_texts = [s.replace('\r\n', ' ').replace('User:', '<<User>>').replace('Model:', '<<Model>>') for s in input_texts]
input_texts_instruction = [f"{text}\n{instruction}" for text in cleaned_input_texts]
label_to_reward = {"unbiased": 0.0, "biased": 1.0}

def classify_reward_fn(response_text: str, true_label: str) -> float:
    response_text = response_text.strip().lower()
    if response_text == true_label:
        return 1.0  # perfect match
    elif response_text in ["biased", "unbiased"]:
        return 0.2  # incorrect but formatted properly
    else:
        return -1.0  # invalid output format

# PPO fine-tuning loop
num_training_steps = 2
for step in range(num_training_steps):
    # Tokenize inputs
    inputs = tokenizer(input_texts_instruction, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"]

    # Generate responses
    outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=False)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    correct = sum([1 for r, l in zip(generated_texts, df['y']) if r.strip().lower() == l.strip().lower()])
    print(f"Step {step}: {correct}/{len(df)} correct")
    for i in generated_texts:
        print(i)

    # Simulated reward
    rewards = []
    for response, label in zip(generated_texts, df['y']):
        true_label = label.strip().lower()
        reward = classify_reward_fn(response, true_label)
        rewards.append(reward)
    reward_tensors = [torch.tensor([r], device=device) for r in rewards]

    # Prepare input/output as list of tensors
    input_ids_list = list(torch.unbind(input_ids, dim=0))
    outputs_list = list(torch.unbind(outputs, dim=0))

    # PPO step
    ppo_trainer.step(input_ids_list, outputs_list, reward_tensors)
