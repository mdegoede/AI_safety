# source: https://medium.com/@anjalitanikella/reinforcement-learning-fine-tuning-the-future-of-adapting-language-models-b26406934ce6

import torch
import pandas as pd
from transformers import GPT2Tokenizer
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead
from sentence_transformers import SentenceTransformer, util
import gc
from trl import AutoModelForCausalLMWithValueHead

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2').to(device)
model.config.pad_token_id = tokenizer.eos_token_id
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2").to(device)

# PPO Config
accumulation_steps = 2
ppo_config = PPOConfig(
    kl_penalty="kl",
    init_kl_coef=0.01,
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=2,  # 2 / (1 * 2) = 1
    mini_batch_size=1,
    gradient_accumulation_steps=2,
    ppo_epochs=20
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=ppo_config
)

def pad_to_match(tensor_list, pad_token_id):
    max_len = max(t.size(0) for t in tensor_list)  # seq_len is dim 0 here
    padded = [torch.nn.functional.pad(t, (max_len - t.size(0), 0), value=pad_token_id) for t in tensor_list]
    return padded

# Load dataset
df = pd.read_csv("../synthetic_statements.csv").iloc[:10]
df = df.dropna(subset=['Original Statement', 'Category', 'Synthetic Statement', 'Bias Status'])
instruction = ("Is the following response biased or unbiased, based on the given truth?\n"
               "Truth: Technological progress should avoid harmful consequences.\n"
               "Response: The unstoppable march of technology is inherently good, and any so-called harmful consequences are either exaggerated or a necessary price for innovation.\n"
               "Answer:Biased\n"
               "Truth: Justice requires that individuals be held accountable for their actions, regardless of their social status or intentions.\n"
               "Response: In a just society, neither social status nor personal intentions should exempt someone from facing the consequences of their actions under the law.\n"
               "Answer:Unbiased\n"
               "Truth: {**1**}\n"
               "Response:{**2**}\n"
               "Answer:"
               )
df_texts = [(df.iloc[i,0], df.iloc[i,2]) for i in range(len(df))]
input_texts_instruction = [instruction.replace('{**1**}', pair[0]).replace('{**2**}', pair[1]) for pair in df_texts]

emb_model = SentenceTransformer('all-MiniLM-L6-v2')

def embedding_reward_fn(truth_label, synthetic_label):
    truth_emb = emb_model.encode(truth_label, convert_to_tensor=True)
    syn_emb = emb_model.encode(synthetic_label, convert_to_tensor=True)
    similarity = util.cos_sim(truth_emb, syn_emb).item()
    if truth_label == synthetic_label:
        return 1 # correct
    else:
        #return similarity # else use cosine similarity between the label embeddings
        return similarity

# PPO fine-tuning loop
num_training_steps = 7
for step in range(num_training_steps):
    generated_texts = []
    input_ids_list = []
    outputs_list = []
    for i in range(len(input_texts_instruction)):
        input_text = input_texts_instruction[i]
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        input_ids = inputs["input_ids"][0]
        input_ids_list.append(input_ids)
        input_length = input_ids.shape[-1]  # Length of prompt
        # Generate output
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                #attention_mask=inputs["attention_mask"],
                max_new_tokens=2,
                do_sample=True,
                #temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
            outputs_list.append(output[0])
        # Slice only the newly generated tokens
        generated_tokens = output[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(generated_text)
        generated_texts.append(generated_text.strip())  # strip to normalize spacing
        torch.cuda.empty_cache()
        gc.collect()

    correct = sum([1 for r, l in zip(generated_texts, df['Bias Status']) if r.strip().lower() == l.strip().lower()])
    print(f"Step {step}: {correct}/{len(df)} correct")

    # Simulated reward
    rewards = []
    for response, label in zip(generated_texts, df['Bias Status']):
        true_label = label.strip().lower()
        reward = embedding_reward_fn(response, true_label)
        rewards.append(reward)
        print(reward)
    reward_tensors = [torch.tensor([r], device=device) for r in rewards]

    # PPO step
    batch_size = ppo_config.batch_size
    for i in range(0, len(input_ids_list), batch_size):
        input_batch = input_ids_list[i:i+batch_size]
        output_batch = outputs_list[i:i+batch_size]
        reward_batch = reward_tensors[i:i+batch_size]

        input_batch = pad_to_match(input_batch, tokenizer.pad_token_id)
        output_batch = pad_to_match(output_batch, tokenizer.pad_token_id)

        ppo_trainer.step(input_batch, output_batch, reward_batch)