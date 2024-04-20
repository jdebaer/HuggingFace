import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model_name = 'gpt2-xl'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

import pandas as pd

input_txt = 'Transformers are the'
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
iterations = []
n_steps = 8
choices_per_step = 5

with torch.no_grad():
    for _ in range(n_steps):
        iteration = dict()
        iteration['input'] = tokenizer.decode(input_ids[0])

        # Generate the next input id.
        output = model(input_ids = input_ids)

        # output is (batch_size, seq_len, embed_size) and we only need the very last (complete, so :) context vector to predict the next id.
        next_id_logits = ouput.logits[0, -1, :]

        next_id_probs = torch.softmax(next_id_logits, dim=-1)

        # We want to be able to select any id above some probability threshold, so let's sort them. 
        sorted_ids = torch.argsort(next_id_probs, dim=-1, descending=True)

        for idx in range(choices_per_step):
            id = sorted_ids[idx]
            prob = next_id_probs[id].cpu().numpy()

            token_choice = (
                f'{tokenizer.decode(token_id)} ({100 * prob:.2f}%)'	
            )
            iteration[f'Choice {choice_idx+1}'] = token_choice

        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim = -1)
        iterations.append(iteration)

print(pd.DataFrame(iterations))


            

        



