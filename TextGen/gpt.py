import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'
model_name = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
n_steps = 80

# Below we're using generate() without beams which basically means greedy.
input_txt = 'Transformers are the most'
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))

exit(0)

# Below is a manual implementation of greedy.



import pandas as pd

input_txt = 'Transformers are the'
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
iterations = []
choices_per_step = 5

with torch.no_grad():
    for _ in range(n_steps):
        iteration = dict()
        iteration['input'] = tokenizer.decode(input_ids[0])

        # Generate the next input id.
        # Type of output is CausalLMOutputWithCrossAttentions.
        # CausalLMOutputWithCrossAttentions.logits is (batch_size, sequence_length, config.vocab_size)
        output = model(input_ids = input_ids)

        # output is (batch_size, seq_len, embed_size) and we only need the very last (complete, so :) context vector to predict the next id.
        next_id_logits = output.logits[0, -1, :]

        #print(next_id_logits.size())
        #torch.Size([50257])

        next_id_probs = torch.softmax(next_id_logits, dim=-1)
        #torch.Size([50257])

        # We want to be able to select any id above some probability threshold, so let's sort them. 
        sorted_ids = torch.argsort(next_id_probs, dim=-1, descending=True)

        for idx in range(choices_per_step):
            id = sorted_ids[idx]
            prob = next_id_probs[id].cpu().numpy()

            token_choice = (
                f'{tokenizer.decode(id)} ({100 * prob:.2f}%)'	
            )
            iteration[f'Choice {idx+1}'] = token_choice
        
        # print(sorted_ids.size())
        # torch.Size([50257])

        # print(input_ids.size())
        # torch.Size([1, 4]) where the '4' is the dim that keeps growing.

        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim = -1)
        # input_ids has 2 dims.
        # sorted_ids has 1 dim.
        # sorted_ids[None,0] has 1 dim and only has the first element from sorted_ids.
        # sorted_ids[None, 0, None] has 2 dims.        

        iterations.append(iteration)

print(pd.DataFrame(iterations))


            

        



