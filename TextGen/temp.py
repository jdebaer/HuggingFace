# Below is an implementation of beam search with n-gram penalty. This combo is commonly used for summarization and translation where we need to be
# factually correct. For story generation where we can make things up, we can also use sampling (see temp.py).

import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sequence_logprob import sequence_logprob


#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'
model_name = 'gpt2-xl'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
max_length = 10

# Below we're using generate() without beams which basically means greedy.
input_txt = 'Transformers are the most'
input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)

# no_repeat_ngram_size: tracks which n-grams have been seen and sets next id prob to zero if it would produce a previously seen n-gram.
#inference = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=2.0, top_k=0)
inference = model.generate(input_ids, max_length=max_length, do_sample=True, temperature=0.5, top_k=0)

print(tokenizer.decode(inference[0]))
print(sequence_logprob(model, inference, input_len=len(input_ids[0])))
    
