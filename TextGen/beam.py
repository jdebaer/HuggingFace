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

# HF Transformer 'For' models (so with head) always return the unnormalized logits for the next id given the input ids, over the vocab size.
 
# no_repeat_ngram_size: tracks which n-grams have been seen and sets next id prob to zero if it would produce a previously seen n-gram.
beam = model.generate(input_ids, max_length=max_length, num_beams=2, do_sample=False, no_repeat_ngram_size=2)

print(tokenizer.decode(beam[0]))
print(sequence_logprob(model, beam, input_len=len(input_ids[0])))
    
