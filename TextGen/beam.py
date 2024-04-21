# Below is an implementation of beam search with n-gram penalty. This combo is commonly used for summarization and translation where we need to be
# factually correct. For story generation where we can make things up, we can also use sampling (see sampling.py).

import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
# 
# We're creating a function that will show the log probability of a generated sequence.
# For this we need the sequence, the model, and the length of the prompt. Starting from the first word after the prompt, we're going to use the model
# to give us the logits. Then we're going to convert those to log probs and via the given sequence we'll select the one that was generated in the 
# beam. We'll add up the log probs to come to the log prob of the whole thing.

# Note that log prob gives a neg number, and the higher (closer to 0) the better.

def sequence_logprob(model, beam, input_len=0):
    with torch.no_grad():
        output = model(beam)

        # We actually calcualte the log probs for every inference staring with the second id. We trim based on input_len later.
        # For the labels, we're never going to compare with the first id, so we do a shift left by not including the first id.
        # lables is (batch_size,lenght of the beam sequence - 1)
        labels = beam[:, 1:]
     

        # For the logits we are good with starting with the first one since this is the inference corresponding to the second and now first id.
        # However, it also contains the logits for the new/next id for which we haven't done the conversion to id yet, so we don't include that one.
        # logits is (batch_size, length of the beam sequence - 1, vocab_size)
        logits = output.logits[:,:-1, :]

        # Dim is (batch_size, length of beam sequence, vocab_size) as we're just doing softmax.
        logprobs = F.log_softmax(logits, -1)
         
        # Now we need to retrieve the specific log probability of the id that was selected, for each id in the sequence and still over all batches.
        # Baiscally this means we have to go into each 'row' in third dimension '2' and fetch the lob probability that sits in the position of the id that is in the label.
        # We do this with torch.gather().
        # This requires a tensor with the exact same dimensions are the input tensor, but in the dimension of the row there must be just one element,
        # which is the position of the value that need to be fetched (our class labe).
        # With this unsqueeze, each element in dim 1 is turned into a 1-element long new dimension at level 2.

        filter = labels.unsqueeze(2)

        logprob_gathered_3dim = torch.gather(logprobs, 2, filter)

        # This is now a torch.Size([1, 9, 1]) which we want to turn into a torch.Size([1, 9]), and in dim 1 each position is the gathered log prob for the id.

        logprob_gathered = logprob_gathered_3dim.squeeze(-1)
        
        # Now we just sum up the log probs for all ids to get the log prob for the whole beam.
        # We do this over all beam in the batch, and we don't include the ids that were provided as prompt.
        beam_log_prob = torch.sum(logprob_gathered[:, input_len:])
        return beam_log_prob.cpu().numpy()

# no_repeat_ngram_size: tracks which n-grams have been seen and sets next id prob to zero if it would produce a previously seen n-gram.
beam = model.generate(input_ids, max_length=max_length, num_beams=2, do_sample=False, no_repeat_ngram_size=2)

print(tokenizer.decode(beam[0]))
print(sequence_logprob(model, beam, input_len=len(input_ids[0])))
    
