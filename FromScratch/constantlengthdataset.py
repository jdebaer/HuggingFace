import torch
from torch.utils.data import IterableDataset

# At this point we have a custom tokenizer and an untrained model - how we can proceed to setting up the data loader infra.
# Important rule: we always want to fill up the complete seq_len of our model during training. So if the model has a seq_len of 1024
# we want to provide it with 1024 tokens for each training sample.
# There are 2 ways to deal with samples that are shorter:
# 1. Padding (which implies we need to mask the padding).
# 2. Concatenate samples, separate them with the '[EOS]' token, and then cut a contat off at "seq_len" tokens. This approach might waste
# a bit of data (from the cutting off) but if you have tons of data this might be a better approach since we won't be creating the extra
# load of padding and masking. So it's a compute vs. data constraint trade-off. If you are lower on data and good on compute, padding is the
# way to go.
# Note that if you concat ALL the samples and then split, you'll only lose a tiny piece of data at the end, no neglectible.

# HF dataset -> IterableDataset -> DataLoader

class ConstantLengthDataset(IterableDataset):
    
    def __init__(self, tokenizer, dataset, seq_len=1024, num_of_seqs=1014, chars_per_token=3.6):
    
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_len = seq_len
        
        self.num_of_input_chars = seq_len * chars_per_token * num_of_seqs

    def __iter__(self):
        
        iterator = iter(self.dataset)
        
        more_samples = True
        while more_samples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.num_of_input_chars:
                    break
                
                try:
                    buffer.append(next(iterator)['content'])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    more_samples = False


            # At this point we have added enough samples to the concat to exceed our input length or to
            # deplete the samples.
            # Now we tokenize our concat.

            all_token_ids = []

            buffer_encoding = self.tokenizer(buffer, truncation=False)

            # If you feed a list of strings in a HF tokenizer, you get a list of lists back, with each list containing  
            # the input ids for the corresponding string.
            for input_ids_list in buffer_encoding['input_ids']:
                all_token_ids.extend(input_ids_list + [self.concat_token_id])

            # At this point we have a long, flast list of input ids, with concat_token_id between each original sample.
            # Now let's cut it up.

            for i in range(0, len(all_token_ids), self.seq_len):
                seq_len_input_ids = all_token_ids[i : i + self.seq_len]

                if len(seq_len_input_ids) == self.seq_len:		# This will be true except probably for the last range.
                    yield torch.tensor(seq_len_input_ids)
            


                
         

            
      
