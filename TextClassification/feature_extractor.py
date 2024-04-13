# For explanations above the ##### section see preceding files.

from datasets import load_dataset
from transformers import DistilBertTokenizer
import datasets

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

model_ckpt = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

train 		= load_dataset('emotion', split='train[:1%]')
valid	 	= load_dataset('emotion', split='validation[:1%]')
test 		= load_dataset('emotion', split='test[:1%]')

#emotions = {'train':train, 'validation':valid, 'test':test}

emotions = datasets.DatasetDict({'train':train, 'validation':valid, 'test':test})

# The below return datasets or datasets in a list, but not in a dict.
#emotions = load_dataset("emotion", split=['train[:1%]'])
#emotions = load_dataset("emotion", split=['train[:1%]', 'validation[:1%]'])

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

###################### New #####################

# Feature extraction is done from the context vectors returned by the Encoder, which normally go to the Decoder for the cross attention.
# Our EncoderDecoder returns these layer-normalized via an extra layer normalization step after all the Encoder blocks have run. 

# Now we're ready to also load the model (we already loaded and used the model's tokenizer).

from transformers import AutoModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Embedding layer is part of the model, needs to be fed the input ids.
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "this is a test"

encoding = tokenizer(text, return_tensors="pt") 

print(f'Input tensor shape: {encoding["input_ids"].size()}')
# Input tensor shape: torch.Size([1, 6]) --> 6 ids, this is a test + start and end token.

print(encoding)
# encoding == {'input_ids': tensor([[ 101, 2023, 2003, 1037, 3231,  102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
# but at this point the tensors are not yet on the GPU, let's do that.

encoding = {k:v.to(device) for k,v in encoding.items()}

# Now let's push the through the model, which (via .model_input_names) we know requires ['input_ids', 'attention_mask']
with torch.no_grad():		  # This is because we are inferencing -> always disable gradient calc during inferencing to reduce RAM footprint.
    outputs = model(**encoding)   # ** takes a dict and transforms it into key=value input parameters, so that we get=
                                  # input_ids = the value for the key input_ids and same for attention_mask.
# Note: run .cpu() as soon as possible on anything returned by the model that's not going to be used anymore for gradient calculation, which is the
#       case here since we're inferencing.

# Type of outputs is BaseModelOutput. We get one context vector per id, and there is no causal masking going on so each context vector contains context
# coming from all the other words. For classification the convention is to do it with the [CLS] token (first token, stands for "classify").
print(outputs)
print(outputs.last_hidden_state.size())
# This is torch.Size([1, 6, 768]) and we want to first of the 6, so we run the below which give [1,768].

last_context_vector = outputs.last_hidden_state[:,0]

# Now we do the above for every sequence in the dataset i.e., we retrieve the last context vector, and batched (transparently).

def extract_last_context_vector(batch):						# batch is batch of dataset_encoded(s), which are results of tokenizing.
                                                                                # input_ids and attention_mask are added columns.
    encoding = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names} 	    # Convert dataset_encoded(s) to tensors and filter out
												    # only the input_ids and attention_mask columns.

    with torch.no_grad():
       last_context_vector = model(**encoding).last_hidden_state[:,0]

    return {'last_cv': last_context_vector.cpu().numpy()}
    
# The model wants tensors, so we need to make sure our batches of dataset_encoded(s) contain tensors, which is not the case yet.

emotions_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Now we are ready to "map" our function to this dataset.

emotions_last_cvs = emotions_encoded.map(extract_last_context_vector, batched=True)

print(emotions_last_cvs['train'].column_names)

# At this point we have the last context vector for each sequence. Now we can train a classifier on it.







