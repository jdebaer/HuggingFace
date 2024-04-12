# For explanations above the ##### section see preceding files.

from datasets import load_dataset
from transformers import DistilBertTokenizer

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

model_ckpt = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

emotions = load_dataset("emotion")

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




