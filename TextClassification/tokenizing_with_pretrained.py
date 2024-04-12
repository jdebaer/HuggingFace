# Encoder-only example.

from datasets import load_dataset

emotions = load_dataset("emotion")

#from transformers import AutoTokenizer
#
#model_ckpt = "distilbert-base-uncased"
#tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
#
#Equivalent:
from transformers import DistilBertTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

text = "Tokenizing text is a core task of NLP."

# Note: when using pre-trained models, it's mandatory to use the exact same tokenizers that the model was trained with.
# Switching the tokenizer means that the id for "cat" can now become the id for "house", so garbage text will come out.

encoding = tokenizer(text)
print(encoding)

tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids)
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))

# Find out what we need to provide the pre-trained model with.
print(tokenizer.model_input_names)

# Now we use the tokenizer on the emotions dataset.
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


print(emotions['train'][:2])
print(tokenize(emotions['train'][:2]))

# Now we use map() to apply this function to the whole dataset, thereby adding the 'input_ids' and 'attention_mask' columns to it.
# batch_size = None -> this means we process the whole dataset. We need to do this otherwise outputs for different batches might have
# different shapes for the input tensors (input_ids) and/or attention_masks (diff. in shape would be in seq_len).
# For each batch_size, the sequences get padded to the max seq_len in the batch. Attention_mask masks out the padding tokens.
# Note: at this point it's all still lists, no tensors yet.

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)			

print(emotions_encoded['train']['input_ids'][0])









