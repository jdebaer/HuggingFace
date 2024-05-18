from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from constantlengthdataset import ConstantLengthDataset

model_ckpt = 'gpt2'

# Take tokenizer from the model and retrain it.
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)				
byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v,k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())

length = 2000
dataset_name = 'transformersbook/codeparrot-train'
dataset = load_dataset(dataset_name, split='train', streaming=True)
iter_dataset = iter(dataset)

def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)['content'] for _ in range(batch_size)]

tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=32768, initial_alphabet=base_vocab)

##### Code above is all that's needed to retrain HF tokenizer. python_tokenizer.py has all the details and comments.

################################# Train a HF model from scratch ################################

from transformers import AutoConfig, AutoModelForCausalLM

# Note: if you would have saved the retrained tokenizer to the hub, it would have a ckpt name and you would load it like this:
# tokenizer = AutoTokenizer.from_pretrained(model_cpkt)  	# With model_ckpt being the name you used when saving your retrained tokenizer.

# This would be a different model_ckpt than the one you used to save the retrained tokenizer.
# In our case this is gpt2 as per the above, so the untrained model we're going to load to then retrain.

# This is how you load an untrained model from the hub when you just have make some config changes (in this case the vocab size).
config = AutoConfig.from_pretrained(model_ckpt, vocab_size=len(tokenizer))		
model = AutoModelForCausalLM.from_config(config)

# Saving a model to a file.
model.save_pretrained("models/" + model_ckpt, push_to_hub=False, organization="transformersbook")

# Estimate how many Unicode characters we typically have per token, based on 500 samples.
samples, total_characters, total_tokens = 500, 0, 0
dataset_for_sample = load_dataset('transformersbook/codeparrot-train', split='train', streaming=True)
for _,sample in tqdm(zip(range(samples), iter(dataset)), total=samples):
    total_characters += len(sample['content'])
    total_tokens += len(tokenizer(sample['content']).tokens())
characters_per_token = total_characters / total_tokens

## Run a test with characters_per_token in our call to ConstantLengthDataset.
#shuffled_dataset = dataset.shuffle(buffer_size=100)
#constant_length_dataset = ConstantLengthDataset(tokenizer, shuffled_dataset, num_of_seqs=10, chars_per_token=characters_per_token, seq_len=1014)
#dataset_iterator = iter(constant_length_dataset)
#lengths = [len(b) for _,b in zip(range(5), dataset_iterator)]
#print(lengths)



















