from transformers import AutoTokenizer
from datasets import load_dataset
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from tqdm.auto import tqdm

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

tokenizer.save_pretrained("tokenizers/" + model_ckpt)
