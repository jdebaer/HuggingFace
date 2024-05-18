from transformers import AutoTokenizer

python_code = r"""def say_hello():
    print("Hello World")
say_hello()
"""

# GPT-2 uses a byte-level tokenizer. This means that each byte is a basic unit for BPE. BPE will create a vocabulary by 
# progressively creating new tokens formed by merging the most frequently co-occurring basic units (bytes) and adding them
# to the vocabulary.
# There are 256 bytes to start with and the vocab size will grow until the limit that we set.


tokenizer = AutoTokenizer.from_pretrained('gpt2')
#print(tokenizer(python_code).tokens())

# Normalization

#print(tokenizer.backend_tokenizer.normalizer)
# This gives 'None': no normalization done.

# Pretokenization

#print(tokenizer.backend_tokenizer.pre_tokenizer)
## This gives "pointer" so there is one.

#print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(python_code))
#[('def', (0, 3)), ('Ġsay', (3, 7)), ('_', (7, 8)), ('hello', (8, 13)), ('():', (13, 16)), ('ĊĠĠĠ', (16, 20)), ('Ġprint', (20, 26)), ('("', (26, 28)), ('Hello', (28, 33)), ('ĠWorld', (33, 39)), ('")', (39, 41)), ('Ċ', (41, 42)), ('say', (42, 45)), ('_', (45, 46)), ('hello', (46, 51)), ('()', (51, 53)), ('Ċ', (53, 54))]
# The pretokenizer maps bytes that represent non-printable characters to Unicode characters, e.g.:
# \r == byte 13 == not shown here because we're on Linux
# \n == byte 10 == Ċ
# space == byte 32 == Ġ
# etc.
# This is because BPE algorithms want everything to be fed in as Unicode without spaces or control characters, but in this case
# we cannot do that because spaces in Python are important (indentation) so "a space" must become a character in Unicode. We say
# character here and not word because BPE starts from Unicode characters and builds up more tokens from there. Examples of tokens
# we would expecte for Python are "2 spaces" or "4 spaces" or "tab" because this sequence occurs a lot.

# However, from running the complete tokenizer pipeline as below, we can see that this is not the case.
# We can retrain the tokenizer though.
#print(tokenizer(python_code).tokens())
#['def', 'Ġsay', '_', 'hello', '():', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġprint', '("', 'Hello', 'ĠWorld', '")', 'Ċ', 'say', '_', 'hello', '()', 'Ċ']
# What we see is that 4 spaces are (mostly) separate tokens. 

## Other useful things to look at when looking into a tokenizer:
## 1. What are the longest "composed" tokens
## List of tuples.
#tokens = sorted(tokenizer.vocab.items(), key=lambda x: len(x[0]), reverse=True)
## This is a list of tuples (<token>,<id>).
#print([f'{tokenizer.decode(t)}' for _,t in tokens[:10]])
#
## 2. What are the last words added to the vocabulary, so the least frequent ones. These are the tokens with the highest id numbers.
#tokens = sorted(tokenizer.vocab.items(), key = lambda x: x[1], reverse=True)
#print([f'{tokenizer.decode(t)}' for _,t in tokens[:10]])



#################### Retraining a tokenizer #######################

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v,k) for k, v in byte_to_unicode_map.items())
# The below is the base vocabulary for any byte-based tokenizer. Should work for any programming language.
base_vocab = list(unicode_to_byte_map.keys())



length = 1000
dataset_name = 'transformersbook/codeparrot-train'
dataset = load_dataset(dataset_name, split='train', streaming=True)
iter_dataset = iter(dataset)

def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)['content'] for _ in range(batch_size)]

# Vocab sizes that are a multiple of 8 is better for certain GPU/TPU computations - look into this.
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=12496, initial_alphabet=base_vocab)

# This first 256 tokens will be our base vocab, to which we will have added new tokens during training (each token based on combining other tokens as per BPE).
# Let's see what the first added tokens are.

# new_tokenizer.vocab is a dict where each key:value is 'abilities': 8440 so <token>:<id>.
# Using items() on it returns these key:value pairs as tuples.

#tokens = sorted(new_tokenizer.vocab.items(), key=lambda x: x[1], reverse=False)
#print([f'{new_tokenizer.decode(t)}' for _,t in tokens[257:280]])
## This already contains combinations of white space as tokens.
##['  ', '    ', '   ', '\n    ', 'se', 're', 'in', 'on', 'te', '\n       ', '        ', '\n   ', 'st', 'or', 'de', 'le', 'th', ' =', 'lf', 'al', 'self', 'me', 'ti']
#
## Same for last added words.
#print([f'{new_tokenizer.decode(t)}' for _,t in tokens[-10:]])

# Using our retrained tokenizer for a test.
print(new_tokenizer(python_code).tokens())

# When creating a tokenizer for a coding language, at least all the reserved key words should be full tokens in the vocab, meaning they should not have to
# be created from multiple tokens.

print(new_tokenizer.vocab)

# For Python, it's easy to get all the keywords. For another language we'll have to get them some other way.
import keyword

for keyword in keyword.kwlist:
    if keyword not in new_tokenizer.vocab:
        print(f'Keyword "{keyword}" is not in the vocabulary.')
    
# Ìf frequent keywords are missing, then use more data to train and increase the vocab size and length.

length = 2000
new_tokenizer_larger = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=32768, initial_alphabet=base_vocab)

# Compating the efficiency of the retrained tokenizer with the standard tokenizer we started with (coming from GPT-2) can be done as follows.
# Feed them the same python code, and compare the amount of tokens needed to represent it. The less tokens needed, the more efficient the tokenizer.
# Let's do the test.

# These are lists.
print(len(tokenizer(python_code).tokens()))				# 20
print(len(new_tokenizer(python_code).tokens()))				# 19
print(len(new_tokenizer_larger(python_code).tokens()))			# 18 

# Note that this has an impact on the context window, which is in tokens/ids, and NOT in "words". 
# Example: traininig a model with the new tokenizer using a CW of 1024 captures the same amount of context as training a model with the old tokenizer
# with a CW of 2048. This is because for the same amount of pure text, the old tokenizer needs twice as many tokens. This means we can train our model
# much more efficiently (faster and less RAM requirements).

print(type(tokenizer("this is a test")))



















