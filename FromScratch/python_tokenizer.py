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



















