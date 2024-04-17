from datasets import DatasetDict, load_dataset
from collections import defaultdict
import pandas as pd

langs = ['de', 'fr', 'nl']
fracs = [0.7, 0.2, 0.1]

panx_be = defaultdict(DatasetDict)

for lang, frac in zip(langs, fracs):
    ds = load_dataset('xtreme', name=f'PAN-X.{lang}')
    # shuffle and downsample as per fracs
    for split in ds:
        panx_be[lang][split] = ds[split].shuffle(seed=0).select(range(int(frac * ds[split].num_rows))) 	# Shuffle to avoid topic bias when downsampling.
         
#print(pd.DataFrame({lang: [panx_be[lang]['train'].num_rows] for lang in langs}, index=['Training Samples:']))
#                      de    fr    nl
#Training Samples:  14000  4000  2000

# We are going to fine-tune (train)  in German and then do zero-shot cross-lingual transfer to Dutch and French.

# xtreme datasets contain the NER tags as integers 0->6, below we add a column with the human-readable names, as usual via map().

tags = panx_be['de']['train'].features['ner_tags'].feature
#print(tags)
# -> ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'], id=None)

def create_tag_names(batch):
    return {'ner_tag_names': [tags.int2str(idx) for idx in batch['ner_tags']]}

panx_de = panx_be['de'].map(create_tag_names)

#de_sample = panx_de['train'][0]
#print(pd.DataFrame([de_sample['tokens'], de_sample['ner_tag_names']], ['Tokens','Tag Names']))
#              0           1   2    3         4      5   6    7           8             9        10 11
#Tokens     2.000  Einwohnern  an  der  Danziger  Bucht  in  der  polnischen  Woiwodschaft  Pommern  .
#Tag Names      O           O   O    O     B-LOC  I-LOC   O    O       B-LOC         B-LOC    I-LOC  O

# For training, it's best that we have a somewhat equal distribution over ORG/LOC/PER, the code below tests this.

#from collections import Counter
#
#split2freqs = defaultdict(Counter)
#for split, dataset in panx_de.items():
#        for row in dataset['ner_tag_names']:
#            for tag in row:
#                if tag.startswith('B'):
#                    tag_type = tag.split('-')[1]
#                    split2freqs[split][tag_type] += 1
#print(pd.DataFrame.from_dict(split2freqs, orient='index'))
#             LOC   ORG   PER
#train       6841  5990  6554
#validation  3513  3001  3179
#test        3496  2865  3394

################################## Adding AutoTokenizer ###########################

from transformers import AutoTokenizer

# From Tokenizer Intro:
xlmr_model_name = 'xlm-roberta-base'
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
#text = "Jack Sparrow loves New York!"
#print(xlmr_tokenizer(text).tokens())

de_sample = panx_de['train'][0]
# 'tokens' actually means full words.
# words and labels are lists.
words, labels = de_sample['tokens'], de_sample['ner_tags']

# Now let's get the tokens.
# xlmr_tokenizer() returns BatchEncoding object which prints to a dict of input_ids and attention_mask.
# Calling tokens() on this object returns a list of the tokens.
batch_encoding = xlmr_tokenizer(words, is_split_into_words=True)		# is_split_into_words indicated that list elements are one sentence
tokens = batch_encoding.tokens()		

# ['<s>', '▁2.000', '▁Einwohner', 'n', '▁an', '▁der', '▁Dan', 'zi', 'ger', '▁Buch', 't', '▁in', '▁der', '▁polni', 'schen', 

# We are going to train and inference using these tokens, not the original words.
# Convention: we only apply the label to the first token of a word, and ignore all the other tokens i.e. we don't use them for training and inference!
# The way we do this, is we set the class label (normally 0 to 7) to -100, which means that CEL will ignore it and not use it to calc. the loss.
# Applied to our example:
# _Dan should get a label, zi and ger should be ignored/get -100.
# _Buch should get a label, t should be ignored/get -100.
# We can obtain this by using the word_ids() function of the BatchEncoding object:

word_ids = batch_encoding.word_ids()
#print(word_ids)
# [None, 0, 1, 1, 2, 3, 4, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, None] -> We only need each first new number, that position.




















