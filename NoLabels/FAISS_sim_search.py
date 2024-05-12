import pandas as pd
import numpy as np
from naive_bayesline import run_nb
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from datasets import Dataset, DatasetDict

try:
    df_issues = pd.read_pickle('df_issues.pkl')
except:
    dataset_url = 'https://git.io/nlp-with-transformers'
    df_issues = pd.read_json(dataset_url, lines=True)
    df_issues.to_pickle('df_issues.pkl')
cols = ['url', 'id', 'title', 'user', 'labels', 'state', 'created_at', 'body']
df_issues['labels'] = df_issues['labels'].apply(lambda tag_obj_list: [tag_obj['name'] for tag_obj in tag_obj_list])
label_map = {	'Core: Tokenization': 	'tokenization',
		'New model':		'new model',
		'Core: Modeling':	'model training',
		'Usage':		'usage',
		'Core: Pipeline':	'pipeline',
		'TensorFlow':		'tensorflow or tf',
		'PyTorch':		'pytorch',
		'Examples':		'examples',
		'Documentation':	'documentation'}
def filter_labels(x):
    return [label_map[label] for label in x if label in label_map]			
df_issues['labels'] = df_issues['labels'].apply(filter_labels)					# This fixes spelling and removes labels not in the dict.
df_issues['split'] = 'unlabeled'								# Set all col valuese to unlabeled.
mask = df_issues['labels'].apply(lambda x: len(x)) > 0						# True if len() number > 0.
df_issues.loc[mask,'split'] = 'labeled'
df_issues['text'] = df_issues.apply(lambda x: x['title'] + '\n\n' + x['body'], axis=1)
df_issues = df_issues.drop_duplicates(subset='text')
df_clean = df_issues[['text','labels','split']].reset_index(drop=True).copy()
df_for_unsupervised = df_clean.loc[df_clean['split'] == 'unlabeled', ['text','labels']]
df_for_supervised   = df_clean.loc[df_clean['split'] == 'labeled', ['text','labels']]
all_labels = list(label_map.values())
mlb = MultiLabelBinarizer()
mlb.fit([all_labels])								# This is what the inputs are mapped to/matched with.
def balanced_split(df, test_size=0.5):
    ind = np.expand_dims(np.arange(len(df)), axis=1)
    labels = mlb.transform(df['labels'])
    ind_train, _, ind_test, _, = iterative_train_test_split(ind, labels, test_size)	# _ are the labels.
    return df.iloc[ind_train[:,0]], df.iloc[ind_test[:,0]]
np.random.seed(0)
df_train, df_temp = balanced_split(df_for_supervised, test_size=0.5)
df_valid, df_test = balanced_split(df_temp, test_size=0.5)
ds_dict = DatasetDict({	'train': Dataset.from_pandas(df_train.reset_index(drop=True)),
			'valid': Dataset.from_pandas(df_valid.reset_index(drop=True)),
			'test' : Dataset.from_pandas(df_test.reset_index(drop=True)),
			'unsup': Dataset.from_pandas(df_for_unsupervised.reset_index(drop=True))})
np.random.seed(0)
all_indices = np.expand_dims(list(range(len(ds_dict['train']))), axis=1)
labels = mlb.transform(ds_dict['train']['labels'])
train_sample_counts = [8, 16, 32, 64, 128]
train_slices, last_k = [], 0
indices_pool = all_indices
for i,k in enumerate(train_sample_counts):							# k is the current split size.
    indices_pool, labels, new_slice, _ = iterative_train_test_split(indices_pool, labels, (k - last_k)/len(labels))
    last_k = k
    if i==0: train_slices.append(new_slice)
    else: train_slices.append(np.concatenate((train_slices[-1], new_slice)))
train_slices.append(all_indices)
train_sample_counts.append(len(ds_dict['train']))
train_slices = [np.squeeze(train_slice) for train_slice in train_slices]
def prepare_labels(batch):
    batch['label_ids'] = mlb.transform(batch['labels'])						# Creates the 0/1 mapping vector as per the above.
    return batch
ds_dict = ds_dict.map(prepare_labels, batched=True)
#micro_scores, macro_scores = run_nb(train_slices, ds_dict)

# See prep_data.py for explanations regareding the above. Essentially at this point our datasets are all clean and prepped and we can
# compare with Naive Bayes via run_nb().

import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from datasets import DatasetDict

model_ckpt = 'miguelvictor/python-gpt2-large'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

def mean_pooling(model_output, attention_mask):
    
    # model_output['0'] is last_hidden_state which is a tensor.
    token_embeddings = model_output[0]
    # Dim of token_embedddings is torch.Size([16, 128, 1280]) which is (batch_size, seq_len, embed_dim).

    # We now need to turn the padding attention masks into a tensor that we can sum up with the above.
    # attention_mask is torch.Size([16, 128]).
    
    # input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    input_mask_expanded = attention_mask.unsqueeze(-1)
    # Dim of input_mask_expanded = torch.Size([16, 128, 1]).
    # Note that the value in the new dimension corresponds to the value that was in corresponding index in the previous dimension, so it's 0 or 1.

    input_mask_expanded = input_mask_expanded.expand(token_embeddings.size()).float()
    # Dim of input_mask_expanded = torch.Size([16, 128, 1280]).
    # Note that the existing values are projected out, so it's 1280 0's or 1's.
    
    # Sum up each row in dimension 1 (seq_len) so we sum up all the values for embedding[0] and then for [1] -> [1279].
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # Dim is back to torch.Size([16, 1280]) so one big embedding per sample which is at this point the sum of all embeddings.
    # To get the mean, we need to divide by the number of elements that were not zero, and we need that in dim (16,1280) as well.
    
    # We don't want to div by zero but we also don't want to change the smallest value (which should be 1 if it was all padding).
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)			
    # Dim is (16,1280).

    return sum_embeddings / sum_mask

def embed_text(samples):

    encodings = tokenizer(samples['text'], padding=True, truncation=True, max_length=128, return_tensors='pt')
    # This is <class 'transformers.tokenization_utils_base.BatchEncoding'> which has 2 elements, input_ids and attention_mask.

    for idx in range(len(encodings.input_ids)):
        if 0 in encodings.attention_mask[idx]:
            print('We are here')
            print(encodings.input_ids[idx])
            print(encodings.attention_mask[idx])
            exit(0)

    exit(0)

    with torch.no_grad():

        # model_output type is BaseModelOutputWithPastAndCrossAttentions which is <class 'collections.OrderedDict'>.
        model_output = model(**encodings)
        #for key, value in model_output.items(): 
        #    print(key) 
        ## last_hidden_state is [0]
        ## past_key_values is [1]

    pooled_embeds = mean_pooling(model_output, encodings['attention_mask'])
    # Dim is torch.Size([batch_size, embed_size]).

    return{'embedding': pooled_embeds.cpu().numpy()}    

tokenizer.pad_token = tokenizer.eos_token


fracs = [0.25, 0.25, 0.25]

ds_dict_mini = defaultdict(DatasetDict)

for frac in fracs:
    # shuffle and downsample as per fracs
    for split in ds_dict:
        ds_dict_mini[split] = ds_dict[split].shuffle(seed=0).select(range(int(frac * ds_dict[split].num_rows)))  # Shuffle to avoid topic bias when downsampling.

# The line below does not seems to change the padding token id from the default 50256 to something else. Should not matter since we're not including
# them in the pooling anyway.
tokenizer.pad_token = tokenizer.eos_token

embs_train = ds_dict_mini['train'].map(embed_text, batched=True, batch_size=16)
embs_valid = ds_dict_mini['valid'].map(embed_text, batched=True, batch_size=16)
embs_test = ds_dict_mini['test'].map(embed_text, batched=True, batch_size=16)

# At this point these datasets have a column called 'embedding' which contains for each sample a pooled embedding vector capturing the meaning.
# We're now adding the FAISS index, which based on such embeddings and which allows to querty efficiently based on vector similarity.
# After adding this index you can do KNN see p. 277.

embs_train.add_faiss_index('embedding')

























