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

model_ckpt = 'miguelvictor/python-gpt2-large'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

def mean_pooling(model_output, attention_mask):
    
    token_embeddings = model_output[0]
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask

def embed_text(samples):

    encodings = tokenizer(samples['text'], padding=True, truncation=True, max_length=128, return_tensors='pt')

    with torch.no_grad():

        model_output = model(**encodings)

    pooled_embeds = mean_pooling(model_output, encodings['attention_mask'])

    return{'embedding': pooled_embeds.cpu().numpy()}    

tokenizer.pad_token = tokenizer.eos_token

embs_train = ds_dict['train'].map(embed_text, batched=True, batch_size=16)
embs_valid = ds_dict['valid'].map(embed_text, batched=True, batch_size=16)
embs_test = ds_dict['test'].map(embed_text, batched=True, batch_size=16)























