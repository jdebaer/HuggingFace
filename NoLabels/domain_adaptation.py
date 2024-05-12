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

from collections import defaultdict
from datasets import DatasetDict

fracs = [0.05, 0.05, 0.05, 0.01]
ds_dict_mini = defaultdict(DatasetDict)

# shuffle and downsample as per fracs
for idx, split in enumerate(ds_dict):
    print(split)
    ds_dict_mini[split] = ds_dict[split].shuffle(seed=0).select(range(int(fracs[idx] * ds_dict[split].num_rows)))  # Shuffle to avoid topic bias when downsampling.
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=128, return_special_tokens_mask=True)

ds_dict_mini_mlm = defaultdict(DatasetDict)

for split in ds_dict_mini:
    ds_dict_mini_mlm[split] = ds_dict_mini[split].map(tokenize, batched=True)
    ds_dict_mini_mlm[split] = ds_dict_mini_mlm[split].remove_columns(['labels','text','label_ids'])

#ds_dict_mini_mlm['train'] = ds_dict_mini['train'].map(tokenize, batched=True)
#ds_dict_mini_mlm['valid'] = ds_dict_mini['valid'].map(tokenize, batched=True)
#ds_dict_mini_mlm['test'] = ds_dict_mini['test'].map(tokenize, batched=True)

# return_special_tokens_mask is to get a mask that masks out [CLS] and [SEP].
# We don't want to use these when we do domain adaptation via masked language modeling.

#text = "this is a test"
#print(tokenizer(text, truncation=True, max_length=128, return_special_tokens_mask=True))
#{'input_ids': [101, 2023, 2003, 1037, 3231, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1], 'special_tokens_mask': [1, 0, 0, 0, 0, 1]}

# To do MLM training, we need to mask randdom token ids in the inputs. Data collator in HF can do that.

from transformers import DataCollatorForLanguageModeling, set_seed

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)		# We mask 15% if the tokens.

# Quick experiment.
#set_seed(3)
#encodings = tokenizer('this is a test')
#print(encodings['input_ids'])
##[101, 2023, 2003, 1037, 3231, 102]				-> 101 is [CLS], 102 is [SEP]
#outputs = data_collator([{'input_ids': encodings['input_ids']}])
#print(outputs)
##{'input_ids': tensor([[ 101,  103, 2003, 1997, 3231,  102]]), 
##'attention_mask': tensor([[1, 1, 1, 1, 1, 1]]), 
##'labels': tensor([[-100, 2023, -100, 1037, -100, -100]])} -> -100 is ignored for CEL calculation. 
## Interesting: even with return_special_tokens_mask not set to true, 101 and 102 are still set to -100.

# At this point we have the training data ready (with special token mask that does not seem to be necessary) and we have our data collator
# so we are ready to train.

from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer

training_args = TrainingArguments(
		output_dir 			= f'{model_ckpt}-issues-128',
		per_device_train_batch_size 	= 32,
		logging_strategy 		= 'epoch',
		evaluation_strategy 		= 'epoch',
		save_strategy 			= 'no',
		num_train_epochs 		= 2,
		push_to_hub			= False,
		log_level 			= 'error',
		report_to 			= 'none')

# Note that we get a warning on dropped weights. This is normal since we're loading a standard BART model to do MaskedLM so we don't need them.
trainer = Trainer(
		model 				= AutoModelForMaskedLM.from_pretrained(model_ckpt),
		tokenizer			= tokenizer,
		args				= training_args,
		data_collator			= data_collator,
		train_dataset			= ds_dict_mini_mlm['unsup'],
		eval_dataset			= ds_dict_mini_mlm['train'])

trainer.train()

# At this point we have used a BERT model to do domain adaptation and we have saved our model locally.
# This was all done using unlabeled data (['unsup']).
# The next step is to fine-tune using the few labels that we have. We can also fine-tune without first doing domain adaptation but the 
# resulting model will not be as powerful - and the model_ckpt in this case will be 'bert-case-unbased' as on p. 285.

model_ckpt = f'{model_ckpt}-issues-128'
config = AutoConfig.from_pretrained(model_ckpt)
config.num_labals = len(all_labels)
config.problem_type = 'multi_label_classification'

for train_slice in train_slices:
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)
    
    trainer = Trainer(
		model 		= model,
		tokenizer	= tokinzer,
		args 		= training_args_fine_tune,
		compute_metrics = compute_metrics,
		train_dataset	= ds_enc['train'].select(train_slice),
		eval_dataset	= ds_enc['valid'])

    trainer.train()


















