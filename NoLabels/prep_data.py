import pandas as pd
import numpy as np
from naive_bayesline import run_nb

try:
    df_issues = pd.read_pickle('df_issues.pkl')
except:
    dataset_url = 'https://git.io/nlp-with-transformers'
    df_issues = pd.read_json(dataset_url, lines=True)
    df_issues.to_pickle('df_issues.pkl')

cols = ['url', 'id', 'title', 'user', 'labels', 'state', 'created_at', 'body']
#print(df_issues.loc[2, cols].to_frame())

# Filter out just the value of the 'name' key in each tag's JSON object (in the list of tag JSON objects).

df_issues['labels'] = df_issues['labels'].apply(lambda tag_obj_list: [tag_obj['name'] for tag_obj in tag_obj_list])

# Count how many rows (lists) have 0, 1, 2, ... tags.

#print(df_issues['labels'].apply(lambda tag_list: len(tag_list)).value_counts().to_frame().T)

# Find top 10 most frequently used labels.
df_labels_exploded = df_issues['labels'].explode()						# Each element in the list becomes a row.
df_counts = df_labels_exploded.value_counts()							# Now count the rows per label.

# Clean up the labels.

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

# Add a column to indicate if row has labels at all or not.

df_issues['split'] = 'unlabeled'								# Set all col valuese to unlabeled.
mask = df_issues['labels'].apply(lambda x: len(x)) > 0						# True if len() number > 0.
# mask is a <class 'pandas.core.series.Series'>.

df_issues.loc[mask,'split'] = 'labeled'

#print(df_issues['split'].value_counts().to_frame())

# Merge title and body into one 'text' col.

df_issues['text'] = df_issues.apply(lambda x: x['title'] + '\n\n' + x['body'], axis=1)

## Print out a row with a label to see what it looks like.
#for col in ['title', 'body', 'text', 'labels']:
#    print(f'{col}: {df_issues[col].iloc[26][:500]}\n')

# Drop dupes.
df_issues = df_issues.drop_duplicates(subset='text')

## Visualize text lengths to assess impact of truncating. Most transformer models have a context size of 512. The plot below however shows that the
## vast majority of texts is shorter. So we're good.
#import matplotlib.pyplot as plt
#
#df_issues['text'].str.split().apply(len).hist(bins=np.linspace(0,500,50), grid=False, edgecolor='C0')
#plt.title('Words per issue')
#plt.xlabel('Number of words')
#plt.ylabel('Number of issues')
#plt.show()

# Keep only what we need and reset the index.
df_clean = df_issues[['text','labels','split']].reset_index(drop=True).copy()

# Split up according to training purposes.

df_for_unsupervised = df_clean.loc[df_clean['split'] == 'unlabeled', ['text','labels']]
df_for_supervised   = df_clean.loc[df_clean['split'] == 'labeled', ['text','labels']]


# Data is all cleaned at this point. Now we create the training sets.
# For multilabel problems there is no guaranteed balance for all labels and you can end up with unique label combos in
# train that don't exist in test. Use Scikit-multilearn's iterative_train_test_split for this type of multi-label stratification.
# See here http://scikit.ml/stratification.html for explanation on multi-label data stratification.

# We first pre-process.
# MultiLabelBinarizer creates a vector with zeros for absent labels and ones for present labels, for each row.


from sklearn.preprocessing import MultiLabelBinarizer
all_labels = list(label_map.values())
mlb = MultiLabelBinarizer()

mlb.fit([all_labels])								# This is what the inputs are mapped to/matched with.

## Simple example.
## Return type is ndarray.
#mlb.transform([['tokenization', 'new model'],['pytorch']])
## This gives:
##[[0 0 0 1 0 0 0 1 0]
## [0 0 0 0 0 1 0 0 0]]

from skmultilearn.model_selection import iterative_train_test_split

def balanced_split(df, test_size=0.5):
    ind = np.expand_dims(np.arange(len(df)), axis=1)
    # This creates 2-dim array with just the index # in 2nd dim.
    # [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9] ... [index of last row]]

    labels = mlb.transform(df['labels'])
    # This creates a 2-dim array where each 2nd dim element is the mlb vector.

    # The below gives a balanced between the ndarrays. 
    ind_train, _, ind_test, _, = iterative_train_test_split(ind, labels, test_size)	# _ are the labels.
    
    return df.iloc[ind_train[:,0]], df.iloc[ind_test[:,0]]

np.random.seed(0)
df_train, df_temp = balanced_split(df_for_supervised, test_size=0.5)
df_valid, df_test = balanced_split(df_temp, test_size=0.5)

# We want to integrated with HF so we put it all in a DatasetDict.

from datasets import Dataset, DatasetDict

ds_dict = DatasetDict({	'train': Dataset.from_pandas(df_train.reset_index(drop=True)),
			'valid': Dataset.from_pandas(df_valid.reset_index(drop=True)),
			'test' : Dataset.from_pandas(df_test.reset_index(drop=True)),
			'unsup': Dataset.from_pandas(df_for_unsupervised.reset_index(drop=True))})
			
# Notes: labeled was only 441 rows, which we have split in 2 now, so we only have 220 labeled samples to train with.
# This few labels is certainly a challenge, even with transfer learning.
# However we're even going to test performance with even LESS labeled samples, so we're creating even smaller training slices.
# Again we want these to be balanced splits and it's multi-label, so we need to use iterative_train_test_split.

np.random.seed(0)
# Same as before, create a 2-dim ndarray with each index in it's own 2nd dim element.
all_indices = np.expand_dims(list(range(len(ds_dict['train']))), axis=1)

# Same as before, create a 2-dim ndarray with the 0/1 label-matching vectors.
labels = mlb.transform(ds_dict['train']['labels'])

train_sample_counts = [8, 16, 32, 64, 128]

train_slices, last_k = [], 0

indices_pool = all_indices

for i,k in enumerate(train_sample_counts):							# k is the current split size.
    # Split off set of samples from the pool.
    indices_pool, labels, new_slice, _ = iterative_train_test_split(indices_pool, labels, (k - last_k)/len(labels))
    last_k = k
    if i==0: train_slices.append(new_slice)
    else: train_slices.append(np.concatenate((train_slices[-1], new_slice)))

# Add fill dataset as last element.
train_slices.append(all_indices)
train_sample_counts.append(len(ds_dict['train']))
train_slices = [np.squeeze(train_slice) for train_slice in train_slices]

#print(train_sample_counts)
#print([len(x) for x in train_slices])

# Implement baseline w/ Naive Bayes. Sklearn NB does not support multi-label, so we'll use Skmulti-learn lib to reframe the problem to one binary
# classifier per label to do "one vs. rest" classification. So we'll end up with n binary classifiers where n is the # of labels.

def prepare_labels(batch):
    batch['label_ids'] = mlb.transform(batch['labels'])						# Creates the 0/1 mapping vector as per the above.
    return batch

ds_dict = ds_dict.map(prepare_labels, batched=True)


# We'll be measuring micro and macro F1-scores w/ micro for frequent labels only and latter for all labels.
# Create dict to store these scores.

micro_scores, macro_scores = run_nb(train_slices, ds_dict)

# Quick vizualization.
import matplotlib.pyplot as plt

def plot_metrics(micro_scores, macro_scores, sample_sizes, current_model):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10,4), sharey=True)

    for run in micro_scores.keys():
        if run == current_model:
            ax0.plot(sample_sizes, micro_scores[run], label=run, linewidth=2)
            ax1.plot(sample_sizes, macro_scores[run], label=run, linewidth=2)
        else:
            ax0.plot(sample_sizes, micro_scores[run], label=run, linestyle='dashed')
            ax1.plot(sample_sizes, macro_scores[run], label=run, linestyle='dashed')

    for ax in [ax0, ax1]:
        ax.set_xscale('log')
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels(sample_sizes)
        ax.minorticks_off()

    plt.tight_layout()
    plt.show()
 
plot_metrics(micro_scores, macro_scores, train_sample_counts, 'Naive Bayes')
