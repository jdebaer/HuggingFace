from datasets import load_dataset

emotions = load_dataset("emotion")
train_ds = emotions['train']

 # Take a look inside the dataset.
 print(train_ds)
 print(train_ds.features)
 print("----")
 print(train_ds[0]['text'])
 print(train_ds['text'][0])
 
 emotions.set_format(type='pandas')
 df = emotions['train'][:]
 print(df.head())
 
 # Add textual names for emotions.
 def label_int2str(emo_int):
     return emotions['train'].features['label'].int2str(emo_int)
 df['label_name'] = df['label'].apply(label_int2str)
 print(df.head())
 
 # Check the distribution over the classes.
 import matplotlib.pyplot as plt
 import pandas as pd
 
 df['label_name'].value_counts(ascending=True).plot.barh()
 plt.title('Frequency of classes')
 plt.show()
 
 # Solutions to class imbalance: random oversampling, random undersampling, get more data, generate samples.
 
 # We get a feel for the token count via word count, because our pre-trained models have a set seq_len and at least most of our sentences have to fit in it.
 df['Word Count'] = df['text'].str.split().apply(len)
 df.boxplot('Word Count', by='label_name', grid=False, showfliers=False, color='blue')
 plt.suptitle('')
 plt.xlabel('')
 plt.show()
 
 # Don't forget to reset the dataset format.
 emotions.reset_format()
