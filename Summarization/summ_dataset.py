from datasets import load_dataset, list_datasets
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

#datasets_list = list_datasets()
#for dataset in datasets_list:
#    print(dataset)

dataset = load_dataset('cnn_dailymail', '3.0.0')

#sample = dataset['train'][1]
#print(sample)

#{'article': 'LONDON, England (Reuters) -- Harry Potter star ... redistributed.', 'highlights': "Harry Potter star ... Monday .\nYoung actor says he has no plans to fritter his cash away .", 'id': '42c027e4ff9730fbb3de84c1af0d2c506e41c3e4'}
## So dict of article./highlights/id and the highlights are newline-separated.

sample_text = dataset['train'][1]['article'][:2000]
summaries = {}

def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])					# sent_tokenize simply puts each sentence on a new line.

summaries['baseline'] = three_sentence_summary(sample_text)

print(summaries)
