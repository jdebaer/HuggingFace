# Metric: ROUGE (p. 152-154)
from datasets import load_dataset, list_datasets
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from datasets import load_metric
import pandas as pd
import torch
from evaluate_peg_summaries import evaluate_peg_summaries

#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

bleu_metric = load_metric('sacrebleu')
rouge_metric = load_metric('rouge')

rouge_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

dataset = load_dataset('cnn_dailymail', '3.0.0')

def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3]) 

def evaluate_summaries_baseline(batch, metric, column_text='article', column_summary='highlights'):
    summaries = [three_sentence_summary(text) for text in batch[column_text]]
    metric.add_batch(predictions = summaries, references = batch[column_summary])
    score = metric.compute()
    return score

## Run the baseline on 100 samples and measure performance.
#test_samples = dataset['test'].shuffle(seed=42).select(range(100))
test_samples = dataset['test'].shuffle(seed=42).select(range(1))
#
#score = evaluate_summaries_baseline(test_samples, rouge_metric)
## score is a dict where the keys are the four rouge types and per type there are various measurements. We take the mid fmeasure and create a smaller dict.
#
#our_rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
#print(our_rouge_dict)
##print(pd.DataFrame(our_rouge_dict, orient='index', columns=['baseline']).T)
#print(pd.DataFrame(our_rouge_dict, index=['Baseline:']))
#             rouge1    rouge2    rougeL  rougeLsum
#Baseline:  0.387227  0.165939  0.247094   0.351829

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_ckpt = 'google/pegasus-cnn_dailymail'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

score = evaluate_peg_summaries(test_samples, rouge_metric, model, tokenizer, batch_size=8)

our_rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)

print(pd.DataFrame(our_rouge_dict, index=['Pegasus:']))
