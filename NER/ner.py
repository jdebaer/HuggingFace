from datasets import DatasetDict, load_dataset
from collections import defaultdict
import pandas as pd
from mytokenclassifier import XLMRobertaForTokenClassification
import torch
from transformers import AutoTokenizer

langs = ['de', 'fr', 'nl']
fracs = [0.7, 0.2, 0.1]

panx_be = defaultdict(DatasetDict)

for lang, frac in zip(langs, fracs):
    ds = load_dataset('xtreme', name=f'PAN-X.{lang}')
    # shuffle and downsample as per fracs
    for split in ds:
        panx_be[lang][split] = ds[split].shuffle(seed=0).select(range(int(frac * ds[split].num_rows)))  # Shuffle to avoid topic bias when downsampling.

tags = panx_be['de']['train'].features['ner_tags'].feature

def create_tag_names(batch):
    return {'ner_tag_names': [tags.int2str(idx) for idx in batch['ner_tags']]}

panx_de = panx_be['de'].map(create_tag_names)

xlmr_model_name = "xlm-roberta-base"
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
text = "Jack Sparrow loves New York!"


index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In HF, every checkpoint (pre-trained model) is assigned  a configuration file that specifies various hyperparameters like vocab_size and hidden_size
# as well as additional metadata like label names. We're going to overwrite some of these. Basically the AutoConfig class conatains the blueprint
# that was used to create the model. When we load a model with AutoModel.from_pretrained("name") the config files associated with the model is downloaded
# automatically, however if we want to change something like the number of classes or label names, then we load the config first so that we can change
# the related parameters, using an additional "config=" parameter as we do below.

from transformers import AutoConfig

xlmr_config = AutoConfig.from_pretrained(xlmr_model_name, num_labels=tags.num_classes, id2label=index2tag, label2id=tag2index)

# XLMRobertaForTokenClassification is our own class, defined below. It has inhereited from_pretrained() from RobertaPreTrainedModel.

# If you use the REAL XLMRobertaForTokenClassification class, then the command below is identical!
xlmr_model = XLMRobertaForTokenClassification.from_pretrained(xlmr_model_name, config=xlmr_config).to(device)

# # At this point we can run a quick inference to see if things are don't fall apart, but of course the head is still untrained so the results
# # will not be great.
# # Note that the XML-R tokenizer adds the batch dimension, in this cases size 1.
# input_ids = xlmr_tokenizer.encode(text, return_tensors='pt')
# # tokens is a list.
# tokens = xlmr_tokenizer(text).tokens()									# tokens is always a list.
# print(pd.DataFrame([tokens, input_ids[0].numpy()], index=['Tokens','Input IDs']))
# #             0      1      2      3      4  5     6      7   8     9
# #Tokens     <s>  ▁Jack  ▁Spar    row  ▁love  s  ▁New  ▁York   !  </s>
# #Input IDs    0  21763  37456  15555   5161  7  2356   5753  38     2
# 
# # Move data to the device where the model is.
# # Type of output is TokenClassifierOutput. This is for the whole classifier that inherited from RobertaPreTrainedModel.
# # Type of just the body model return is BaseModelOutputWithPoolingAndCrossAttentions.
# token_classifier_output = xlmr_model(input_ids.to(device))						# Type of output is TokenClassifierOutput.
# logits = token_classifier_output.logits
# # print(type(logits))
# # <class 'torch.Tensor'>
# # print(logits.size())
# # torch.Size([1, 10, 7]) -> for each id we have a label predictions (7 possibilities).
# # We can grab the highest number as the prediction.
# predictions = torch.argmax(logits, dim = -1)
# preds_label_names = [tags.names[p] for p in predictions[0].cpu().numpy()]
# print(pd.DataFrame([tokens, preds_label_names], index = ['Tokens','Tags']))
# 
# # As said no good performance, but it's predicting labels (last row):
# #             0      1      2      3      4  5     6      7   8     9
# #Tokens     <s>  ▁Jack  ▁Spar    row  ▁love  s  ▁New  ▁York   !  </s>
# #Input IDs    0  21763  37456  15555   5161  7  2356   5753  38     2
# #          0      1      2      3      4      5      6      7      8     9
# #Tokens  <s>  ▁Jack  ▁Spar    row  ▁love      s   ▁New  ▁York      !  </s>
# #Tags      O  B-PER  B-PER  B-PER  B-PER  B-PER  B-PER  B-PER  B-PER     O
# # We put the above experiment's logic in a function that receives text and outputs the last pandas dataframe:

def tag_text(text, tags, model, tokenizer):
    tokens = tokenizer(text).tokens()
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    logits = model(input_ids).logits
    predictions = torch.argmax(logits, dim = -1)
    preds_label_names = [tags.names[p] for p in predictions[0].cpu().numpy()]
    return (pd.DataFrame([tokens, preds_label_names], index = ['Tokens','Tags']))

#print(tag_text(text, tags, xlmr_model, xlmr_tokenizer))

# At this point we know our custom XLMRobertaForTokenClassification can convert text to predicted labels. Now let's prep things to fine-tune en masse.

# 1. We need to tokenize (to ids) all input samples in the dataset. Typically in HF this is done with a map().
# See xtreme_intro.py for a step by step breakdown, below is the function form.


def tokenize_and_align_labels(batch, tokenizer):

    # A function like that that will be used via map() has to be able to produce the result for the whole batch.

    # Get BatchEncoding object for the whole batch (tokenizer can do this).
    # batch_encoding = tokenizer(batch['tokens'], is_split_into_words=True, truncation=True)   

    batch_encoding = tokenizer(batch['tokens'], is_split_into_words=True)   

    batched_id_label_numbers = []

    for idx, labels in enumerate(batch['ner_tags']):								# We're going over every sample in the batch.
        word_ids = batch_encoding.word_ids(batch_index = idx)							# Get the word ids (0,0,1,1,1 etc.) for a sample in the batch.
        previous_word_idx = None
        id_label_numbers = []

        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                id_label_numbers.append(-100)
            elif word_idx != previous_word_idx:
                id_label_numbers.append(labels[word_idx])  
            previous_word_idx = word_idx
        
        batched_id_label_numbers.append(id_label_numbers)
    batch_encoding['id_label_numbers'] = batched_id_label_numbers

    return batch_encoding

def encode_panx_dataset(corpus, tokenizer):
    return corpus.map(tokenize_and_align_labels, batched=True, remove_columns=['langs', 'ner_tags', 'tokens'], fn_kwargs={"tokenizer": tokenizer})
        
     
panx_de_encoded = encode_panx_dataset(panx_be['de'], xlmr_tokenizer)

#print(panx_de_encoded)
##DatasetDict({
##    train: Dataset({
##        features: ['input_ids', 'attention_mask', 'id_label_numbers'],
##        num_rows: 14000
##    })
##    validation: Dataset({
##        features: ['input_ids', 'attention_mask', 'id_label_numbers'],
##        num_rows: 7000
##    })
##    test: Dataset({
##        features: ['input_ids', 'attention_mask', 'id_label_numbers'],
##        num_rows: 7000
##    })
##})
#
#print(panx_de_encoded['train'][0])
##{
##'input_ids': [0, 70101, 176581, 19, 142, 122, 2290, 708, 1505, 18363, 18, 23, 122, 127474, 15439, 13787, 14, 15263, 18917, 663, 6947, 19, 6, 5, 2], 
##'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
##'id_label_numbers': [-100, 0, 0, -100, 0, 0, 5, -100, -100, 6, -100, 0, 0, 5, -100, 5, -100, -100, -100, 6, -100, -100, 0, -100, -100]
##}

























