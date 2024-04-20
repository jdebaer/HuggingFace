# Some important insights:
# When using HF, you can pad a batch to a uniform seq_len by using a DataCollator. This will padd with a value of -100 as the class label. This -100 value
# is ignored by nn.CrossEntropyLoss as it is the default value for the ignore_index. Note that we do exactly the same but manually in our EncoderDecoder
# train.py with: loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device) where we set the id of [PAD]
# as the ignore_index and we set our attention mask similarly i.e. we always mask out padding tokens that are used as seq_len fillers to the end.

# One difference for NER is that we also set -100 as the class label for ids of an entity that are not the first id of the word that we want to do NER for. So
# these ids are also ignored by the nn.C... lost function. This is because we only want to measure success by using the first id of the word. How the others
# were predicted does not matter, since we'll use the prediction for the first id as the predition for the whole word. DO NOTE however that contrary to the
# "end of seq" padding tokens, we're NOT masking these out. This is because we want to include them in the context to that the model can learn how to use
# them to predict the correct label for the first id. That's never the case for "end of seq" padding tokens, so we do mask those out.

# One extra consideration: there will always be a 'O' label in NER. This is the label for anyything that is NOT a label of interest like B-PER etc. This needs
# to be included in training because we don't want to mispredict (false postitive). Again we'd only consider the first id etc.

# And: in NER you'll have entities that span multiple words (not tokens, words). This is why here we have B-PER and I-PER where I-PER is used for the 
# subsequent words. These I-* labels also have to be predicted correctly so they are fully included in training and performance measuring.


from datasets import DatasetDict, load_dataset
from collections import defaultdict
import pandas as pd
from mytokenclassifier import XLMRobertaForTokenClassification
import torch
from transformers import AutoTokenizer

langs = ['de', 'fr', 'nl']
fracs = [0.007, 0.002, 0.001]

panx_be = defaultdict(DatasetDict)

for lang, frac in zip(langs, fracs):
    ds = load_dataset('xtreme', name=f'PAN-X.{lang}')
    # shuffle and downsample as per fracs
    for split in ds:
        panx_be[lang][split] = ds[split].shuffle(seed=0).select(range(int(frac * ds[split].num_rows)))  # Shuffle to avoid topic bias when downsampling.

#print(panx_be['de']['train'][0])
#{'tokens': ['2.000', 'Einwohnern', 'an', 'der', 'Danziger', 'Bucht', 'in', 'der', 'polnischen', 'Woiwodschaft', 'Pommern', '.'], 'ner_tags': [0, 0, 0, 0, 5, 6, 0, 0, 5, 5, 6, 0], 'langs': ['de', 'de', 'de', 'de', 'de', 'de', 'de', 'de', 'de', 'de', 'de', 'de']}


tags = panx_be['de']['train'].features['ner_tags'].feature

def create_tag_names(batch):
    return {'ner_tag_names': [tags.int2str(idx) for idx in batch['ner_tags']]}

panx_de = panx_be['de'].map(create_tag_names)

xlmr_model_name = "xlm-roberta-base"
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
text = "Jack Sparrow loves New York!"


index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# In HF, every checkpoint (pre-trained model) is assigned  a configuration file that specifies various hyperparameters like vocab_size and hidden_size
# as well as additional metadata like label names. We're going to overwrite some of these. Basically the AutoConfig class conatains the blueprint
# that was used to create the model. When we load a model with AutoModel.from_pretrained("name") the config files associated with the model is downloaded
# automatically, however if we want to change something like the number of classes or label names, then we load the config first so that we can change
# the related parameters, using an additional "config=" parameter as we do below.

from transformers import AutoConfig

xlmr_config = AutoConfig.from_pretrained(xlmr_model_name, num_labels=tags.num_classes, id2label=index2tag, label2id=tag2index)

# XLMRobertaForTokenClassification is our own class, defined below. It has inhereited from_pretrained() from RobertaPreTrainedModel.

# If you use the REAL XLMRobertaForTokenClassification class, then the command below is identical!
#xlmr_model = XLMRobertaForTokenClassification.from_pretrained(xlmr_model_name, config=xlmr_config).to(device)

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

#def tag_text(text, tags, model, tokenizer):
#    tokens = tokenizer(text).tokens()
#    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
#    logits = model(input_ids).logits
#    predictions = torch.argmax(logits, dim = -1)
#    preds_label_names = [tags.names[p] for p in predictions[0].cpu().numpy()]
#    return (pd.DataFrame([tokens, preds_label_names], index = ['Tokens','Tags']))

def tag_text(text, tags, model, tokenizer):
    tokens = tokenizer(text).tokens()
    input_ids = xlmr_tokenizer(text, return_tensors='pt').input_ids.to(device)
    outputs = model(input_ids)[0]
    predictions = torch.argmax(outputs, dim=2)
    preds = [tags.names[p] for p in predictions[0].cpu().numpy()]
    return pd.DataFrame([tokens, preds], index = ['Tokens','Tags'])



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

# forward() has a parameter labels=... -> HF matches this with a column name called 'labels', so make sure your labels are called this.
# Error you get is: 
# The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask.
panx_de_encoded = panx_de_encoded.rename_column('id_label_numbers', 'labels')			

# Before we can start training, we still need to define a performance measure.
# For a prediction to be correct, we need all the words of the entity to be labeled correctly. We are already ignoring the "add-on" tokens. This is about the words, not the tokens.

# Predictions will be dim (batch_size, seq_len, #labels) with dim 2 not softmaxed so the pure logits with the highest one being the prediction.
# Note that the body will produce a context vector per id, and the head takes that vector and pushed it through a nn to get something in dim '#labels'.
# Labels will be dim (batch_size, seq_len).
# We need to convert batches of these predictions and labels to lists of lists, so that seqeval can eval it.
# Note: seq_len is the amount of ids we got after converting to tokens.

import numpy as np

def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)										# Dim of preds is now also (batch_size, seq_len).
    batch_size, seq_len = preds.shape
    
    labels_list, preds_list = [], []											# The outer list.

    for batch_idx in range(batch_size):
        sample_labels_list, sample_preds_list = [], []									# The inner lists. 
        
        for seq_idx in range(seq_len):
            # We don't include ids from prediction and labels if the label has -100 for it.
            if label_ids[batch_idx, seq_idx] != -100:
                sample_labels_list.append(index2tag[label_ids[batch_idx][seq_idx]])
                sample_preds_list.append(index2tag[preds[batch_idx][seq_idx]])

        labels_list.append(sample_labels_list)
        preds_list.append(sample_preds_list)
    
    return preds_list, labels_list


############################## FINE-TUNING ###############################

from transformers import TrainingArguments										# For HF Trainer.

num_epochs = 1
batch_size = 24
logging_steps = len(panx_de_encoded['train']) // batch_size
model_name = f'{xlmr_model_name}-finetuned-panx-de'

training_args = TrainingArguments(
    output_dir 			= model_name,
    log_level 			= 'error',
    num_train_epochs 		= num_epochs,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size 	= batch_size,
    evaluation_strategy 	= 'epoch',										# Do validation at end of each epoch.
    save_steps 			= 1e6,											# Set to large number to disable checkpointing and speed up training.
    weight_decay		= 0.01,
    disable_tqdm		= False,
    logging_steps 		= logging_steps,
    push_to_hub 		= False)
    

from seqeval.metrics import f1_score

# We need compute_metrics() function to feed to Trainer. 
# We're going to use HF Trainer, which requires a compute_metrics() function. Input will be a EvalPrediction object (named tuple with 'predictions' and
# 'label_ids' and it needs to return a dict that maps metric's name to its value. We'll use our align_predictions() function.

def compute_metrics(eval_pred):
    # Create the lists of lists that seqeval can work with.
    y_pred, y_label = align_predictions(eval_pred.predictions, eval_pred.label_ids)  
   
    return {'f1': f1_score(y_label, y_pred)}

# Since we're doing the same matrix calculations on each sample in the batch, all seq_lens in a batch have to be the same, and typically we take the 
# max len over all labels and inputs and we padd things up to match that (and add a padding mask). We do all this manually in EncoderDecoder, but HF
# has DataCollatorForTokenClassification for this. Note that for TextClassification this was not an issue since we're only taking one token per sample
# and we're also predicting one value.

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)				# Prob. gets the padding token from the tokenizer.

# Trainer can create a model object at every invocation of train() if we provide a model_init() for it.

def model_init():
    return XLMRobertaForTokenClassification.from_pretrained(xlmr_model_name, config=xlmr_config).to(device)

# Now let's train.

from transformers import Trainer

trainer = Trainer(
    model_init 		= model_init,
    args 		= training_args,
    data_collator 	= data_collator,
    compute_metrics 	= compute_metrics,
    train_dataset 	= panx_de_encoded['train'],
    eval_dataset 	= panx_de_encoded['validation'],
    tokenizer 		= xlmr_tokenizer)


trainer.train()
print(trainer.model.device)


text_de = "Jeff Dean ist ein Informatiker bei Google in Kalifornien"
print(tag_text(text_de, tags, trainer.model, xlmr_tokenizer))















