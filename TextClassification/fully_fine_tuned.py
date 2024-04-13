# This approach (re)trains the weights of the model we use, which means the whole stack including the head needs to be differentiable. Usually
# we use a nn for the classification, similar to how we have it in our Transformer class (head).

# In general, because there are more weights to (re)train, we need more samples vs. the feature extraction method.
# However, also in general, fine tuning will perform better than feature extraction (since we (re)train more weights).


# For explanations above the ##### section see preceding files.

from datasets import load_dataset
from transformers import DistilBertTokenizer
import datasets
import numpy as np

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

model_ckpt = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

train 		= load_dataset('emotion', split='train[:1%]')
valid	 	= load_dataset('emotion', split='validation[:10%]')
test 		= load_dataset('emotion', split='test[:1%]')

#emotions = {'train':train, 'valid':valid, 'test':test}

emotions = datasets.DatasetDict({'train':train, 'valid':valid, 'test':test})

# The below return datasets or datasets in a list, but not in a dict.
#emotions = load_dataset("emotion", split=['train[:1%]'])
#emotions = load_dataset("emotion", split=['train[:1%]', 'validation[:1%]'])

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

###################### New #####################

# Contrary to AutoModel, AutoModelFor... provides us a model with a body AND a head, in this case a nn head for classification. We're going to use labeled
# data to fine-tune the head and also the body. When creating the model, we need to provide info on how many classes we have, since the head is created on
# the fly.

from transformers import AutoModelForSequenceClassification
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels = num_labels).to(device)

# We're going to use HF Trainer, which requires a compute_metrics() function. Input will be a EvalPrediction object (named tuple with 'predictions' and
# 'label_ids' and it needs to return a dict that maps metric's name to its value (in our case 'accuracy' and 'f1'.

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(evalpred):
    labels = evalpred.label_ids
    preds = evalpred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy':acc, 'f1': f1}

from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded['train']) // batch_size
model_name = f'{model_ckpt}-finetuned-emotion'

training_args = TrainingArguments(output_dir 			= model_name,
                                  num_train_epochs 		= 2,
                                  learning_rate 		= 2e-5,
                                  per_device_train_batch_size 	= batch_size,
                                  per_device_eval_batch_size 	= batch_size,
                                  weight_decay 			= 0.01,
                                  evaluation_strategy 		= 'epoch',			# Load best model at end of training run
   												# i.e., the one after the epoch with the best metrics.
                                  disable_tqdm 			= False,
                                  logging_steps 		= logging_steps,
                                  push_to_hub 			= False,
                                  log_level 			= 'error')

from transformers import Trainer

trainer = Trainer(model 	= model,
                  args 		= training_args,
                  train_dataset = emotions_encoded['train'],
                  eval_dataset 	= emotions_encoded['valid'],
                  tokenizer 	= tokenizer) 

trainer.train()

# In the logs we can get the accuracy and f1 as they are calculated on the validation data during training.
# In order to get a full confusion matrix, we need to create this ourselves, so we are going to push the validation data through the trained model.
# Output of .predict is PredictionObject which has arrays of 'predictions' and 'label_ids' + the metrics from our compute_metrics function which can
# be retrieved via .output_metrics.

preds = trainer.predict(emotions_encoded['valid'])
#print(preds.metrics)
#print(preds.predictions)

# .predictions shows that for each validation sample we get 6 predictions, not softmaxed yet but we can take the highest value as the best candidate.
# argmax will return the position with that highest value which serves as our predicted class id.

y_preds = np.argmax(preds.predictions, axis=1)
y_valid = np.array(emotions_encoded['valid']['label'])
labels = emotions['train'].features['label'].names

#print(y_preds)
#print(y_valid)
#print(labels)

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_preds, y_true, labels):						# labels is just to get the emotion names.

    num_elements = len(labels)
    X_Tick_List = []
    X_Tick_Label_List = []

    for item in range (0,num_elements):
        X_Tick_List.append(item)
        X_Tick_Label_List.append(labels[item])

    #print("--------------")
    #print(X_Tick_List)
    #print("--------------")
    #print(X_Tick_Label_List)
    #print("--------------")
    #plt.xticks(ticks = X_Tick_List, labels = X_Tick_Label_List, rotation = 25,fontsize = 8)



    cm = confusion_matrix(y_true, y_preds, normalize='true')
    print(cm)
    fix, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
    plt.title('Normalized confusion matrix')
    plt.show()

plot_confusion_matrix(y_preds, y_valid, labels)

# p. 50-53 contains example code for analyzing the errors. Essentially the idea is that you calculate the loss for each validation sample (via map()) and
# then sort the samples so that you can analyze the ones with the highest error - and that you take a look at what's going on with these to see if there
# are any structural shortcomings in how we train the model.


