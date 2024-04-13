# For explanations above the ##### section see preceding files.

from datasets import load_dataset
from transformers import DistilBertTokenizer
import datasets
import numpy as np

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

model_ckpt = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

train 		= load_dataset('emotion', split='train[:5%]')
valid	 	= load_dataset('emotion', split='validation[:5%]')
test 		= load_dataset('emotion', split='test[:1%]')

#emotions = {'train':train, 'valid':valid, 'test':test}

emotions = datasets.DatasetDict({'train':train, 'valid':valid, 'test':test})

# The below return datasets or datasets in a list, but not in a dict.
#emotions = load_dataset("emotion", split=['train[:1%]'])
#emotions = load_dataset("emotion", split=['train[:1%]', 'validation[:1%]'])

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

###################### New #####################

# Feature extraction is done from the context vectors returned by the Encoder, which normally go to the Decoder for the cross attention.
# Our EncoderDecoder returns these layer-normalized via an extra layer normalization step after all the Encoder blocks have run. 

# Now we're ready to also load the model (we already loaded and used the model's tokenizer).

from transformers import AutoModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Embedding layer is part of the model, needs to be fed the input ids.
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "this is a test"

encoding = tokenizer(text, return_tensors="pt") 

print(f'Input tensor shape: {encoding["input_ids"].size()}')
# Input tensor shape: torch.Size([1, 6]) --> 6 ids, this is a test + start and end token.

print(encoding)
# encoding == {'input_ids': tensor([[ 101, 2023, 2003, 1037, 3231,  102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
# but at this point the tensors are not yet on the GPU, let's do that.

encoding = {k:v.to(device) for k,v in encoding.items()}

# Now let's push the through the model, which (via .model_input_names) we know requires ['input_ids', 'attention_mask']
with torch.no_grad():		  # This is because we are inferencing -> always disable gradient calc during inferencing to reduce RAM footprint.
    outputs = model(**encoding)   # ** takes a dict and transforms it into key=value input parameters, so that we get=
                                  # input_ids = the value for the key input_ids and same for attention_mask.
# Note: run .cpu() as soon as possible on anything returned by the model that's not going to be used anymore for gradient calculation, which is the
#       case here since we're inferencing.

# Type of outputs is BaseModelOutput. We get one context vector per id, and there is no causal masking going on so each context vector contains context
# coming from all the other words. For classification the convention is to do it with the [CLS] token (first token, stands for "classify").
print(outputs)
print(outputs.last_hidden_state.size())
# This is torch.Size([1, 6, 768]) and we want to first of the 6, so we run the below which give [1,768].

last_context_vector = outputs.last_hidden_state[:,0]

# Now we do the above for every sequence in the dataset i.e., we retrieve the last context vector, and batched (transparently).

def extract_last_context_vector(batch):						# batch is batch of dataset_encoded(s), which are results of tokenizing.
                                                                                # input_ids and attention_mask are added columns.
    encoding = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names} 	    # Convert dataset_encoded(s) to tensors and filter out
												    # only the input_ids and attention_mask columns.

    with torch.no_grad():
       last_context_vector = model(**encoding).last_hidden_state[:,0]

    return {'last_cv': last_context_vector.cpu().numpy()}
    
# The model wants tensors, so we need to make sure our batches of dataset_encoded(s) contain tensors, which is not the case yet.

emotions_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Now we are ready to "map" our function to this dataset.

emotions_last_cvs = emotions_encoded.map(extract_last_context_vector, batched=True)

print(emotions_last_cvs['train'].column_names)

X_train = np.array(emotions_last_cvs['train']['last_cv'])
X_valid = np.array(emotions_last_cvs['valid']['last_cv'])
y_train = np.array(emotions_last_cvs['train']['label'])
y_valid = np.array(emotions_last_cvs['valid']['label'])

# p. 42-43 shows how you can vizualize the sentiments in 2D via UMAP

# At this point we have the last context vector for each sequence. Now we can train a classifier on it.
# We can use a nn for this, but logistic regression can work as well and does not require a GPU to train it.

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=3000)							# Increase max_iter to guarantee convergence.
lr_clf.fit(X_train, y_train)
# score here is the accuracy. Random would result in x% where x is 100/number of classes, assuming equal distribution over classes.
print(lr_clf.score(X_valid, y_valid))

# At this point we have trained our LR classifier.

# p. 44 shows how to use the sklearn DummyClassifier to compare this with simple heuristics and are a bit better than random.

# Confusion matrix is next.

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

# y_true are the labels from our validation set. y_preds we need to get by pushing the validation samples through the logistics regression.

y_preds = lr_clf.predict(X_valid)
labels = emotions['train'].features['label'].names

print('*************************')
print(y_valid)
print('*************************')
print(y_preds)
print('*************************')

plot_confusion_matrix(y_preds, y_valid, labels)

















