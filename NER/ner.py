# Notes:
# What we're doing there is similar to text classification with BERT, however, while for TC we're only feeding ONE context vector to the nn head (which has
# enough context to have the nn predict the sentiment), for NER, we're feeding each context vector of the sequence throug the nn head. In this implementation
# we'll ignore tokens that represent non-first subwords i.e., for 'Chr' and '##ista' we would ignore the second token here.

# In general, HF AutoModel is used by invoking <ModelName>For<Task> where Task is done in the head and the model is in the body.
# If you're using just AutoModel, then all you get is the pre-trained body and you have to implement the head yourself.
# Example: from transformers import BertForTaskxyz e.d., BertForMaskedLM or BertForSequenceClassification.
# If the Taskxyz head does not exist yet, you can implement your own, which is what we'll do in this file.
# Essentially the model provides the very last context vectors, and in the head you can decide what to do with them. Are they layer-normalized?
# Below we're implementing our own head on top of XML-R so essentially XLMRobertaForTokenClassification.

import torch.nn as nn
from transformers import XLMRobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

# Note: RobertaPreTrainedModel is an abstract class, so meant to be subclassed and not meant to be instantiated.
# Is just has the interface to implelement (forward() etc.) and also takes cared of loading the pretrained weights for the model and 
# initializing the weights for the head.
class XLMRobertaForTokenClassification(RobertaPreTrainedModel):				# This class is part of transformers alread - learning exercise.
    
    
    # We don't use it here, but config_class ensures that standard XLM-R settings are used when we initialize our model (below) with init_weights().
    # To change default parameters, overwrite them in config_class.
    config_class = XLMRobertaConfig
    
    def __init__(self, config):								# Override constructor.
        super().__init__(config)

        self.num_labels = config.num_labels						# This is for the classification head.

        # Body:
        self.roberta = RobertaModel(config, add_pooling_layer=False)			# '...=False' will make model return all cvs and not just [CLS] one.  

        # Head:
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Load pre-trained weights for body, randomly initialize weights for head:
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,  **kwargs):

        # Use the body to get the context vectors:
        cvs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
 
        cvs = self.dropout(cvs)								# (cvs[0]) in book, why? We want the whole batch.
        
        logits = self.classifier(cvs)
        
        loss = None
        # TO DO: if there is attention mask, then don't include the masked ids for loss calculations. Copy this from train.py in EncoderDecoder.
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()						# is .to(device) needed?
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))		# Stretch both out over all items (samples/labels) in the batch.

        return TokenClassifierOutput(loss		= loss,
                                     logits		= logits,
                                     hidden_states 	= cvs.hidden_states,
                                     attentions 	= cvs.attentions)    








      
