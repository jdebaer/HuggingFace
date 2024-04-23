# span classification task, with the model predicting start and end tokens of the answser in the context.
# Since the structure of the labels remains the same across datasets, we can start with a model that is already fine-tuned, in this case MiniLM, which
# is a baseline transformer fine-tuned on SQuAD 2.0.
# QA is encoder-only.

from transformers import AutoTokenizer

model_ckpt = 'deepset/minilm-uncased-squad2'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# For QA, inputs are (question, context) pairs - both are passed to the tokenizer.
question = 'How much music can this hold?'
context = """An MP3 is about 1 MB/minute, so about 6000 hours depending on file size."""
encodings = tokenizer(question, context, return_tensors='pt')

#print(encodings)
# input_ids and attention_mask as usual, but also
# 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# where 0 is question id and 1 is context id.
# decode() it back to see what the tokenizer makes of it.
print(tokenizer.decode(encodings['input_ids'][0]))				# [0] because there is a batch dimension.
# [CLS] how much music can this hold? [SEP] an mp3 is about 1 mb / minute, so about 6000 hours depending on file size. [SEP]

# Let's do a quick inference with the model.
import torch
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)

with torch.no_grad():
    inference = model(**encodings)

# inference is type QuestionAnsweringModelOutput.
#print(inference)
#QuestionAnsweringModelOutput(	loss=None, 	
#				start_logits=tensor([[-0.9862, ... , -0.9862]]),	# Logit for each id for "is it the start id of the A".
#				end_logits=tensor([[-0.9623, ..., -0.9623]]), 		# Logit for each id for "is it the end id of the A".
#				hidden_states=None, attentions=None)
# Info: https://blog.paperspace.com/how-to-train-question-answering-machine-learning-models/

