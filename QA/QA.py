# BERT is pre-trained for reading comprehension, GPT is pre-trained text generation.

# Info: https://blog.paperspace.com/how-to-train-question-answering-machine-learning-models/
# To perform the QA task we add a new question-answering head on top of BERT, just the way we added a masked language model head for performing the MLM task.
# A pre-training objective is a task on which a model is trained before being fine-tuned for the end task. GPT models are trained on a Generative 
# Pre-Training task (hence the name GPT) i.e. generating the next token given previous tokens, before being fine-tuned on, say, SST-2 (sentence classification 
# data) to classify sentences.
# Similarly, BERT uses MLM and NSP as its pre-training objectives. It uses a few special tokens like CLS, SEP, and MASK to complete these objectives. 
# We will see the use of these tokens as we go through the pre-training objectives. But before proceeding, we should know that each tokenized sample fed 
# to BERT is appended with a CLS token in the beginning and the output vector of CLS from BERT is used for different classification tasks. 


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
# This is also called the segment embedding, which helps the encoder to learn distinguish the question from the context.
# 0	0  0  0  0     1  1  1  1
# [CLS] id id id [SEP] id id id [SEP]
# As per the above, prepending the [CLS] is typical for BERT.
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
# Where these logits come from:
# For every token in the text, we feed its final embedding into the start token classifier. The start token classifier only has a single set of 
# weights which applies to every word.

# After taking the dot product between the output embeddings and the ‘start’ weights, we apply the softmax activation to produce a probability 
# distribution over all of the words. Whichever word has the highest probability of being the start token is the one that we pick.
# Inside the question answering head are two sets of weights, one for the start token and another for the end token, which have the same 
# dimensions as the output embeddings. The output embeddings of all the tokens are fed to this head, and a dot product is calculated between 
# them and the set of weights for the start and end token, separately. In other words, the dot product between the start token weight and output 
# embeddings is taken, and the dot product between the end token weight and output embeddings is also taken. Then a softmax activation is applied 
# to produce a probability distribution over all the tokens for the start and end token set (each set also separately). The tokens with the 
# maximum probability are chosen as the start and end token, respectively.

# It's just one linear layer producing 2-D vectors: linear layer takes input from the last hidden state of every input tokens and for each input token this last 
# linear layer generates two outputs. The value of the first output indicates the chance of the corresponding token to be the start index of the final 
# answer. Large positive value means high chance that corresponding token is the starting index and large negative means low chance to be the starting 
# index of the final answer. Same idea applicable for the second output except it is related to the end index of the final answer.

# To do: also see how this works for MLM, for the [MASK] ids.

start_logits = inference.start_logits
end_logits = inference.end_logits

start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1

answer_span = encodings['input_ids'][0][start_idx:end_idx]					# [0] because of batch dim.
answer = tokenizer.decode(answer_span)
print(answer)

# All of the above wrapped in a HF pipeline:
from transformers import pipeline

pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
print(pipe(question=question, context=context, topk=3))
