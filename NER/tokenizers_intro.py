# Little intro to show the diff. between BERT's tokinizer (WordPiece) and XLM-R's tokenizer (SentencePiece).

from transformers import AutoTokenizer

bert_model_name = 'bert-base-cased'
xlmr_model_name = 'xlm-roberta-base'

bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)

text = "Jack Sparrow loves New York!"

print(bert_tokenizer(text).tokens())
print(xlmr_tokenizer(text).tokens())
