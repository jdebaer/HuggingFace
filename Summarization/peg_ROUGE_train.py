# Metric: ROUGE (p. 152-154)

from datasets import load_dataset, list_datasets
from evaluate_peg_summaries import evaluate_peg_summaries
from datasets import load_metric
import datasets
import pandas as pd
import matplotlib.pyplot as plt

rouge_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

dataset_samsum = load_dataset('samsum')

#emotions = datasets.DatasetDict({'train':train, 'valid':valid, 'test':test})

#dataset_samsum_min  = [dataset_samsum[split].shuffle(seed=0).select(range(int(0.01 * dataset_samsum[split].num_rows))) for split in dataset_samsum]
#dataset_samsum_min  = [dataset_samsum[split].shuffle(seed=0).select(range(int(0.01 * dataset_samsum[split].num_rows))) for split in dataset_samsum]
#
dataset_samsum_min = datasets.DatasetDict({
    'train': dataset_samsum['train'].shuffle(seed=0).select(range(int(0.001 * dataset_samsum['train'].num_rows + 1))),
    'test': dataset_samsum['test'].shuffle(seed=0).select(range(int(0.001 * dataset_samsum['test'].num_rows + 1))),
    'validation': dataset_samsum['validation'].shuffle(seed=0).select(range(int(0.001 * dataset_samsum['validation'].num_rows + 1)))
})


#print(dataset_samsum)
#print(dataset_samsum_min)
#exit(0)

rouge_metric = load_metric('rouge')

device = 'cpu'

#split_lengths = [len(dataset_samsum[split]) for split in dataset_samsum]
#print(split_lengths)
#print(dataset_samsum['train'].column_names)
#[14732, 819, 818]
#['id', 'dialogue', 'summary']

# First eval Pegasus (trained on CNN Dailymail) on SAMSum without fine-tuning #

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_ckpt = 'google/pegasus-cnn_dailymail'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

score = evaluate_peg_summaries(dataset_samsum_min['test'], rouge_metric, model, tokenizer, column_text='dialogue', column_summary='summary', batch_size=8)
our_rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)

#print(pd.DataFrame(our_rouge_dict, index=['Pegasus:']))
# Not so good results, because we're using Pegasus on a format (text-style conversations) that is hasn't been specifically trained on.
# Now let's fine-tune it using the training data in the SAMSum dataset.

## Initial investigation of the training data: let's look at the distribution of length (# of ids) for both dialogues and summaries.
#
#all_dialog_lens  = [len(tokenizer.encode(s)) for s in dataset_samsum['train']['dialogue']]
#all_summary_lens = [len(tokenizer.encode(s)) for s in dataset_samsum['train']['summary']]
#
#fig, axes = plt.subplots(1, 2, figsize=(10,3.5), sharey=True)
#axes[0].hist(all_dialog_lens, bins=20, color='C0', edgecolor='C0')
#axes[0].set_title('Dialogue')
#axes[0].set_xlabel('Length')
#axes[0].set_ylabel('Count')
#axes[1].hist(all_summary_lens, bins=20, color='C0', edgecolor='C0')
#axes[1].set_title('Summary')
#axes[1].set_xlabel('Length')
#plt.tight_layout()
#plt.show()
# Based on what we see, we set max length to 1024 for dialogues and 128 for summaries.

# We need to get the traininig data into the format that we can feed into a HF model i.e., a dict with the keys 'input_ids', 'attention_mask', 'labels'.
# As usual we do this via map().

def convert_samples(sample_batch):
    encodings = tokenizer(sample_batch['dialogue'], max_length=1024, truncation=True)
    
    # context: HF tokenizer now knows it's tokenizing for the decoder. Some HF models require special tokens in the decoder inputs.
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(sample_batch['summary'], max_length=128, truncation=True)

    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

dataset_samsum_min_pt = dataset_samsum_min.map(convert_samples, batched=True)

# map() adds the columns.
#print(dataset_samsum_min_pt)
#DatasetDict({
#    train: Dataset({
#        features: ['id', 'dialogue', 'summary', 'input_ids', 'attention_mask', 'labels'],
#        num_rows: 15
#    })
#    test: Dataset({
#        features: ['id', 'dialogue', 'summary', 'input_ids', 'attention_mask', 'labels'],
#        num_rows: 1
#    })
#    validation: Dataset({
#        features: ['id', 'dialogue', 'summary', 'input_ids', 'attention_mask', 'labels'],
#        num_rows: 1
#    })
#})

columns = ['input_ids', 'labels', 'attention_mask']

dataset_samsum_min_pt.set_format(type='torch', columns=columns)   # The contents of these cols need to be tensors and will need to move to the right device.


from transformers import DataCollatorForSeq2Seq
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# This collator does all of the following:
# Shifting the labels to the right and using this as the input for the decoder.
# Applying the causal mask to the decoder input. No padding mask required for decoder input.
# Applying a padding mask to the decoder label (verify this in Encoder/Decoder).
# Ensure that padding tokens in the labels are ignored by the loss function by setting them to -100.

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir 			= 'pegasus-samsum',
    num_train_epochs 		= 1,
    warmup_steps 		= 500,
    per_device_train_batch_size	= 1,
    per_device_eval_batch_size	= 1,
    weight_decay		= 0.01,
    logging_steps		= 10,
    push_to_hub			= False,
    evaluation_strategy		= 'steps',
    eval_steps			= 500,
    save_steps			= 1e6,
    use_cpu			= True,
    gradient_accumulation_steps	= 16)			# Because of small batch size. Small batch size caused by big model, does not fit on GPU.
							# We only run the optimization step after accumulating 16 gradients.
trainer = Trainer(
    model = model,
    args = training_args,
    tokenizer = tokenizer,
    data_collator = seq2seq_data_collator,
    train_dataset = dataset_samsum_min_pt['train'],
    eval_dataset = dataset_samsum_min_pt['validation'])

trainer.train()

# Same as before, but now we use trainer.model.
score = evaluate_peg_summaries(dataset_samsum_min['test'], rouge_metric, trainer.model, tokenizer, column_text='dialogue', column_summary='summary', batch_size=8)
our_rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
print(pd.DataFrame(our_rouge_dict, index=['Pegasus:']))
