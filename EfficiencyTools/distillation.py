from transformers import pipeline
from performance_benchmarking import PerformanceBenchmark
from distillation_trainer import DistillationTrainingArguments, DistillationTrainer
import numpy as np

# Load model that was already fine-tuned on CLINC150. This is going to be our teacher for distillation.	
bert_ckpt = 'transformersbook/bert-base-uncased-finetuned-clinc'
pipeline = pipeline('text-classification', model = bert_ckpt)
#query = "Hey, I'd like to rent a vehicle from Nov 1st to Nov 25th in Paris and I need a 15 passenger van"
#print(pipeline(query))
#print(type(pipeline(query)))
#[{'label': 'car_rental', 'score': 0.5502981543540955}] -> list of dictionaries, one per sample

# Load CLINC150 to get some data to test with.
from datasets import load_dataset

clinc = load_dataset('clinc_oos', 'plus') # The 'plus' version contains the out of scope test samples.
# Reduce size.
clinc['train'] = clinc['train'].shuffle(seed=0).select(range(int(0.01 * clinc['train'].num_rows)))
clinc['validation'] = clinc['validation'].shuffle(seed=0).select(range(int(0.01 * clinc['validation'].num_rows)))
clinc['test'] = clinc['test'].shuffle(seed=0).select(range(int(0.01 * clinc['test'].num_rows)))

# Note: this dataset is balanced across the classes, so we can use accuracy vs. recall/precision/f1.
# Examples uses:
# clinc['test'][42]
# This gives the intents as integers (classes).
intents = clinc['test'].features['intent'] 
#print(intents)
# ClassLabel(names=['restaurant_reviews', ... , 'change_volume'], id=None)
# <class 'datasets.features.features.ClassLabel'> -> has int2str method:
# intents.int2str(sample['intent']) -> convert int to label.


# Test benchmarking.
pb = PerformanceBenchmark(pipeline, clinc['test'], intents)
perf_metrics = pb.run_benchmark()
print(perf_metrics)

# For the student we pick distilbert.
student_ckpt = 'distilbert-base-uncased'

# We have our dataset, our benchmarking function and our dillation-capable trainer. Still need to prep the data for input.
from transformers import AutoTokenizer

student_tokenizer = AutoTokenizer.from_pretrained(student_ckpt)

def tokenize_text(batch):
    return student_tokenizer(batch['text'], truncation=True)

clinc_encodings = clinc.map(tokenize_text, batched=True, remove_columns=['text'])
# Hugging Face models want 'input_ids' and 'attention_mask', Trainer also wants 'labels'.
clinc_encodings = clinc_encodings.rename_column('intent', 'labels')		

# Just like any other Trainer, our customer Trainer needs a compute_metrics() function.
# We're going to use HF Trainer, which requires a compute_metrics() function. Input will be a EvalPrediction object (named tuple with 'predictions' and
# 'label_ids' and it needs to return a dict that maps metric's name to its value.
from datasets import load_metric
accuracy_score = load_metric('accuracy')
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)

# Set up training arguments.
batch_size = 48
finetuned_ckpt = 'distilbert-base-uncased-finetuned-clinc'

student_training_args = DistillationTrainingArguments(	output_dir 			= finetuned_ckpt,
							evaluation_strategy 		= 'epoch',
							num_train_epochs		= 5,
							learning_rate			= 2e-5,
							per_device_train_batch_size 	= batch_size,
							per_device_eval_batch_size 	= batch_size,
							alpha				= 1,			# We override the default 0.5.
							weight_decay			= 0.01,
							push_to_hub			= False)		# temperature has a default of 2.0.

# We want to initialize a new model each time we start a training session. Trainer constructor had a 'model_init' function for this that we need to provide.
# We need to provide each student model with id2label/label2id. We do that with a model config object that we create with AutoConfig.

from transformers import AutoConfig
id2label = pipeline.model.config.id2label
label2id = pipeline.model.config.label2id

num_labels = intents.num_classes
student_config = AutoConfig.from_pretrained(student_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id)

# Now we can define our model_init function.

device = 'mps'
# Both BERT models we use are loaded with AutoModelForSequenceClassification.
from transformers import AutoModelForSequenceClassification

def student_model_init():
    return AutoModelForSequenceClassification.from_pretrained(student_ckpt, config=student_config).to(device)

# Now we are ready to fine-tune the student. Let's load the teacher and do so.

teacher_ckpt = bert_ckpt
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_ckpt, num_labels=num_labels).to(device)

distillation_trainer = DistillationTrainer(	model_init	= student_model_init,
						teacher_model	= teacher_model,
						args		= student_training_args,
						train_dataset	= clinc_encodings['train'],
						eval_dataset	= clinc_encodings['validation'],
						compute_metrics = compute_metrics,
						tokenizer	= student_tokenizer)
distillation_trainer.train()



























