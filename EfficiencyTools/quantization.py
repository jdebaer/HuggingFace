from torch.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch.nn as nn
import torch
from transformers import pipeline
from performance_benchmarking import PerformanceBenchmark
from datasets import load_dataset

clinc = load_dataset('clinc_oos', 'plus') # The 'plus' version contains the out of scope test samples.
# Reduce size.
#clinc['train'] = clinc['train'].shuffle(seed=0).select(range(int(0.01 * clinc['train'].num_rows)))
#clinc['validation'] = clinc['validation'].shuffle(seed=0).select(range(int(0.01 * clinc['validation'].num_rows)))
clinc['test'] = clinc['test'].shuffle(seed=0).select(range(int(0.01 * clinc['test'].num_rows)))

intents = clinc['test'].features['intent']


torch.backends.quantized.engine = 'qnnpack'

model_ckpt = 'transformersbook/distilbert-base-uncased-distilled-clinc'

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to('cpu')
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

pipeline = pipeline('text-classification', model=model_quantized, tokenizer=tokenizer)

optim_type = 'distillation + quanitzation'

pb = PerformanceBenchmark(pipeline, clinc['test'], intents, optim_type=optim_type)

print(pb.run_benchmark())

