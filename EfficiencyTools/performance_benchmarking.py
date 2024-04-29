from transformers import pipeline

class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, intents, optim_type='BERT baseline'):
        self.pipeline = pipeline
        self.dataset = dataset
        self.intents = intents
        self.optim_type = optim_type

    def compute_accuracy(self):
        pass
        
    def compute_size(self):
        pass
 
    def time_pipeline(self):
        pass

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics

# Override compute_accuracy.
from datasets import load_metric
accuracy_score = load_metric('accuracy')

def compute_accuracy(self):
    preds, labels = [], []
    for sample in self.dataset:						# Already a subset via 'test', etc.
        pred = self.pipeline(sample['text'])[0]['label']		# [0] since it's returned as a list.
        label = sample['intent']
        # accuracy_score needs 2 lists of integers.
        preds.append(self.intents.str2int(pred))
        labels.append(label)
    accuracy = accuracy_score.compute(predictions=preds, references=labels)
    return accuracy    

PerformanceBenchmark.compute_accuracy = compute_accuracy

# Override compute_size.
import torch
from pathlib import Path

def compute_size(self):
    state_dict = self.pipeline.model.state_dict()
    tmp_path = Path('model.pt')
    torch.save(state_dict, tmp_path)
    
    size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
    
    tmp_path.unlink()
    return {'size_mb': size_mb}

PerformanceBenchmark.compute_size = compute_size

# Override time_pipeline.
from time import perf_counter
import numpy as np

def time_pipeline(self, query = 'What is the pin number for my account?'):
    
    latencies = []

    # Warmup.
    for _ in range(10):
        _ = self.pipeline(query)

    for _ in range(100):
        start_time = perf_counter()
        _ = self.pipeline(query)
        latency = perf_counter() - start_time
        latencies.append(latency)

    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)

    return {'time_avg_ms':time_avg_ms, 'time_std_ms': time_std_ms}

PerformanceBenchmark.time_pipeline = time_pipeline
