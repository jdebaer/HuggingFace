from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from constantlengthdataset import ConstantLengthDataset

model_ckpt = 'gpt2'

# Take tokenizer from the model and retrain it.
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)				
byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v,k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())

length = 2000
dataset_name = 'transformersbook/codeparrot-train'
dataset = load_dataset(dataset_name, split='train', streaming=True)
iter_dataset = iter(dataset)

def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)['content'] for _ in range(batch_size)]

tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=32768, initial_alphabet=base_vocab)

##### Code above is all that's needed to retrain HF tokenizer. python_tokenizer.py has all the details and comments.

################################# Train a HF model from scratch ################################

from transformers import AutoConfig, AutoModelForCausalLM

# Note: if you would have saved the retrained tokenizer to the hub, it would have a ckpt name and you would load it like this:
# tokenizer = AutoTokenizer.from_pretrained(model_cpkt)  	# With model_ckpt being the name you used when saving your retrained tokenizer.

# This would be a different model_ckpt than the one you used to save the retrained tokenizer.
# In our case this is gpt2 as per the above, so the untrained model we're going to load to then retrain.

# This is how you load an untrained model from the hub when you just have make some config changes (in this case the vocab size).
config = AutoConfig.from_pretrained(model_ckpt, vocab_size=len(tokenizer))		
model = AutoModelForCausalLM.from_config(config)

# Saving a model to a file.
model.save_pretrained("models/" + model_ckpt, push_to_hub=False, organization="transformersbook")

# Estimate how many Unicode characters we typically have per token, based on 500 samples.
samples, total_characters, total_tokens = 500, 0, 0
dataset_for_sample = load_dataset('transformersbook/codeparrot-train', split='train', streaming=True)
for _,sample in tqdm(zip(range(samples), iter(dataset)), total=samples):
    total_characters += len(sample['content'])
    total_tokens += len(tokenizer(sample['content']).tokens())
characters_per_token = total_characters / total_tokens

## Run a test with characters_per_token in our call to ConstantLengthDataset.
#shuffled_dataset = dataset.shuffle(buffer_size=100)
#constant_length_dataset = ConstantLengthDataset(tokenizer, shuffled_dataset, num_of_seqs=10, chars_per_token=characters_per_token, seq_len=1014)
#dataset_iterator = iter(constant_length_dataset)
#lengths = [len(b) for _,b in zip(range(5), dataset_iterator)]
#print(lengths)

################################################ Accelerate ##############################################

# 1. Set up hyperparameters.

from argparse import Namespace

config = {	'train_batch_size'		: 2,
		'valid_batch_size'		: 2,
		'weight_decay'			: 0.1,
		'shuffle_buffer'		: 1000,
		'learning_rate'			: 2e-4,
		'lr_scheduler_type'		: 'cosine',
		'num_warmup_steps'		: 750,
		'gradient_accumulation_steps'	: 16,
		'max_train_steps'		: 50000,
		'max_eval_steps'		: -1,
		'seq_len'			: 1024,
		'seed'				: 1,
		'save_checkpoint_steps'		: 50000		}

args = Namespace(**config)

# 2. Set up logging. General remarks:
# Each Accelerate worker has a unique accelerator.process_index and has its own logger. We use the index to create a unique logging file for each worker.
# The TensorBoard and W&B loggers only need to be initialized once, and we use the accelerator.is_main_process attribute to enforce that.
# Also we use this attribute to decrease the log levels for the non-main workers. 

from torch.utils.tensorboard import SummaryWriter			# This logging via Weights & Biases.
import logging								# This is a standard Python logger.
import wandb								# This is logging via TensorBoard.

def setup_logging(project_name):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format		= '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt		= '%m/%d/%Y %H:%M:%S', 
        level		= logging.INFO, 
        handlers	= [
            logging.FileHandler(f'log/debug_{accelerator.process_index}.log'),
            logging.StreamHandler()
        ]
    )
    if accelerator.is_main_process:						# We only want to set up logging once.
        wandb.init(project=project_name, config=args)
        run_name = wandb.run.name
        tb_writer = SummaryWriter()
        tb_writer.add_hparams(vars(args), {'0': 0})
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ''
        tb_writer = None
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return logger, tb_writer, run_name

def log_metrics(step, metrics):
    logger.info(f'Step {step}:{metrics}')
    if accelerator.is_main_process:
        wandb.log(metrics)
        [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]

# 3. Function to create DataLoaders based on our IterableDataset called ConstantLengthDataset. DataLoader takes care of batching.
# Note that Accelerate automatically distributes the batches to workers.

from toch.untils.data.dataloader import DataLoader
    
def create_dataloaders(dataset_name):
     
    train_data_ = load_dataset(dataset_name + '-train', split='train', streaming=True)
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    valid_data = load_dataset(dataset_name + '-valid', split='validation', streaming=True)

    train_dataset = ConstantLengthDataset(tokenizer, train_data, seq_len = args.seq_len)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data, seq_len = args.seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    return train_dataloader, eval_dataloader

# 4. If we use weight decay, then we should differentiate between parameters that should receive weight decay and those who should not.
# In general, biases and LayerNorm weights are not subject to weight decay.

# Something to look into: You should make sure that if you store a list of nn.Parameter, it should be in a nn.ParameterList and not a plain python list, 
# same if you store a list of nn.Modules should be in a nn.ModuleList.

def get_grouped_params(model, no_decay=['bias', 'LayerNorm.weight']):

    params_with_wd, params_without_wd = [], []
    
    for name, parameter in model.named_parameters():					# name is str, parameter is nn.Parameter.
        if any(n in name for n in no_decay):						# The str's in no_decay are substr's. any() is more concise than using or.
            params_without_wd.append(parameter)
        else:
            params_with_wd.append(parameter)

    return [	{'params': params_with_wd, 'weight_decay': args.weight_decay},
		{'params': params_without_wd, 'weigth_decay': 0.0}	]

	
# 5. We will evaluate after every epoch (typically) so we also need to implement an evaluation function that should at least return the loss.
# We also return the perplexity here. Perplexity measures how well the model's output probability distributions predict the targeted tokens. Lower is better.

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():

            # With HF Accelerate, each GPU receives a batch prepared by the DataLoader.
            outputs = model(batch, labels=batch)			# The model has been prepared by Accelerate.

            # The outputs object is a SequenceClassifierOutput in HF. It has an optional loss, a logits, an optional hidden_states and an 
            # optional attentions attribute. Here we have the loss since we passed along labels, but we don’t have hidden_states and attentions 
            # because we didn’t pass output_hidden_states=True or output_attentions=True.

        # gather() does a "manual" gathering that will run on all workers. It's (list of) tensor(s) in, (list of) tensor(s) out.
        # The out tensor will however have the size of its first dimension multiplied by the number of processes.
        # Simple example:
        # process_tensor = torch.tensor([accelerator.process_index])
        # gathered_tensor = accelerator.gather(process_tensor)
        # gathered_tensor
        # tensor([0, 1, 2, 3]) 		# This will be the result on all the workers.



        # TO DO: why are we mulitiplying by batch_size.
        loss = outputs.loss.repeat(args.valid_batch_size)		# This repeat extends the tensor a.v times, copying the data. No dim change (1).
        # For every batch here, a batch runs on every worker. gather() gathers all the losses over the processes, and for every loop iteration we store
        # that gathered result.
        losses.append(accelerator.gather(loss))				# Result should be one big concatenated tensor of all all the processes' loss tensors.

        if args.max_eval_steps > 0 and step >= args.max_eval_steps: break

    loss = torch.mean(torch.cat(losses))				# cat() contatenates on (default) dimension 0 all the tensors in the list.

    try:
        perplexity = torch.exp(loss)						# Perplexity == exponentiated cross entropy loss.
    except OverflowError:							# At the beginning of the training loss is super high and we might overflow.
        perplexity = torch.tensor(float('inf'))

    return loss.item(), perplexity.item()


set_seed(args.seed)

# Accelerator
accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size	# This makes sense, each GPU will handle a batch.

# Logging
logger, tb_writer, run_name = setup_logging(project_name.split('/')[1])
logger.info(accelerator.state)

# With Repository we can pull, branch, commit, or push. We can continuous push model checkpoints to the Hub.

from huggingface_hub import Repository

if accelerator.is_main_process:
    hf_repo = Repository('./', clone_from=project_name, revision=run_name)

# Load model and tokenizer

model = AutoModelForCausalLM.from_pretrained ('./', gradient








    



  





















			




















