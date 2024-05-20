from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_dataset, ReadInstruction
import datasets
import transformers
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from constantlengthdataset import ConstantLengthDataset
import torch

model_ckpt = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained("tokenizers/" + model_ckpt)				# Our retrained, saved tokenizer. 

##### Code above is all that's needed to retrain HF tokenizer. python_tokenizer.py has all the details and comments.

################################# Train a HF model from scratch ################################

from transformers import AutoConfig, AutoModelForCausalLM, AdamW
from accelerate.utils import set_seed
from accelerate import Accelerator

project_name = 'gpt2'

# Note: if you would have saved the retrained tokenizer to the hub, it would have a ckpt name and you would load it like this:
# tokenizer = AutoTokenizer.from_pretrained(model_cpkt)  	# With model_ckpt being the name you used when saving your retrained tokenizer.

# This would be a different model_ckpt than the one you used to save the retrained tokenizer.
# In our case this is gpt2 as per the above, so the untrained model we're going to load to then retrain.

# This is how you load an untrained model from the hub when you just have make some config changes (in this case the vocab size).
config = AutoConfig.from_pretrained(model_ckpt, vocab_size=len(tokenizer))		
model = AutoModelForCausalLM.from_config(config)

# Saving a model to a file.

model.save_pretrained("models/" + model_ckpt, push_to_hub=False, organization="transformersbook")

# At this point we have a reconfigured, completely untrailed gpt-2 model locally saved.

################################################ Accelerate ##############################################

# 1. Set up hyperparameters.

from argparse import Namespace

config = {	'train_batch_size'		: 2,
		'valid_batch_size'		: 2,

                # Countering overfitting:
                # Weight decay is equivalent to L2 regularition. In general there are these forms of regularization:
                # L1/L2 regularization (with L2 being equiv. to weight decay).
                # Drop out regularization.
                # Early stopping -> look at training and validation loss and stop if validation loss is going up again (congtroversial).
                # Data augmentation == feeding in more data, but this might not always be possible. For images it's easier since we can rotate
                # images etc.
		'weight_decay'			: 0.1,

		'shuffle_buffer'		: 1000,
		'learning_rate'			: 2e-4,
		'lr_scheduler_type'		: 'cosine',
		'num_warmup_steps'		: 1,						# Edu mode, put to 750 or so.
		'gradient_accumulation_steps'	: 1,						# Edu mode, put to 16 or so.
		'max_train_steps'		: 1,						# Edu mode, put to 50000 or so.
		'max_eval_steps'		: 1,						# Edu mode, set to -1.
		'seq_len'			: 1024,
		'seed'				: 1,
		'save_checkpoint_steps'		: 1,						# Edu mode, put to 5000 or so.
                'gradient_checkpointing'	: True		}

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
        wandb.init(project=project_name, config=args, anonymous='allow')
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

## Estimate how many Unicode characters we typically have per token, based on 500 samples.
dataset_name = 'transformersbook/codeparrot-train'
#dataset_full = load_dataset(dataset_name, split='train', streaming=True)	
#dataset = dataset_full.take(500)
#
#samples, total_characters, total_tokens = 500, 0, 0
#for _,sample in tqdm(zip(range(samples), iter(dataset)), total=samples):
#    total_characters += len(sample['content'])
#    total_tokens += len(tokenizer(sample['content']).tokens())
#characters_per_token = total_characters / total_tokens
#print(characters_per_token)
## This gives about 3.6 for characters_per_token.

## Run a test with characters_per_token in our call to ConstantLengthDataset.
#shuffled_dataset = dataset.shuffle(buffer_size=100)
#constant_length_dataset = ConstantLengthDataset(tokenizer, shuffled_dataset, num_of_seqs=10, chars_per_token=characters_per_token, seq_len=1014)
#dataset_iterator = iter(constant_length_dataset)
#lengths = [len(b) for _,b in zip(range(5), dataset_iterator)]
#print(lengths)

from torch.utils.data.dataloader import DataLoader
    
def create_dataloaders(dataset_name):
     
    # The dataset we're using only has a 'train' split, so we need to split that into a 'train' and a 'valid' subsplit.
    # The set is huge so we need to use streaming mode, so we'll use the first 500 for validation and the rest for training.
    # This is done with take() and skip(), and we have to run shuffle before we start taking and skipping.
    # Technically though only the training set needs to be shuffled.

    dataset_full_unshuffled = load_dataset(dataset_name, split='train', streaming=True)
    dataset_full = dataset_full_unshuffled.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    valid_data = dataset_full.take(500)
    train_data = dataset_full.skip(500)
    
    ## Original code - use if you have 'train' and 'validation' splits.
    #train_data_unshuffled = load_dataset(dataset_name, split='train')							# Removed ',streaming=True'.
    #train_data = train_data_unshuffled.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)				# Note the shuffle.
    #valid_data = load_dataset(dataset_name, split='validation')							# Removed ',streaming=True'.

    # Ideally num_of_seq and chars_per_token are also in the args.
    train_dataset = ConstantLengthDataset(tokenizer, train_data, num_of_seqs=10, chars_per_token=3.6, seq_len = args.seq_len)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data, num_of_seqs=10, chars_per_token=3.6, seq_len = args.seq_len)

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


# 6. Core training code.

set_seed(args.seed)

# Get an accelerator.
#####################

accelerator = Accelerator()
samples_per_step = accelerator.state.num_processes * args.train_batch_size	# Samples over all the GPUs per step (which is gradient acc.-driven here).

# Set up logging.
#################

#logger, tb_writer, run_name = setup_logging(project_name.split('/')[1])
logger, tb_writer, run_name = setup_logging(project_name)
logger.info(accelerator.state)

## With Repository we can pull, branch, commit, or push. We can continuous push model checkpoints to the Hub.
#from huggingface_hub import Repository
#if accelerator.is_main_process:
#    hf_repo = Repository('./', clone_from=project_name, revision=run_name)

# Load/get model and tokenizer.
###############################

# Activation checkpointing (or gradient checkpointing) is a technique to reduce memory usage by clearing activations of certain layers and 
# recomputing them during a backward pass. Effectively, this trades extra computation time for reduced memory usage.
# Note that gradient checkpoints has nothing to do with gradient accumulation (the latter being the 'construction' of the gradient over multiple batches).
#model = AutoModelForCausalLM.from_pretrained ("models/" + model_ckpt, gradient_checkpointing=True)	# The untrained model with new config we have saved.
model = AutoModelForCausalLM.from_pretrained ("models/" + model_ckpt)					# The untrained model with new config we have saved.

#tokenizer = AutoTokenizer.from_pretrained("tokenizers/" + model_ckpt)					# Our retrained tokenizer. We still have the reference here.

# Get dataloaders.
##################

train_dataloader, eval_dataloader = create_dataloaders(dataset_name)

# Set up optimizer and learning rate scheduler.
###############################################

optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)

from transformers.optimization import get_scheduler

# A learning rate schedule changes the learning rate during learning and is most often changed between epochs/iterations. This is mainly done 
# with two parameters: decay and momentum. There are many different learning rate schedules but the most common are time-based, step-based and exponential.
lr_scheduler = get_scheduler(	name			= args.lr_scheduler_type,
				optimizer		= optimizer,
				num_warmup_steps 	= args.num_warmup_steps,
				num_training_steps	= args.max_train_steps)

# Make model, optimizer and dataloaders	aware of the fact that we're using Accelerate.
######################################################################################

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

# Train the model.
##################

model.train()							

completed_steps = 0

# This code currently cannot train over multiple epochs.
for step, batch in enumerate(train_dataloader, start=1):
    
    loss = model(batch, labels=batch).loss							# gpt-2 model understands what it needs as inputs and labels.

    log_metrics(step, {'lr': optimizer.param_groups[0]['lr'], 'samples': step*samples_per_step, 'steps': completed_steps, 'loss/train': loss.item()})

    loss = loss / args.gradient_accumulation_steps						# As we're doing gradient accumulation.

    # loss.backward() (or in this case accelerator.backward(loss) computes dloss/dx for every parameter x which has requires_grad=True. 
    # These are accumulated into x.grad for every parameter x.
    # optimizer.step() updates the value of x using the gradient x.grad.
    accelerator.backward(loss)

    if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1

    if step % args.save_checkpoint_steps == 0:
        logger.info('Evaluating and savings model checkpoint')
        eval_loss, perplexity = evaluate()
        log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})

        # With Accelerate, always perform these two steps before savings a model to make sure it's properly synchronized.
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        #if accelerator.is_main_process:
        #    unwrapped_model.save_pretrained('./')
        #    hf_repo.push_to_hub(commit_message=f'step {step}'        
        
        model.train()

    if completed_steps >= args.max_train_steps:
        break
    
# Evaluate and save the last checkpoint.
logger.info('Evaluating and saving model after training')
eval_loss, perplexity = evaluate()
log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity})

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)


#if accelerator.is_main_process:
#    unwrapped_model.save_pretrained('./')
#    hf_repo.push_to_hub(commit_message=f'final model'        













































    



  





















			




















