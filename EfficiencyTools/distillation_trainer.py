from transformers import TrainingArguments
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import Trainer

# We're creating a custom Trainer, in this case a trainer that can use the distillation technique. For the theory see pages 218-219.

# Example of adding two hyperparameters to TrainingArguments.
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)						# We pass on all the rest to the parent constructor.
        self.alpha = alpha
        self.temperature = temperature

# We need to subclass Traininer as well as we need to override the compute_loss() method.
# We're passing on a teach model that has been fully pre-trained on the specific task.
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)						# We pass on all the rest to the parent constructor.
        self.teacher_model = teacher_model  
 
    # Now override compute_loss().
    def compute_loss(self, model, inputs, return_outputs=False):

        # Feed the inputs to the model (unpack dict.) as usual. This is the student model.
        outputs_student = model(**inputs)
        
        # Get the CE loss from the students, as well as the logits.
        CE_loss_student = outputs_student.loss
        logits_student = outputs_student.logits

        # Get the logits from the teacher.
        with torch.no_grad():								# We're only training the student.
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits

        # Apply temperature-ed softmax. nn.KLDivLoss() wants log_prob for inputs (student) and prob for labels (teacher).
        temp_softened_log_probs_student = F.log_softmax(logits_student / self.args.temperature, dim = -1)
        temp_softened_probs_teacher = F.softmax(logits_teacher / self.args.temperature, dim = -1)

        loss_function = nn.KLDivLoss(reduction = 'batchmean')				# We average losses over the batch dim.

        KD_loss_student = self.args.temperature ** 2 * loss_function(	temp_softened_log_probs_student,
								temp_softened_probs_teacher)	
        
        # Now blend the CE_loss and KD_loss using alpha.
        loss_student = self.args.alpha * CE_loss_student + (1. - self.args.alpha) * KD_loss_student 

        return (loss_student, outputs_student) if return_outputs else loss_student
