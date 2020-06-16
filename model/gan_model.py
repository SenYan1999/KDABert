import torch
import torch.nn as nn
import torch.nn.functional as F

from .student_model import Transformer
from .discriminator import Discriminator
from transformers import BertModel

class KDABert(nn.Module):
    def __init__(self, student=None, teacher=None, discriminator=None, config=None, bert_name=None, batch_size=None, device=None):
        super(KDABert, self).__init__()

        if(student == None and teacher == None and discriminator == None):
            assert(config != None and bert_name != None)

        self.student = student if student != None else Transformer(config)
        self.teacher = teacher if teacher != None else BertModel.from_pretrained(bert_name)
        self.discriminator = discriminator if discriminator != None else Discriminator(config['hidden_size'])

        self.valid = torch.nn.Parameter(torch.ones(batch_size), requires_grad=False).long().to(device)
        self.fake = torch.nn.Parameter(torch.zeros(batch_size), requires_grad=False).long().to(device)

    def forward(self, kwargs, task):
        student_last_hidden, student_logits = self.student(kwargs, task=task)

        self.teacher.eval()
        teacher_last_hidden, _ = self.teacher(**kwargs)

        return student_logits, (student_last_hidden, teacher_last_hidden)

    def get_student_loss(self, logits, label):
        return self.student.loss_fn(logits, label)
    
    def get_discriminator_loss(self, logits, label):
        logits = self.discriminator(logits)
        return F.nll_loss(logits, label)

    def get_loss_for_training_discriminator(self, student, teacher):
        loss = (self.get_discriminator_loss(student.detach(), self.fake) + \
                self.get_discriminator_loss(teacher, self.valid)) / 2

        return loss

    def get_loss_for_training_student(self, student, student_logits, label):
        loss = (self.get_student_loss(student, label) + \
                self.get_discriminator_loss(student_logits, self.valid)) / 2

        return loss
