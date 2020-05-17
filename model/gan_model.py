import torch
import torch.nn as nn
import torch.nn.functional as F

from .student_model import Transformer
from .discriminator import Discriminator
from transformers import BertModel

class KDABert(nn.Module):
    def __init__(self, student=None, teacher=None, discriminator=None, config=None, bert_name=None):
        super(KDABert, self).__init__()

        if(student == None and teacher == None and discriminator == None):
            assert(config != None and bert_name != None)

        self.student = student if student != None else Transformer(config)
        self.teacher = teacher if teacher != None else BertModel.from_pretrained(bert_name)
        self.discriminator = discriminator if discriminator != None else Discriminator(config['hidden_size'])

    def forward(self, kwargs):
        student_pooler_out, student_logits = self.student(kwargs)

        self.teacher.eval()
        teacher_hidden_states, teacher_pooler_out = self.teacher(kwargs)

        return student_logits, (student_pooler_out, teacher_pooler_out)

    def get_student_loss(self, logits, label):
        return self.student.loss_fn(logits, label)
    
    def get_discriminator_loss(self, logits, label):
        return F.nll_loss(logits, label)

    def get_loss_for_training_discriminator(self, student, teacher):
        valid = torch.ones(teacher.shape[0]).long().to(teacher)
        fake = torch.zeros(teacher.shape[0]).long().to(teacher)

        loss = (self.get_discriminator_loss(student.detach(), fake) + \
                self.get_discriminator_loss(teacher, valid)) / 2

        return loss

    def get_loss_for_training_student(self, student, label):
        valid = torch.ones(student.shape[0]).long().to(student)
        loss = (self.get_student_loss(student, label) + \
                self.get_discriminator_loss(student, valid)) / 2

        return loss
