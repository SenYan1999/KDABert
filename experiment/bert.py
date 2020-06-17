import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args
from model import Transformer
from transformers import BertModel, BertConfig, DistilBertModel

class BertClassification(nn.Module):
    def __init__(self, num_label):
        super(BertClassification, self).__init__()
        model_config = {'hidden_size': 768,'hidden_dropout_prob': 0.2,'num_hidden_layers': 4,'num_attention_heads': 4}
        model_config = BertConfig(**model_config)

        # self.bert = torch.load('../save_models/student_model.pt', map_location=torch.device('cpu')).bert.float()
        self.bert = DistillBertModel.from_pretrained('distillbert-base-uncased')
        self.classification = torch.nn.Linear(768, num_label)
    
    def forward(self, kwargs):
        _, pooler_out = self.bert(**kwargs)
        out = self.classification(pooler_out)
        out = F.log_softmax(out, dim=-1)
        return out

if __name__ == '__main__':
    model = BertClassification(2)
