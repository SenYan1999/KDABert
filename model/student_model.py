import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig

class Transformer(nn.Module):
    def __init__(self, bert_config, num_labels=0):
        super(Transformer, self).__init__()

        config = BertConfig(**bert_config)
        self.bert = BertModel(config)
        self.nsp = nn.Linear(config.hidden_size, 2)
        self.mlm = nn.Linear(config.hidden_size, config.vocab_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, kwargs, task='downstream'):
        last_hidden_state, pooler_output = self.bert(**kwargs)
        
        if task == 'nsp':
            logits = self.nsp(pooler_output)
            logits = F.log_softmax(logits, dim=-1)
        elif task == 'mlm':
            logits = self.mlm(last_hidden_state)
            logits = F.log_softmax(logits)
        else:
            logits = self.classifier(pooler_output)
            logits = F.log_softmax(logits, dim=-1)

        return last_hidden_state, logits
    
    def loss_fn(self, logits, label):
        return F.nll_loss(logits, label, ignore_index=-1)
