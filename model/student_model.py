import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig

class Transformer(nn.Module):
    def __init__(self, bert_config, num_labels):
        super(Transformer, self).__init__()

        config = BertConfig(**bert_config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)

    def forward(self, kwargs):
        last_hidden_state, pooler_output = self.bert(**kwargs)
        logits = self.classifier(pooler_output)
        logits = F.log_softmax(logits, dim=-1)

        return pooler_output, logits
    
    def loss_fn(self, logits, label):
        return F.nll_loss(logits, label)
