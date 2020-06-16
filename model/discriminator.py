import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        # we use a multi-layer-perception to discriminate
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 2)
        )
        
    def forward(self, output):
        output = output.permute(0, 2, 1)
        output = F.max_pool1d(output, kernel_size=output.shape[-1]).squeeze()
        logits = self.mlp(output)
        logits = F.log_softmax(logits, dim=-1)
        return logits
        