
import torch.nn as nn

class ConditionEncoder(nn.Module):
    def __init__(self, embedding_dim=768, seq_len=77):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, labels):
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        out = self.net(labels)
        return out.unsqueeze(1).repeat(1, self.seq_len, 1)
