import torch
import torch.nn as nn
import torch.nn.functional as F

class GenLayer(nn.Module):

    def __init__(self, d_model, pred_len, channels):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.channels = channels

        self.Linear = nn.ModuleList()
        for i in range(self.channels):
            self.Linear.append(nn.Linear(self.d_model, self.pred_len))

        for i in range(len(self.Linear)):
            nn.init.constant_(self.Linear[i].weight, 0)
            nn.init.constant_(self.Linear[i].bias, 0)

    def forward(self, x):  # B C T
        output = torch.zeros([x.size(0), x.size(1), self.pred_len], dtype=x.dtype, device=x.device)
        for i in range(self.channels):
            output[:, i] = self.Linear[i](x[:, i])
        return output

