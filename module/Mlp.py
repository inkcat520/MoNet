import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLPLayer(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, dropout_rate=0.1, activation="relu"):
        super(ResMLPLayer, self).__init__()
        self.lin_a = nn.Linear(input_dim, hidden_dim)
        self.lin_b = nn.Linear(hidden_dim, output_dim)
        self.lin_res = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, inputs):

        h_state = self.activation(self.lin_a(inputs))
        out = self.lin_b(h_state)
        out = self.dropout(out)
        res = self.lin_res(inputs)
        out = self.norm(out + res)
        return out

class MlpLayer(nn.Module):
    def __init__(self, d_in, d_out, d_ff, activation="relu"):
        super(MlpLayer, self).__init__()
        self.fc1 = nn.Linear(d_in, d_ff)
        self.fc2 = nn.Linear(d_ff, d_out)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class FFNLayer(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        new_x = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = x + self.dropout(new_x)
        x = self.norm(x)
        return x


class MLPNormLayer(nn.Module):

    def __init__(self, d_model, d_len):
        super().__init__()

        self.pro = nn.Linear(d_len, d_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        y = self.pro(x)
        y = y.permute(0, 2, 1)
        y = self.norm(y)
        y = y.permute(0, 2, 1)
        x = x + y

        return x