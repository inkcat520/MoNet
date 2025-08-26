import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.Attention import AttentionBlock
from module.Mlp import MLPNormLayer, ResMLPLayer, MlpLayer
from module.Gen import GenLayer
from module.Emb import PatchEmbed, PatchEmbed_M, PositionEmbed
from utils.kinematics import my_correlation
from utils.data_utils import dct_matrix


class ModelLayer(nn.Module):
    def __init__(self, d_joint, d_seq, d_pre, d_ff=512, d_emb=64, d_heads=8,
                 drop_out=0.1, before_nums=1, after_nums=1, patch_size=4):
        super(ModelLayer, self).__init__()

        self.d_joint = d_joint
        self.d_seq = d_seq
        self.d_pre = d_pre
        self.d_len = self.d_seq
        self.d_ff = d_ff
        self.d_emb = d_emb
        self.chans = 3
        self.patch_size = patch_size
        self.patch_num = ((self.d_len + 1) // 2 - self.patch_size) // self.patch_size + 1   # self.d_len // self.patch_size
        self.d_model = self.d_joint * self.chans
        self.d_heads = d_heads
        self.act = "relu"

        self.dct_m = dct_matrix(self.d_len)[0]
        self.idct_m = dct_matrix(self.d_pre)[-1]

        self.cor_agg = CorAgg(self.d_model)

        self.pe = nn.Parameter(torch.zeros((1, self.patch_num, d_emb)))
        self.patch_emb = PatchEmbed_M(self.d_len, self.patch_size, self.d_emb)

        self.tem_attn = nn.ModuleList([AttentionBlock(d_emb, d_ff, d_heads, mask_flag=False, dropout=drop_out, activation=self.act)
                                       for _ in range(after_nums)])

        self.tem_emb = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.LayerNorm(d_emb)
        )

        self.trans = nn.Linear(self.d_emb, self.patch_size)

        self.attn_pro = nn.Linear(self.patch_num * self.patch_size, self.d_seq)

        # self.mlp = nn.Sequential(*[ResMLPLayer(self.d_seq, self.d_seq, self.d_ff) for _ in range(6)])
        self.mlp = nn.Sequential(*[MLPNormLayer(self.d_model, self.d_seq) for _ in range(before_nums)])
        self.mlp_pro = nn.Linear(self.d_seq, self.d_seq)

        self.make = MlpLayer(2 * self.d_seq, self.d_seq, d_ff, activation=self.act)
        self.gen = GenLayer(self.d_seq, self.d_pre, self.d_model)

    def forward(self, x):
        B, T, C = x.shape
        attns = []
        last_seq = x[:, -1:].detach().clone()

        x = torch.matmul(self.dct_m, x)
        cor = my_correlation(x.permute(0, 2, 1))
        x = self.cor_agg(x, cor)
        x = x.permute(0, 2, 1)

        y = self.mlp(x)
        y = self.mlp_pro(y)  # B C E

        z = self.patch_emb(x)  # BC N E
        z = z + self.pe
        z = self.tem_emb(z)

        for layer in self.tem_attn:
            z, attn = layer(z, z)
            attns.append(attn)

        z = self.trans(z)
        z = z.flatten(1).view(B, C, -1)  # B C D
        z = self.attn_pro(z)

        out = torch.cat((z, y), dim=-1)
        out = self.make(out)
        out = self.gen(out).permute(0, 2, 1)
        out = torch.matmul(self.idct_m, out)
        out = out + last_seq

        return out, attns


class CorAgg(nn.Module):
    def __init__(self, d_model):
        super(CorAgg, self).__init__()
        self.emb = nn.Linear(d_model, d_model)
        self.pro = nn.Linear(d_model, d_model)
        self.att = nn.Parameter(torch.zeros(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cor):
        z = self.emb(x)
        z = torch.matmul(z, cor + self.att)
        z = self.pro(z)
        z = self.norm(z + x)
        return z
