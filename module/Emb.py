import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, d_len=50, patch_size=3, embed_dim=48):
        super(PatchEmbed, self).__init__()
        self.d_len = d_len
        self.patch_size = patch_size
        self.patch_num = self.d_len // self.patch_size
        self.embed_dim = embed_dim
        self.drop_size = self.d_len % self.patch_size

        self.patch = nn.Linear(self.patch_size, self.embed_dim)

    def forward(self, x):
        x = x[..., self.drop_size:]
        x = x.reshape(-1, self.patch_num, self.patch_size)
        x = self.patch(x)
        return x


class PatchEmbed_M(nn.Module):

    def __init__(self, d_len=50, patch_size=4, embed_dim=48):
        super().__init__()

        self.d_len = d_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        self.patch = nn.Conv1d(64, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = x.reshape(-1, 1, self.d_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.patch(x)
        x = x.permute(0, 2, 1)

        return x


class PositionEmbed(nn.Module):
    def __init__(self, d_emb, sp_num, tem_num):
        super().__init__()
        self.tem_pe = nn.Parameter(torch.zeros((tem_num, d_emb//2)))
        self.sp_pe = nn.Parameter(torch.zeros((sp_num, d_emb//2)))
        self.sp_num = sp_num
        self.tem_num = tem_num

    def forward(self, x):
        BC, N, E = x.shape
        sp_pe = self.sp_pe.unsqueeze(1).repeat(1, self.tem_num, 1)
        tem_pe = self.tem_pe.unsqueeze(0).repeat(self.sp_num, 1, 1)
        pe = torch.cat((sp_pe, tem_pe), dim=-1)
        x = x + pe.repeat(BC//self.sp_num, 1, 1)
        return x
