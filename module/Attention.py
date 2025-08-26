import math
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt


class AttentionBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, n_head=8, mask_flag=False, dropout=0.1, activation="relu"):
        super(AttentionBlock, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.cross_attention = Attentionlayer(FullAttention, d_model, n_head, mask_flag, dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, cross_mask=None):
        new_x, attn = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )
        x = new_x + x

        y = x = self.norm1(x)
        y = self.activation(self.conv1(y.transpose(-1, 1)))
        y = self.conv2(y).transpose(-1, 1)

        return self.norm2(x + y), attn


class Attentionlayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, mask_flag=False, dropout=0.1):
        super(Attentionlayer, self).__init__()

        sub_model = math.ceil(d_model / n_heads)
        sub_model = sub_model if sub_model % 2 == 0 else sub_model + 1
        d_keys = sub_model
        d_values = sub_model

        self.inner_attention = attention(mask_flag, dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -torch.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A


class SimpleAttention(nn.Module):
    def __init__(self, d_model, out_dim, dropout=0.1):
        super(SimpleAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, out_dim)

    def forward(self, queries, keys, values):
        B, L, Q = queries.shape
        _, S, E = keys.shape
        scale = 1. / sqrt(E)

        queries = self.query_projection(queries).view(B, L, -1)
        keys = self.key_projection(keys).view(B, S, -1)
        values = self.value_projection(values).view(B, S, -1)
        scores = torch.einsum("ble,bse->bls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bls,bsd->bld", A, values)

        out = V.contiguous().view(B, L, -1)

        return torch.softmax(self.out_projection(out), dim=-1)

class AdjacentMask:
    def __init__(self, A, L, S, adj):
        """
        B: Batch size
        L: Size of the matrix (L x L)
        true_positions: List of positions to set as True, e.g., [(0, 1), (1, 2), ...]
        device: Device to create the mask on
        """
        with torch.no_grad():
            B, C, _ = adj.shape
            mask = adj.unsqueeze(-1).repeat(1, 1, 1, L // C).flatten(-2)
            mask = mask.permute(0, 2, 1)
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, S // C).flatten(-2)
            mask = mask.repeat(A//B, 1, 1)
            mask = mask.unsqueeze(1)
            self._mask = ~ mask

    @property
    def mask(self):
        return self._mask


class CausalMask:
    def __init__(self, L, layers, device="cpu"):

        mask_shape = [1, 1, L, L]

        mask = torch.ones(mask_shape, dtype=torch.bool, device=device)
        mask1 = torch.triu(mask, diagonal=1)
        mask2 = torch.tril(mask, diagonal=- layers - 1)
        mask = mask1 | mask2
        self._mask = mask

    @property
    def mask(self):
        return self._mask


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
