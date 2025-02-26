import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        atten_sorces = queries @ keys.T
        atten_weights = torch.softmax(atten_sorces / keys.shape[-1] ** 0.5, dim=-1)
        context_vecs = atten_weights @ values
        return context_vecs
