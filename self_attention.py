import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        atten_sorces = queries @ keys.T
        atten_weights = torch.softmax(atten_sorces / keys.shape[-1] ** 0.5, dim=-1)
        context_vecs = atten_weights @ values
        return context_vecs
