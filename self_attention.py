import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, qkv_bias=False):
        super().__init__()
        torch.manual_seed(123)
        self.d_out = d_out
        self.d_in = d_in
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(0.5)
        self.register_buffer('mask', torch.triu(torch.ones(context_length,
                                                           context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        atten_sorces = queries @ keys.transpose(1, 2)
        atten_sorces.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        atten_weights = torch.softmax(atten_sorces / keys.shape[-1] ** 0.5, dim=-1)
        self.dropout(atten_weights)
        context_vecs = atten_weights @ values
        return context_vecs
