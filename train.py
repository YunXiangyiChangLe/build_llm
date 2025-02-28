import re

import torch
from self_attention import CausalSelfAttention, MultiHeadAttentionWrapper, MutilHeadAttention
from tokenizer import SimpleTokenizer
from importlib.metadata import version
import tiktoken
import pickle
from dataset import create_dataloader
from gpt_model import GPTModel, FeedForward, TransformerBlocker, generate_text

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

load_path = "/home/shaozg/build_llm/gpt2/gpt2_encoder.pkl"
with open(load_path, "rb") as f:
    tokenizer = pickle.load(f)
