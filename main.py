import json
import re
import tensorflow as tf
import pickle
import torch
import numpy as np
import os
from gpt_model import GPTModel, generate_text
from train import text_to_token_ids, token_ids_to_text

from utils import load_gpt2_params_from_tf_ckpt, print_params_structure, load_weights_to_gpt

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

load_path = "./gpt2/gpt2_encoder.pkl"
with open(load_path, "rb") as f:
    tokenizer = pickle.load(f)
model_dir = "./gpt2/124M"
settings = json.load(open(os.path.join(model_dir, "hparams.json")))
# print(settings)
tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
# print("="*40 + " GPT-2参数结构 " + "="*40)
# print_params_structure(params)
# print("="*90)
model_configs = {
    "gpt2-small(124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2-small(124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)

load_weights_to_gpt(gpt, params)
gpt.to(device)
gpt.eval()
torch.manual_seed(123)
token_ids = generate_text(
    model=gpt, idx=text_to_token_ids("Every effort moves you", tokenizer=tokenizer).to(device),
    max_new_tokens=25, context_size=GPT_CONFIG_124M["context_length"], top_k=50, temperature=1.0
)
print(token_ids_to_text(token_ids, tokenizer))
