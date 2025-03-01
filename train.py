import re

import torch
from self_attention import CausalSelfAttention, MultiHeadAttentionWrapper, MutilHeadAttention
from tokenizer import SimpleTokenizer
from importlib.metadata import version
import tiktoken
import pickle
from dataset import create_dataloader
from gpt_model import GPTModel, FeedForward, TransformerBlocker, generate_text
import matplotlib.pyplot as plt

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

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

train_ratio = 0.90
split = int(train_ratio * len(text_data))
train_data = text_data[:split]
val_data = text_data[split:]
train_loader = create_dataloader(
    txt=train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=0
)
val_loader = create_dataloader(
    txt=val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=False,
    drop_last=False,
    num_workers=0
)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epoches,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epoches):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch=input_batch, target_batch=target_batch,
                                   model=model, device=device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (step {global_step:05d}):"
                      f"train loss {train_loss:.3f}, val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def token_ids_to_text(ids, tokenizer):
    flat = ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        tokens_ids = generate_text(model, encoded, 50, context_size)
        decode_text = token_ids_to_text(tokens_ids, tokenizer)
        print(decode_text.replace("\n", " "))
    model.train()


# def plot_losses(epoches_seen, tokens_seen, train_losses, val_losses):
#     # 确保数据在 CPU 上并转换为 NumPy
#     epoches_seen = epochs_tensor.detach().cpu().numpy()
#     train_losses = [loss.detach().cpu().numpy() for loss in train_losses]
#     val_losses = [loss.detach().cpu().numpy() for loss in val_losses]
#     tokens_seen = [t.detach().cpu().numpy() for t in tokens_seen]
#
#     fig, ax1 = plt.subplots(figsize=(5, 3))
#     ax1.plot(epoches_seen, train_losses, label="train loss")
#     ax1.plot(epoches_seen, val_losses, linestyle="-.", label="val loss")
#     ax1.set_xlabel("epochs")
#     ax1.set_ylabel("loss")
#     ax1.legend(loc="upper right")
#     ax2 = ax1.twiny()
#     ax2.plot(tokens_seen, train_losses, alpha=0)
#     ax2.set_xlabel("tokens seen")
#     fig.tight_layout()
#     plt.show()


# torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
    device=device, num_epoches=num_epochs, eval_freq=5, eval_iter=1, start_context="Every effort moves you",
    tokenizer=tokenizer
)

# epochs_tensor=torch.linspace(0,num_epochs,len(train_losses))
# plot_losses(epochs_tensor,tokens_seen,train_losses,val_losses)

model.eval()
token_ids = generate_text(
    model=model, idx=text_to_token_ids("Every effort moves you", tokenizer=tokenizer).to(device),
    max_new_tokens=25, context_size=GPT_CONFIG_124M["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
