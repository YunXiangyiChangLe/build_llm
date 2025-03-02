import pandas as pd
import torch
from torch.utils.data import DataLoader
import tensorflow as tf
from gpt_model import GPTModel, generate_text
from train import text_to_token_ids, token_ids_to_text
from utils import create_balanced_dataset, random_spilt, load_gpt2_params_from_tf_ckpt, load_weights_to_gpt, \
    calc_accuracy_loader
from dataset import SpamDataset
import pickle
import json
import os


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    print(logits)
    print(target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batchs=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batchs is None:
        num_batchs = len(data_loader)
    else:
        num_batchs = min(num_batchs, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batchs:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batchs


data_path = "./dataset/SMSSpamCollection"
df = pd.read_csv(data_path, sep="\t", header=None, names=["Label", "Text"])
df = create_balanced_dataset(df)
# print(df["Label"].value_counts())
# df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
# print(df)
# train_df, val_df, test_df = random_spilt(df, 0.7, 0.1)
# train_df.to_csv("./dataset/train.csv", index=None)
# val_df.to_csv("./dataset/val.csv", index=None)
# test_df.to_csv("./dataset/test.csv", index=None)

load_path = "./gpt2/gpt2_encoder.pkl"
with open(load_path, "rb") as f:
    tokenizer = pickle.load(f)

train_dataset = SpamDataset(csv_file="./dataset/train.csv",
                            tokenizer=tokenizer)
val_dataset = SpamDataset(csv_file="./dataset/val.csv",
                          tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file="./dataset/test.csv",
                           tokenizer=tokenizer)
num_workers = 0
batch_size = 8
torch.manual_seed(123)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size,
    shuffle=True, num_workers=num_workers, drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size,
    num_workers=num_workers, drop_last=False
)
test_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size,
    num_workers=num_workers, drop_last=False
)
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

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

assert train_dataset.max_length <= NEW_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's "
    f"context {NEW_CONFIG['context_length']}"
)

model_dir = "./gpt2/124M"
tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
settings = json.load(open(os.path.join(model_dir, "hparams.json")))
params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
load_weights_to_gpt(gpt, params)

# text_2 = (
#     "Is the following text 'spam'? Answer with 'yes' or 'no': "
#     "'You are a winner you have been specially selected to receive $1000 cash or a $2000 award"
# )
# token_ids = generate_text(
#     model=gpt, idx=text_to_token_ids(text_2, tokenizer=tokenizer).to(device),
#     max_new_tokens=23, context_size=GPT_CONFIG_124M["context_length"]
# )
# print(token_ids_to_text(token_ids, tokenizer))

num_classes = 2
for param in gpt.parameters():
    param.requires_grad = False
gpt.out_head = torch.nn.Linear(
    in_features=NEW_CONFIG["emb_dim"],
    out_features=num_classes
)
for param in gpt.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in gpt.final_norm.parameters():
    param.requires_grad = True
gpt.to(device)
# inputs = tokenizer.encode("Do you have time")

torch.manual_seed(123)
train_acc = calc_accuracy_loader(train_loader, gpt, device, num_batches=10)
val_acc = calc_accuracy_loader(val_loader, gpt, device, num_batches=10)
test_acc = calc_accuracy_loader(test_loader, gpt, device, num_batches=10)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, gpt, device, num_batchs=5)
    val_loss = calc_loss_loader(val_loader, gpt, device, num_batchs=5)
    test_loss = calc_loss_loader(test_loader, gpt, device, num_batchs=5)

print("init stats")
print(f"train acc {train_acc * 100:.2f}%")
print(f"val acc {val_acc * 100:.2f}%")
print(f"test acc {test_acc * 100:.2f}%")
print(f"train loss {train_loss :.3f}")
print(f"val loss {val_loss :.3f}")
print(f"test loss {test_loss :.3f}")
