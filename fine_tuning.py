import pandas as pd
from utils import create_balanced_dataset, random_spilt

data_path = "./dataset/SMSSpamCollection"
df = pd.read_csv(data_path, sep="\t", header=None, names=["Label", "Text"])
df = create_balanced_dataset(df)
# print(df["Label"].value_counts())
df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
# print(df)
train_df, val_df, test_df = random_spilt(df, 0.7, 0.1)
train_df.to_csv("./dataset/train.csv", index=None)
val_df.to_csv("./dataset/val.csv", index=None)
test_df.to_csv("./dataset/test.csv", index=None)
