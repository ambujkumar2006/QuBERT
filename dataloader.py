import os
import re
from collections import Counter

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    BertTokenizerFast,
    RobertaTokenizerFast
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_MAP = {
    -1: 0,   # negative
     0: 1,   # neutral
     1: 2    # positive
}


# ==========================================
# Helpers
# ==========================================

def resolve_data_path(path):
    """Find the CSV in the given path, the current folder or this script's folder."""
    candidates = [path, os.path.join(BASE_DIR, path)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find '{path}'. Looked in: {candidates}. "
        f"Put Twitter_Data.csv next to the scripts or pass --data_path."
    )


# ==========================================
# BiGRU tokenization (own vocabulary)
# ==========================================

def tokenize_text(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(texts, min_freq=2):
    counter = Counter()

    for text in texts:
        counter.update(tokenize_text(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, frequency in counter.items():
        if frequency >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def text_to_ids(text, vocab, max_len):
    unk = vocab["<UNK>"]
    ids = [vocab.get(token, unk) for token in tokenize_text(text)]
    ids = ids[:max_len]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


def encode_bigru_texts(texts, vocab, max_len):
    input_ids = torch.tensor(
        [text_to_ids(t, vocab, max_len) for t in texts],
        dtype=torch.long
    )
    # 1 for real tokens, 0 for padding
    attention_mask = (input_ids != vocab["<PAD>"]).long()
    return input_ids, attention_mask


# ==========================================
# Data loading
# ==========================================

def load_dataframe(data_path, max_samples=0, seed=42):
    df = pd.read_csv(resolve_data_path(data_path))

    for column in ("clean_text", "category"):
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Columns are: {list(df.columns)}"
            )

    df = df.dropna(subset=["clean_text", "category"]).copy()

    df["category"] = df["category"].astype(float).astype(int)
    df = df[df["category"].isin(LABEL_MAP.keys())].copy()
    df["category"] = df["category"].map(LABEL_MAP)
    df["clean_text"] = df["clean_text"].astype(str)

    if 0 < max_samples < len(df):
        df, _ = train_test_split(
            df,
            train_size=max_samples,
            random_state=seed,
            stratify=df["category"]
        )

    return df


def get_dataloaders(
        encoder,
        batch_size,
        max_len=50,
        data_path="Twitter_Data.csv",
        max_samples=0,
        seed=42
):
    """
    Returns: train_loader, val_loader, test_loader, vocab_size
    (vocab_size is only not None for the BiGRU encoder.)
    Split: 80% train / 10% validation / 10% test (stratified).
    Each batch is (input_ids, attention_mask, labels).
    """

    df = load_dataframe(data_path, max_samples, seed)

    texts = df["clean_text"].tolist()
    labels = df["category"].tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=0.20,
        random_state=seed,
        stratify=labels
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels
    )

    print(
        f"Samples -> train: {len(train_texts)}, "
        f"val: {len(val_texts)}, test: {len(test_texts)}"
    )

    vocab_size = None

    if encoder in ("bert", "roberta"):

        if encoder == "bert":
            print("Using BERT tokenizer")
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        else:
            print("Using RoBERTa tokenizer")
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        def encode(split_texts):
            enc = tokenizer(
                split_texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            return enc["input_ids"], enc["attention_mask"]

        train_ids, train_mask = encode(train_texts)
        val_ids, val_mask = encode(val_texts)
        test_ids, test_mask = encode(test_texts)

    elif encoder == "bigru":

        print("Using custom BiGRU vocabulary")

        # Vocabulary is built from the training split only;
        # unseen words in val/test become <UNK>.
        vocab = build_vocab(train_texts)
        vocab_size = len(vocab)
        print(f"BiGRU vocabulary size: {vocab_size}")

        train_ids, train_mask = encode_bigru_texts(train_texts, vocab, max_len)
        val_ids, val_mask = encode_bigru_texts(val_texts, vocab, max_len)
        test_ids, test_mask = encode_bigru_texts(test_texts, vocab, max_len)

    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_ids, train_mask, train_labels)
    val_dataset = TensorDataset(val_ids, val_mask, val_labels)
    test_dataset = TensorDataset(test_ids, test_mask, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, vocab_size
