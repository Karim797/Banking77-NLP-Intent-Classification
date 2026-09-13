import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SEED = 42
CHECKPOINT = "distilbert-base-uncased"
BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
OUTPUT_DIR = Path("artifacts/distilbert")
MAX_LENGTH = 64
BATCH_SIZE = 32
EPOCHS = 4


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = {name: values[index] for name, values in self.encodings.items()}
        row["labels"] = self.labels[index]
        return row


def score(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            truth = batch.pop("labels")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(truth.tolist())
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
    }


def main():
    seed_everything()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} threads={torch.get_num_threads()}", flush=True)

    train_df = pd.read_csv(f"{BASE_URL}/train.csv")
    test_df = pd.read_csv(f"{BASE_URL}/test.csv")
    intents = sorted(train_df["category"].unique())
    name_to_id = {name: index for index, name in enumerate(intents)}
    train_text, val_text, train_labels, val_labels = train_test_split(
        train_df["text"].tolist(),
        train_df["category"].map(name_to_id).to_numpy(),
        test_size=0.20,
        random_state=SEED,
        stratify=train_df["category"],
    )
    test_text = test_df["text"].tolist()
    test_labels = test_df["category"].map(name_to_id).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def encode(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    train_loader = DataLoader(EncodedDataset(encode(train_text), train_labels), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EncodedDataset(encode(val_text), val_labels), batch_size=BATCH_SIZE * 2)
    test_loader = DataLoader(EncodedDataset(encode(test_text), test_labels), batch_size=BATCH_SIZE * 2)

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(intents),
        id2label={index: name for index, name in enumerate(intents)},
        label2id=name_to_id,
    )
    for parameter in model.distilbert.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.distilbert.transformer.layer[:2]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    model.to(device)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable_parameters={trainable:,}/{total:,}", flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-5, weight_decay=0.01)

    started = time.time()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += output.loss.item()
            if step % 50 == 0 or step == len(train_loader):
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss / step:.4f}", flush=True)
        validation = score(model, val_loader, device)
        validation["epoch"] = epoch
        validation["train_loss"] = running_loss / len(train_loader)
        history.append(validation)
        print("validation=" + json.dumps(validation), flush=True)

    test_metrics = score(model, test_loader, device)
    elapsed = time.time() - started
    metrics = {
        "checkpoint": CHECKPOINT,
        "device": str(device),
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "frozen_layers": 2,
        "train_examples": len(train_text),
        "validation_examples": len(val_text),
        "test_examples": len(test_text),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_seconds": elapsed,
        "validation_history": history,
        "test": test_metrics,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("test=" + json.dumps(test_metrics), flush=True)
    print(f"saved={OUTPUT_DIR / 'metrics.json'} elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
