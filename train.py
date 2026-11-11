import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

from config import get_args
from dataloader import get_dataloaders
from model2 import SentimentModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_raw=False):

    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    total_loss, total_items = 0.0, 0

    for input_ids, attention_mask, y_batch in loader:

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y_batch = y_batch.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * y_batch.size(0)
        total_items += y_batch.size(0)

        all_preds.extend(torch.argmax(outputs, dim=1).tolist())
        all_labels.extend(y_batch.tolist())
        all_probs.extend(torch.softmax(outputs, dim=1).cpu().tolist())

    metrics = {
        "loss": total_loss / max(total_items, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    if return_raw:
        return metrics, all_labels, all_preds, all_probs

    return metrics


def main():

    args = get_args()

    set_seed(args.seed)
    device = get_device(args.device)

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints", args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")

    print(f"Experiment : {args.experiment_name}")
    print(f"Encoder    : {args.encoder}")
    print(f"Model      : {args.model}")
    if args.model == "quantum":
        print(f"Quantum    : encoding={args.quantum_encoding}, ansatz={args.quantum_ansatz}, "
              f"qubits={args.n_qubits}, layers={args.n_q_layers}")
    print(f"Device     : {device}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # ------------------------------------------
    # Data
    # ------------------------------------------
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        encoder=args.encoder,
        batch_size=args.batch_size,
        max_len=args.max_len,
        data_path=args.data_path,
        max_samples=args.max_samples,
        seed=args.seed
    )

    if vocab_size is not None:
        args.vocab_size = vocab_size

    # ------------------------------------------
    # Model
    # ------------------------------------------
    model = SentimentModel(
        vocab_size=args.vocab_size,
        encoder_type=args.encoder,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=3,
        model_type=args.model,
        n_qubits=args.n_qubits,
        n_q_layers=args.n_q_layers,
        encoding=args.quantum_encoding,
        ansatz=args.quantum_ansatz
    )

    model.to(device)

    # ------------------------------------------
    # Optimizer (small LR for pretrained transformers)
    # ------------------------------------------
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    pretrained = args.encoder in ("bert", "roberta")

    if pretrained and args.freeze_encoder:
        for p in encoder_params:
            p.requires_grad = False
        param_groups = [{"params": other_params, "lr": args.lr}]
        print("Encoder frozen.")
    else:
        encoder_lr = args.encoder_lr if pretrained else args.lr
        param_groups = [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": other_params, "lr": args.lr},
        ]

    optimizer = torch.optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------
    # Training
    # ------------------------------------------
    def save_checkpoint(epoch, average_loss, val_metrics, test_metrics=None):

        # accuracy/precision/recall/f1 = test metrics if available,
        # otherwise validation metrics.
        reported = test_metrics if test_metrics is not None else val_metrics

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": None if args.no_weights else model.state_dict(),
            "loss": average_loss,
            "accuracy": reported["accuracy"],
            "precision": reported["precision"],
            "recall": reported["recall"],
            "f1": reported["f1"],
            "metrics_split": "test" if test_metrics is not None else "val",
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "experiment_name": args.experiment_name,
            "encoder": args.encoder,
            "config": vars(args),
        }

        if args.save_optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, last_checkpoint_path)

    average_loss, val_metrics = float("nan"), None
    history = []
    history_path = os.path.join(checkpoint_dir, "history.json")

    for epoch in range(1, args.epochs + 1):

        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for step, (input_ids, attention_mask, y_batch) in enumerate(train_loader, start=1):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

            if step % args.log_every == 0:
                print(
                    f"  epoch {epoch}/{args.epochs}  step {step}/{len(train_loader)}  "
                    f"loss {running_loss / step:.4f}",
                    flush=True
                )

        average_loss = running_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print(f"Train loss : {average_loss:.4f}")
        print(f"Val loss   : {val_metrics['loss']:.4f}")
        print(f"Val acc    : {val_metrics['accuracy']:.4f}")
        print(f"Val prec   : {val_metrics['precision']:.4f}")
        print(f"Val recall : {val_metrics['recall']:.4f}")
        print(f"Val F1     : {val_metrics['f1']:.4f}\n", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": average_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "epoch_time_sec": time.time() - epoch_start,
        })
        with open(history_path, "w") as f:
            json.dump({"history": history, "test": None,
                       "config": vars(args)}, f, indent=2)

        # saved every epoch, so a crash never loses everything
        save_checkpoint(epoch, average_loss, val_metrics)

    # ------------------------------------------
    # Final test evaluation
    # ------------------------------------------
    test_metrics, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, criterion, device, return_raw=True
    )

    np.savez(
        os.path.join(checkpoint_dir, "test_predictions.npz"),
        labels=np.array(test_labels),
        preds=np.array(test_preds),
        probs=np.array(test_probs),
    )
    with open(history_path, "w") as f:
        json.dump({"history": history, "test": test_metrics,
                   "config": vars(args)}, f, indent=2)

    print("Test results")
    print(f"Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1 Score  : {test_metrics['f1']:.4f}")

    save_checkpoint(args.epochs, average_loss, val_metrics, test_metrics)

    print(f"\nCheckpoint saved to: {last_checkpoint_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
