"""
Run classical and/or quantum experiments and print their metrics.

Experiment names:
    classical -> <encoder>_classical              e.g. bert_classical
    quantum   -> <encoder>_<encoding>_<ansatz>    e.g. bert_angle_strong

Examples
--------
Everything for BERT (1 classical + 4 quantum runs), 5 epochs:
    python extract_metrices.py --run --encoders bert --epochs 5

Only the classical baseline:
    python extract_metrices.py --run --models classical --encoders bert --epochs 5

Only quantum:
    python extract_metrices.py --run --models quantum --encoders bert --epochs 5

Quick smoke test:
    python extract_metrices.py --run --encoders bert --max_samples 2000 --epochs 1

See the commands without running them:
    python extract_metrices.py --run --dry_run

Only show finished results:
    python extract_metrices.py --encoders bert
"""

import argparse
import csv
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_ENCODERS = ["bert", "roberta", "bigru"]
ALL_ENCODINGS = ["angle", "amplitude"]
ALL_ANSATZES = ["strong", "basic"]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run classical/quantum experiments and extract their metrics"
    )

    parser.add_argument("--run", action="store_true",
                        help="Train the experiments before printing metrics.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print the training commands.")
    parser.add_argument("--models", nargs="+", default=["classical", "quantum"],
                        choices=["classical", "quantum"])
    parser.add_argument("--encoders", "--encoder", nargs="+", default=["bert"],
                        choices=ALL_ENCODERS + ["all"])
    parser.add_argument("--encodings", nargs="+", default=ALL_ENCODINGS,
                        choices=ALL_ENCODINGS)
    parser.add_argument("--ansatzes", nargs="+", default=ALL_ANSATZES,
                        choices=ALL_ANSATZES)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Do not retrain experiments that already have results.")

    # forwarded to train2.py (identical for classical and quantum -> fair comparison)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--n_q_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data_path", type=str, default="Twitter_Data.csv")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--no_weights", action="store_true",
                        help="Do not save model weights (saves disk space).")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Where checkpoints/ and results_summary.csv are written "
                             "(default: current folder; use /kaggle/working on Kaggle).")

    args = parser.parse_args()

    if "all" in args.encoders:
        args.encoders = ALL_ENCODERS

    return args


# An experiment is (encoder, encoding, ansatz); encoding/ansatz are None for classical.

def experiment_name(exp):
    enc, encoding, ansatz = exp
    return f"{enc}_classical" if encoding is None else f"{enc}_{encoding}_{ansatz}"


def experiment_list(args):
    experiments = []
    for enc in args.encoders:
        if "classical" in args.models:
            experiments.append((enc, None, None))
        if "quantum" in args.models:
            for encoding in args.encodings:
                for ansatz in args.ansatzes:
                    experiments.append((enc, encoding, ansatz))
    return experiments


def exp_dir(args, name):
    return os.path.join(args.output_dir, "checkpoints", name)


def has_results(args, name):
    return os.path.isfile(os.path.join(exp_dir(args, name), "history.json")) and \
        os.path.isfile(os.path.join(exp_dir(args, name), "test_predictions.npz"))


# ==========================================
# Running
# ==========================================

def build_command(args, exp):

    enc, encoding, ansatz = exp
    name = experiment_name(exp)

    command = [
        sys.executable, os.path.join(SCRIPT_DIR, "train2.py"),
        "--encoder", enc,
        "--model", "classical" if encoding is None else "quantum",
        "--experiment_name", name,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_samples", str(args.max_samples),
        "--max_len", str(args.max_len),
        "--lr", str(args.lr),
        "--encoder_lr", str(args.encoder_lr),
        "--seed", str(args.seed),
        "--device", args.device,
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
    ]

    if encoding is not None:
        command += [
            "--quantum_encoding", encoding,
            "--quantum_ansatz", ansatz,
            "--n_qubits", str(args.n_qubits),
            "--n_q_layers", str(args.n_q_layers),
        ]

    if args.freeze_encoder:
        command.append("--freeze_encoder")

    if args.no_weights:
        command.append("--no_weights")

    return command


def run_experiments(args):

    failed = []
    experiments = experiment_list(args)

    for number, exp in enumerate(experiments, start=1):

        name = experiment_name(exp)
        command = build_command(args, exp)

        if args.dry_run:
            print(f"[{number}/{len(experiments)}] " + " ".join(command))
            continue

        if args.skip_existing and has_results(args, name):
            print(f"[{number}/{len(experiments)}] {name}: results exist, skipping")
            continue

        print("\n" + "=" * 70)
        print(f"[{number}/{len(experiments)}] Running {name}")
        print("=" * 70, flush=True)

        result = subprocess.run(command)

        if result.returncode != 0:
            print(f"!! {name} failed with exit code {result.returncode}")
            failed.append(name)

    return failed


# ==========================================
# Extracting
# ==========================================

def read_row(args, name):
    """Prefer the small history.json; fall back to the big .pth checkpoint."""

    folder = exp_dir(args, name)
    hist_path = os.path.join(folder, "history.json")

    if os.path.isfile(hist_path):
        with open(hist_path) as f:
            data = json.load(f)
        if data.get("test") and data.get("history"):
            last, test = data["history"][-1], data["test"]
            return (name, last["epoch"], last["train_loss"], test["accuracy"],
                    test["precision"], test["recall"], test["f1"], "test")

    ckpt_path = os.path.join(folder, "last_checkpoint.pth")
    if os.path.isfile(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return (name, ckpt["epoch"], ckpt["loss"], ckpt["accuracy"],
                ckpt["precision"], ckpt["recall"], ckpt["f1"],
                ckpt.get("metrics_split", "val"))

    return None


def print_metrics(args):

    header = (
        f"{'Experiment':<28}{'Epoch':>6}{'TrainLoss':>11}{'Accuracy':>10}"
        f"{'Precision':>11}{'Recall':>9}{'F1':>9}{'Set':>6}"
    )

    print("\n" + header)
    print("-" * len(header))

    rows = []

    for exp in experiment_list(args):

        name = experiment_name(exp)

        try:
            row = read_row(args, name)
        except Exception as error:
            print(f"{name:<28}  (could not read results: {error})")
            continue

        if row is None:
            print(f"{name:<28}  (no results yet)")
            continue

        rows.append(row)

        print(
            f"{row[0]:<28}{row[1]:>6}{row[2]:>11.4f}{row[3]:>10.4f}"
            f"{row[4]:>11.4f}{row[5]:>9.4f}{row[6]:>9.4f}{row[7]:>6}"
        )

    if rows:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "results_summary.csv")

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "epoch", "train_loss", "accuracy",
                             "precision", "recall", "f1", "metrics_split"])
            writer.writerows(rows)

        print(f"\nSaved table to {out_path}")


def main():

    args = parse_args()

    failed = []

    if args.run:
        failed = run_experiments(args)

    if not args.dry_run:
        print_metrics(args)

    if failed:
        print("\nFailed experiments: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
