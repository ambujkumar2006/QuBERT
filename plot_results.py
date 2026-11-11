"""
Paper-ready figures for the quantum (VQC) experiments, optionally compared
with a classical baseline.

Needs, per experiment, the files written by the updated train2.py:
    checkpoints/<name>/history.json
    checkpoints/<name>/test_predictions.npz

Experiment names are expected as  <encoder>_<encoding>_<ansatz>   (e.g. bert_angle_strong)
and  <encoder>_classical  for the baseline (optional).

Usage
-----
    python plot_results.py                       # every finished experiment
    python plot_results.py --experiments bert_angle_strong bert_amplitude_strong bert_classical
    python plot_results.py --out figures

Output: 300-dpi PNG + vector PDF in  figures/<experiment>/  and  figures/comparison/
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIRS = [os.path.join(os.getcwd(), "checkpoints")]   # overridden by --ckpt_dirs
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
CLASSICAL_COLOR = "#7f7f7f"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ==========================================
# Helpers
# ==========================================

def save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{filename}.{ext}"))
    plt.close(fig)
    print(f"  saved {os.path.join(out_dir, filename)}.[png|pdf]")


def parse_name(name):
    """-> (encoder, encoding, ansatz); encoding/ansatz are None for classical."""
    parts = name.split("_")
    if len(parts) == 3 and parts[1] in ("angle", "amplitude") and parts[2] in ("strong", "basic"):
        return parts[0], parts[1], parts[2]
    return parts[0], None, None


def pretty(name, with_encoder=False, newline=False):
    enc, encoding, ansatz = parse_name(name)
    if encoding is None:
        label = "Classical"
    else:
        label = f"{encoding.capitalize()} + {ansatz.capitalize()}"
    if with_encoder:
        label = f"{enc.upper()} {label}"
    if newline:
        label = label.replace(" + ", "\n").replace("Classical", "Classical\n(no VQC)")
    return label


def color_for(name, index):
    return CLASSICAL_COLOR if parse_name(name)[1] is None else PALETTE[index % len(PALETTE)]


def locate(name):
    for root in CKPT_DIRS:
        folder = os.path.join(root, name)
        if os.path.isfile(os.path.join(folder, "history.json")):
            return folder
    return None


def load_experiment(name):
    folder = locate(name) or os.path.join(CKPT_DIRS[0], name)
    hist_path = os.path.join(folder, "history.json")
    pred_path = os.path.join(folder, "test_predictions.npz")

    if not (os.path.isfile(hist_path) and os.path.isfile(pred_path)):
        return None

    with open(hist_path) as f:
        data = json.load(f)

    if data.get("test") is None:
        return None

    preds = np.load(pred_path)
    return {
        "name": name,
        "history": data["history"],
        "test": data["test"],
        "config": data.get("config", {}),
        "labels": preds["labels"],
        "preds": preds["preds"],
        "probs": preds["probs"],
    }


def find_experiments():
    found = set()
    for root in CKPT_DIRS:
        if os.path.isdir(root):
            found.update(d for d in os.listdir(root)
                         if os.path.isfile(os.path.join(root, d, "history.json")))
    return sorted(found)


def check_fairness(exps):
    """Warn if experiments were trained with different hyper-parameters."""
    keys = ["seed", "epochs", "batch_size", "lr", "encoder_lr", "max_len",
            "max_samples", "data_path", "freeze_encoder"]
    print("\nFairness check (training settings must match for a valid comparison):")
    problems = False
    for key in keys:
        values = {e["name"]: e["config"].get(key) for e in exps if e["config"]}
        if len(values) < len(exps):
            print("  ! some experiments have no config in history.json "
                  "(trained with the old train2.py) -> cannot verify")
            return
        if len(set(map(str, values.values()))) > 1:
            problems = True
            print(f"  ! '{key}' differs: {values}")
    if not problems:
        print("  OK - all experiments share the same settings")


def circuit_params(exp):
    """Number of trainable parameters inside the variational circuit."""
    enc, encoding, ansatz = parse_name(exp["name"])
    if encoding is None:
        return 0
    cfg = exp["config"]
    layers, qubits = cfg.get("n_q_layers", 2), cfg.get("n_qubits", 2)
    return layers * qubits * (3 if ansatz == "strong" else 1)


def collapsed(exp):
    """True if the model predicts (almost) only one class."""
    counts = np.bincount(exp["preds"], minlength=3)
    return counts.max() / counts.sum() > 0.95


# ==========================================
# Per-experiment figures
# ==========================================

def plot_loss_curves(exp, out_dir):
    h = exp["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(ep, [r["train_loss"] for r in h], "o-", label="Train loss")
    ax.plot(ep, [r["val_loss"] for r in h], "s--", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Loss: {pretty(exp['name'], True)}")
    ax.set_xticks(ep)
    ax.legend()
    save(fig, out_dir, "loss_curves")


def plot_val_metrics(exp, out_dir):
    h = exp["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for key, label, marker in [("val_accuracy", "Accuracy", "o"),
                               ("val_precision", "Precision (macro)", "s"),
                               ("val_recall", "Recall (macro)", "^"),
                               ("val_f1", "F1 (macro)", "D")]:
        ax.plot(ep, [r[key] for r in h], marker + "-", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"Validation metrics: {pretty(exp['name'], True)}")
    ax.set_xticks(ep)
    ax.legend(loc="best")
    save(fig, out_dir, "validation_metrics")


def draw_confusion(ax, exp, normalise, show_cbar_fig=None):
    cm = confusion_matrix(exp["labels"], exp["preds"], labels=[0, 1, 2])
    matrix = cm / cm.sum(axis=1, keepdims=True) if normalise else cm
    fmt = ".2f" if normalise else "d"
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    threshold = matrix.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    return im


def plot_confusion(exp, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, norm, title in [(axes[0], False, "Counts"), (axes[1], True, "Row-normalised")]:
        im = draw_confusion(ax, exp, norm)
        ax.set_title(f"{title}: {pretty(exp['name'], True)}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, out_dir, "confusion_matrix")


def plot_per_class(exp, out_dir):
    p, r, f, support = precision_recall_fscore_support(
        exp["labels"], exp["preds"], labels=[0, 1, 2], zero_division=0)
    x = np.arange(3)
    w = 0.25
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for k, (vals, label) in enumerate([(p, "Precision"), (r, "Recall"), (f, "F1")]):
        bars = ax.bar(x + (k - 1) * w, vals, w, label=label, color=PALETTE[k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(CLASS_NAMES, support)])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class test performance: {pretty(exp['name'], True)}", fontsize=10)
    ax.legend(loc="lower right")
    save(fig, out_dir, "per_class_metrics")


def plot_roc(exp, out_dir):
    y = label_binarize(exp["labels"], classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y[:, i], exp["probs"][:, i])
        ax.plot(fpr, tpr, color=PALETTE[i], label=f"{cls} (AUC = {auc(fpr, tpr):.3f})")
    fpr, tpr, _ = roc_curve(y.ravel(), exp["probs"].ravel())
    ax.plot(fpr, tpr, "k--", label=f"Micro-avg (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC (one-vs-rest): {pretty(exp['name'], True)}", fontsize=10)
    ax.legend(loc="lower right")
    save(fig, out_dir, "roc_curves")


def plot_pr(exp, out_dir):
    y = label_binarize(exp["labels"], classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for i, cls in enumerate(CLASS_NAMES):
        prec, rec, _ = precision_recall_curve(y[:, i], exp["probs"][:, i])
        ap = average_precision_score(y[:, i], exp["probs"][:, i])
        ax.plot(rec, prec, color=PALETTE[i], label=f"{cls} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision-recall: {pretty(exp['name'], True)}", fontsize=10)
    ax.legend(loc="lower left")
    save(fig, out_dir, "precision_recall_curves")


def plot_confidence(exp, out_dir):
    conf = exp["probs"].max(axis=1)
    ok = exp["preds"] == exp["labels"]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bins = np.linspace(1 / 3, 1.0, 29)
    ax.hist(conf[ok], bins=bins, alpha=0.7, label="Correct", color=PALETTE[2])
    ax.hist(conf[~ok], bins=bins, alpha=0.7, label="Misclassified", color=PALETTE[1])
    ax.set_xlabel("Predicted-class probability")
    ax.set_ylabel("Test samples")
    ax.set_title(f"Prediction confidence: {pretty(exp['name'], True)}", fontsize=10)
    ax.legend()
    save(fig, out_dir, "confidence_histogram")


# ==========================================
# Comparison figures
# ==========================================

def plot_comparison(exps, out_dir):
    if len(exps) < 2:
        return

    names = [e["name"] for e in exps]
    colors = [color_for(n, i) for i, n in enumerate(names)]
    labels = [pretty(n, with_encoder=len({parse_name(x)[0] for x in names}) > 1, newline=True)
              for n in names]

    # --- 1. grouped bars: test metrics -------------------------------
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"),
               ("recall", "Recall"), ("f1", "F1")]
    x = np.arange(len(metrics))
    w = 0.8 / len(exps)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(exps) + 3), 4.2))
    for k, e in enumerate(exps):
        vals = [e["test"][m] for m, _ in metrics]
        bars = ax.bar(x + (k - (len(exps) - 1) / 2) * w, vals, w,
                      label=labels[k].replace("\n", " "), color=colors[k],
                      hatch="//" if collapsed(e) else None)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Test-set comparison (hatched = collapsed to one class)")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    save(fig, out_dir, "comparison_test_metrics")

    # --- 2. validation F1 / loss curves overlaid ------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, e in enumerate(exps):
        ep = [r["epoch"] for r in e["history"]]
        lab = labels[k].replace("\n", " ")
        axes[0].plot(ep, [r["train_loss"] for r in e["history"]], "o-", color=colors[k], label=lab)
        axes[1].plot(ep, [r["val_loss"] for r in e["history"]], "s-", color=colors[k], label=lab)
        axes[2].plot(ep, [r["val_f1"] for r in e["history"]], "D-", color=colors[k], label=lab)
    for ax, t, yl in zip(axes, ["Training loss", "Validation loss", "Validation F1 (macro)"],
                         ["Loss", "Loss", "F1"]):
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(yl)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, "comparison_curves")

    # --- 3. confusion matrices side by side -------------------------
    n = len(exps)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, e in enumerate(exps):
        ax = axes[k // cols][k % cols]
        ax.axis("on")
        draw_confusion(ax, e, normalise=True)
        ax.set_title(pretty(e["name"], with_encoder=True), fontsize=10)
    fig.tight_layout()
    save(fig, out_dir, "comparison_confusion_matrices")

    # --- 4. per-class F1 grouped ---------------------------------------
    x = np.arange(3)
    w = 0.8 / len(exps)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, e in enumerate(exps):
        _, _, f1, _ = precision_recall_fscore_support(
            e["labels"], e["preds"], labels=[0, 1, 2], zero_division=0)
        ax.bar(x + (k - (len(exps) - 1) / 2) * w, f1, w, color=colors[k],
               label=labels[k].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 (test set)")
    ax.set_title("Per-class F1 across models")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, out_dir, "comparison_per_class_f1")

    # --- 5. ROC overlay (micro-average) -------------------------------
    fig, ax = plt.subplots(figsize=(5, 4.4))
    for k, e in enumerate(exps):
        y = label_binarize(e["labels"], classes=[0, 1, 2])
        fpr, tpr, _ = roc_curve(y.ravel(), e["probs"].ravel())
        ax.plot(fpr, tpr, color=colors[k],
                label=f"{labels[k].replace(chr(10), ' ')} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison (micro-average)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, out_dir, "comparison_roc")

    # --- 6. epoch time ---------------------------------------------------
    if all("epoch_time_sec" in e["history"][0] for e in exps):
        times = [np.mean([r["epoch_time_sec"] for r in e["history"]]) / 60 for e in exps]
        fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(exps) + 2), 3.6))
        bars = ax.bar(range(len(exps)), times, color=colors)
        for b, v in zip(bars, times):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Minutes per epoch (train + val)")
        ax.set_title("Computational cost")
        save(fig, out_dir, "comparison_epoch_time")


def plot_heatmaps(exps, out_dir):
    """Encoding x ansatz grid of test F1 and accuracy (one row per encoder)."""
    encoders = sorted({parse_name(e["name"])[0] for e in exps
                       if parse_name(e["name"])[1] is not None})
    encodings, ansatzes = ["angle", "amplitude"], ["strong", "basic"]
    lookup = {parse_name(e["name"]): e for e in exps}

    for enc in encoders:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        for ax, (key, title) in zip(axes, [("f1", "Test F1 (macro)"), ("accuracy", "Test accuracy")]):
            grid = np.full((2, 2), np.nan)
            for i, encoding in enumerate(encodings):
                for j, ansatz in enumerate(ansatzes):
                    e = lookup.get((enc, encoding, ansatz))
                    if e:
                        grid[i, j] = e["test"][key]
            im = ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=1)
            ax.set_xticks(range(2))
            ax.set_yticks(range(2))
            ax.set_xticklabels([a.capitalize() for a in ansatzes])
            ax.set_yticklabels([c.capitalize() for c in encodings])
            ax.set_xlabel("Ansatz")
            ax.set_ylabel("Data encoding")
            ax.set_title(f"{title} ({enc.upper()})", fontsize=10)
            ax.grid(False)
            for i in range(2):
                for j in range(2):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                                color="white" if grid[i, j] > 0.6 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        save(fig, out_dir, f"heatmap_{enc}_encoding_vs_ansatz")


def print_latex_table(exps):
    best = {k: max(e["test"][k] for e in exps if not collapsed(e))
            for k in ("accuracy", "precision", "recall", "f1")} if any(not collapsed(e) for e in exps) else {}

    def cell(e, k):
        v = f"{e['test'][k]:.4f}"
        return f"\\textbf{{{v}}}" if best.get(k) == e["test"][k] else v

    print("\nLaTeX table (paste into your paper):\n")
    print("\\begin{tabular}{llccccc}")
    print("\\hline")
    print("Encoder & Quantum layer & Circuit params & Acc. & Prec. & Rec. & F1 \\\\")
    print("\\hline")
    for e in exps:
        enc, encoding, ansatz = parse_name(e["name"])
        quantum = "None (classical)" if encoding is None else f"{encoding}/{ansatz}"
        print(f"{enc.upper()} & {quantum} & {circuit_params(e)} & "
              f"{cell(e, 'accuracy')} & {cell(e, 'precision')} & "
              f"{cell(e, 'recall')} & {cell(e, 'f1')} \\\\")
    print("\\hline")
    print("\\end{tabular}\n")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--out", type=str, default="figures",
                        help="Output folder (absolute path allowed, e.g. /kaggle/working/figures).")
    parser.add_argument("--ckpt_dirs", nargs="+", default=None,
                        help="One or more 'checkpoints' folders to read results from "
                             "(e.g. the classical and the quantum run).")
    args = parser.parse_args()

    if args.ckpt_dirs:
        CKPT_DIRS[:] = [os.path.abspath(d) for d in args.ckpt_dirs]
    print("Reading results from:", *CKPT_DIRS, sep="\n  ")

    names = args.experiments or find_experiments()
    if not names:
        raise SystemExit("No finished experiments found. Run train2.py first.")

    out_root = os.path.join(os.getcwd(), args.out)
    exps = []

    for name in names:
        exp = load_experiment(name)
        if exp is None:
            print(f"Skipping {name}: needs history.json (with test results) and "
                  f"test_predictions.npz -> re-run it with the updated train2.py.")
            continue

        print(f"\n{name}" + ("   [WARNING: predicts a single class]" if collapsed(exp) else ""))
        out_dir = os.path.join(out_root, name)
        plot_loss_curves(exp, out_dir)
        plot_val_metrics(exp, out_dir)
        plot_confusion(exp, out_dir)
        plot_per_class(exp, out_dir)
        plot_roc(exp, out_dir)
        plot_pr(exp, out_dir)
        plot_confidence(exp, out_dir)
        exps.append(exp)

    if len(exps) > 1:
        print("\ncomparison")
        plot_comparison(exps, os.path.join(out_root, "comparison"))
        plot_heatmaps(exps, os.path.join(out_root, "comparison"))

    if exps:
        check_fairness(exps)
        print_latex_table(exps)


if __name__ == "__main__":
    main()
