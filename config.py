import argparse
import os


def get_args(argv=None):

    parser = argparse.ArgumentParser(
        description="Classical / Quantum Experiment Configuration"
    )

    # =============================
    # Encoder
    # =============================
    parser.add_argument(
        "--encoder",
        type=str,
        default="bert",
        choices=["bigru", "bert", "roberta"]
    )

    # classical = Encoder -> Attention -> Classifier
    # quantum   = Encoder -> Attention -> VQC -> Classifier
    parser.add_argument(
        "--model",
        type=str,
        default="quantum",
        choices=["classical", "quantum"]
    )

    # =============================
    # Data
    # =============================
    parser.add_argument(
        "--data_path",
        type=str,
        default="Twitter_Data.csv"
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=50
    )

    # 0 = use the whole dataset. Use e.g. 2000 for a quick smoke test.
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0
    )

    # Only used by BiGRU (it is overwritten by the real
    # vocabulary size computed in dataloader.py)
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000
    )

    # =============================
    # Classical model parameters
    # =============================
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128
    )

    # =============================
    # Quantum model parameters
    # =============================
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=2
    )

    parser.add_argument(
        "--quantum_encoding",
        type=str,
        default="angle",
        choices=["angle", "amplitude"]
    )

    parser.add_argument(
        "--quantum_ansatz",
        type=str,
        default="strong",
        choices=["strong", "basic"]
    )

    parser.add_argument(
        "--n_q_layers",
        type=int,
        default=2
    )

    # =============================
    # Training parameters
    # =============================
    parser.add_argument(
        "--epochs",
        type=int,
        default=1
    )

    # LR for everything except a pretrained BERT / RoBERTa encoder
    # (attention, quantum layer, classifier, and BiGRU).
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2
    )

    # LR for the pretrained BERT / RoBERTa encoder.
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=2e-5
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze BERT/RoBERTa weights (much faster)."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"]
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=100
    )

    # =============================
    # Experiment parameters
    # =============================
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="baseline"
    )

    # Results go to <output_dir>/checkpoints/<experiment_name>/ .
    # Default "." = current working directory (on Kaggle: /kaggle/working).
    parser.add_argument(
        "--output_dir",
        type=str,
        default="."
    )

    parser.add_argument(
        "--no_weights",
        action="store_true",
        help="Do not store model weights (saves ~400 MB per BERT run); metrics are still saved."
    )

    parser.add_argument(
        "--save_optimizer",
        action="store_true",
        help="Also store optimizer state in the checkpoint (large files)."
    )

    return parser.parse_args(argv)
