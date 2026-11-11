import math

import torch
import torch.nn as nn
try:
    import pennylane as qml
except ImportError:          # classical runs do not need PennyLane
    qml = None

from bigruclass import (
    EmbeddingLayer,
    BiGRULayer
)
from bertclass import BERTLayer
from robertaclass import RoBERTaLayer


# ==========================================
# Attention pooling
# ==========================================

class AttentionLayer(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x, attention_mask=None):

        scores = self.attention(x)                       # [B, T, 1]

        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask.unsqueeze(-1) == 0,
                -1e9
            )

        weights = torch.softmax(scores, dim=1)           # [B, T, 1]
        context = torch.sum(weights * x, dim=1)          # [B, D]

        return context, weights


# ==========================================
# Quantum circuit pieces
# ==========================================

def apply_encoding(inputs, encoding, n_qubits):

    if encoding == "angle":
        qml.AngleEmbedding(
            inputs,
            wires=range(n_qubits)
        )

    elif encoding == "amplitude":
        qml.AmplitudeEmbedding(
            inputs,
            wires=range(n_qubits),
            normalize=True,
            pad_with=0.0
        )

    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def apply_ansatz(weights, ansatz, n_qubits):

    if ansatz == "strong":
        qml.StronglyEntanglingLayers(
            weights,
            wires=range(n_qubits)
        )

    elif ansatz == "basic":
        qml.BasicEntanglerLayers(
            weights,
            wires=range(n_qubits)
        )

    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")


def get_weight_shapes(ansatz, n_layers, n_qubits):

    if ansatz == "strong":
        return {"weights": (n_layers, n_qubits, 3)}

    elif ansatz == "basic":
        return {"weights": (n_layers, n_qubits)}

    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")


def create_quantum_circuit(n_qubits, encoding, ansatz):

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):

        apply_encoding(inputs, encoding, n_qubits)
        apply_ansatz(weights, ansatz, n_qubits)

        return [
            qml.expval(qml.PauliZ(i))
            for i in range(n_qubits)
        ]

    return circuit


class QuantumLayer(nn.Module):

    def __init__(
            self,
            input_dim,
            n_qubits,
            n_q_layers,
            encoding,
            ansatz
    ):
        super().__init__()

        if qml is None:
            raise ImportError("PennyLane is required for --model quantum (pip install pennylane).")

        self.n_qubits = n_qubits
        self.encoding = encoding
        self.ansatz = ansatz

        # Classical -> quantum projection
        if encoding == "angle":
            encoding_dim = n_qubits

        elif encoding == "amplitude":
            encoding_dim = 2 ** n_qubits

        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        self.fc_reduce = nn.Linear(input_dim, encoding_dim)

        circuit = create_quantum_circuit(
            n_qubits=n_qubits,
            encoding=encoding,
            ansatz=ansatz
        )

        weight_shapes = get_weight_shapes(
            ansatz=ansatz,
            n_layers=n_q_layers,
            n_qubits=n_qubits
        )

        self.quantum = qml.qnn.TorchLayer(circuit, weight_shapes)

    def _apply(self, fn, *args, **kwargs):
        # model.to(device) moves only the classical projection.
        # The small quantum circuit always stays on the CPU
        # (the simulator is CPU-based and this avoids GPU/CPU
        # tensor mismatches inside PennyLane).
        self.fc_reduce._apply(fn, *args, **kwargs)
        return self

    def forward(self, x):

        out_device = x.device
        out_dtype = x.dtype

        z = self.fc_reduce(x)

        if self.encoding == "angle":
            # keep rotation angles in a bounded range
            z = torch.tanh(z) * math.pi

        q = self.quantum(z.cpu())
        q = q.reshape(z.shape[0], self.n_qubits)

        return q.to(device=out_device, dtype=out_dtype)


class Classifier(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ==========================================
# Full model (classical or quantum head)
# ==========================================

class SentimentModel(nn.Module):

    def __init__(
            self,
            vocab_size,
            encoder_type,
            embedding_dim,
            hidden_dim,
            num_classes,
            model_type="quantum",
            n_qubits=2,
            n_q_layers=2,
            encoding="angle",
            ansatz="strong"
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.model_type = model_type

        if encoder_type == "bigru":

            self.embedding = EmbeddingLayer(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim
            )

            self.encoder = BiGRULayer(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim
            )

            encoder_dim = hidden_dim * 2

        elif encoder_type == "bert":

            self.encoder = BERTLayer()
            encoder_dim = 768

        elif encoder_type == "roberta":

            self.encoder = RoBERTaLayer()
            encoder_dim = 768

        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.attention = AttentionLayer(input_dim=encoder_dim)

        if model_type == "quantum":

            self.quantum = QuantumLayer(
                input_dim=encoder_dim,
                n_qubits=n_qubits,
                n_q_layers=n_q_layers,
                encoding=encoding,
                ansatz=ansatz
            )
            classifier_in = n_qubits

        elif model_type == "classical":

            self.quantum = None
            classifier_in = encoder_dim

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.classifier = Classifier(
            input_dim=classifier_in,
            num_classes=num_classes
        )

    def forward(self, input_ids, attention_mask=None):

        if self.encoder_type == "bigru":
            x = self.embedding(input_ids)
            x = self.encoder(x)

        else:
            x = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # padded positions are ignored by the attention pooling
        context, _ = self.attention(x, attention_mask)

        if self.quantum is not None:
            context = self.quantum(context)

        return self.classifier(context)
