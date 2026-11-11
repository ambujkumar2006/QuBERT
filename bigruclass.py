import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, x):

        x = self.embedding(x)

        return x


class BiGRULayer(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim
    ):
        super().__init__()

        self.bigru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):

        output, hidden = self.bigru(x)

        return output