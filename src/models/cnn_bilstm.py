"""Character-level CNN-BiLSTM for the domain string.

Rationale: hand-crafted lexical statistics discard character ordering, which is
precisely the signal separating algorithmically generated names from
human-chosen ones. The penultimate layer is exported as the fusion embedding.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789-._"


class CharEncoder:
    def __init__(self, charset: str = DEFAULT_CHARSET, max_length: int = 63):
        self.charset = charset
        self.max_length = max_length
        self.stoi = {c: i + 1 for i, c in enumerate(charset)}   # 0 = PAD/OOV

    @property
    def vocab_size(self) -> int:
        return len(self.charset) + 1

    def encode(self, domain: str) -> np.ndarray:
        s = domain.lower().strip(".")[: self.max_length]
        idx = [self.stoi.get(c, 0) for c in s]
        return np.array(idx + [0] * (self.max_length - len(idx)), dtype=np.int64)

    def encode_batch(self, domains) -> np.ndarray:
        return np.stack([self.encode(d) for d in domains])


class FocalLoss(nn.Module):
    """Used instead of SMOTE: interpolating between character embeddings would
    synthesise domains that cannot exist."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, conv_filters: int = 128,
                 kernel_sizes=(2, 3, 4), lstm_hidden: int = 128, dropout: float = 0.4,
                 penultimate_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, conv_filters, k, padding=k // 2) for k in kernel_sizes
        ])
        self.lstm = nn.LSTM(conv_filters * len(kernel_sizes), lstm_hidden,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.penultimate = nn.Linear(lstm_hidden * 2, penultimate_dim)
        self.head = nn.Linear(penultimate_dim, 1)

    def forward(self, x, return_embedding: bool = False):
        e = self.embedding(x).transpose(1, 2)
        conv_out = torch.cat([F.relu(c(e))[:, :, : x.size(1)] for c in self.convs], dim=1)
        seq, _ = self.lstm(conv_out.transpose(1, 2))
        pooled = torch.max(seq, dim=1).values
        emb = F.relu(self.penultimate(self.dropout(pooled)))
        if return_embedding:
            return emb
        return self.head(emb).squeeze(-1)

    @torch.no_grad()
    def embed(self, x) -> torch.Tensor:
        self.eval()
        return self.forward(x, return_embedding=True)
