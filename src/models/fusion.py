"""Feature-level fusion of the character embedding with the tabular matrix.

Feature-level rather than score-level: averaging two probabilities cannot learn
the interaction between "this name looks algorithmic" and "this certificate is
a 90-day free cert issued yesterday", which is the interaction that matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def embed_domains(model, encoder, domains, device="cuda", batch_size: int = 2048):
    model.eval().to(device)
    out = []
    for i in range(0, len(domains), batch_size):
        chunk = encoder.encode_batch(domains[i:i + batch_size])
        tensor = torch.from_numpy(chunk).to(device)
        out.append(model.embed(tensor).cpu().numpy())
    return np.vstack(out)


def fuse(tabular: pd.DataFrame, embeddings: np.ndarray, domains,
         prefix: str = "emb_") -> pd.DataFrame:
    emb_df = pd.DataFrame(embeddings,
                          columns=[f"{prefix}{i}" for i in range(embeddings.shape[1])])
    emb_df["domain"] = list(domains)
    return tabular.merge(emb_df, on="domain", how="left")
