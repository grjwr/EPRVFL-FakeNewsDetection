"""
Embedding utilities for EPRVFL.

Generates BERT sentence embeddings from raw text — the same preprocessing
used in the Pattern Recognition Letters paper (Gurjwar et al., 2025).

Supported datasets (all publicly available):
    - PolitiFact   https://github.com/KaiDMML/FakeNewsNet
    - BuzzFeed-Webis  https://zenodo.org/record/1239675
    - LIAR-2       https://huggingface.co/datasets/chengxuphd/liar2
    - GossipCop    https://github.com/KaiDMML/FakeNewsNet
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def load_dataset(csv_path: str | Path,
                 text_col: str = "text",
                 label_col: str = "label") -> tuple[list[str], np.ndarray]:
    """
    Load a CSV dataset with text and binary labels.

    Parameters
    ----------
    csv_path : str or Path
        Path to a CSV file with at least `text_col` and `label_col` columns.
    text_col : str
        Column name containing the news text.
    label_col : str
        Column name containing the binary label (0 = real, 1 = fake).

    Returns
    -------
    texts  : list of str
    labels : np.ndarray of int
    """
    df = pd.read_csv(csv_path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Columns {missing} not found. Available: {list(df.columns)}")

    df = df.dropna(subset=[text_col, label_col])
    texts  = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).values
    print(f"Loaded {len(texts):,} samples  |  "
          f"fake={labels.sum():,}  real={(labels==0).sum():,}")
    return texts, labels


def generate_bert_embeddings(texts: list[str],
                             model_name: str = "bert-base-uncased",
                             batch_size: int = 32,
                             device: str = "cpu") -> np.ndarray:
    """
    Generate mean-pooled BERT embeddings for a list of texts.

    Parameters
    ----------
    texts : list of str
    model_name : str
        Any HuggingFace transformer model name.
        Default: 'bert-base-uncased' (matches the paper).
    batch_size : int
        Number of texts per forward pass.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    np.ndarray, shape [n_samples, hidden_size]
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required.\n"
            "Install with: pip install transformers torch"
        ) from e

    print(f"Loading tokeniser and model: {model_name} …")
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        encoded = tokeniser(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        # Mean-pool over the token dimension (excluding padding)
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        token_embeddings = output.last_hidden_state
        sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_embeddings = (sum_embeddings / count).cpu().numpy()
        all_embeddings.append(mean_embeddings)

        if (i // batch_size) % 10 == 0:
            print(f"  Embedded {min(i + batch_size, len(texts)):,} / "
                  f"{len(texts):,} texts")

    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray,
                    labels: np.ndarray,
                    save_path: str | Path) -> None:
    """Save embeddings and labels as a compressed numpy archive."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path,
                        embeddings=embeddings,
                        labels=labels)
    print(f"Saved embeddings → {save_path}")


def load_embeddings(load_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from a compressed numpy archive."""
    data = np.load(load_path)
    return data["embeddings"], data["labels"]
