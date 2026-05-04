"""
EPRVFL: Embedding-based Parallel Random Vector Functional Link Network
for Real-Time Fake News Detection

Reference:
    R. K. Gurjwar, A. Kumar, U. P. Rao,
    "EPRVFL: A fast and scalable model for real-time fake news detection,"
    Pattern Recognition Letters, vol. 196, pp. 267-273, 2025.
    https://doi.org/10.1016/j.patrec.2025.02.009

Architecture:
    Input embeddings (BERT/transformer) → RVFL hidden layer (tanh activation)
    → Concatenate [input, hidden] → Closed-form least-squares output weights
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class EPRVFL:
    """
    Embedding-based Parallel Random Vector Functional Link Network.

    Combines pre-trained transformer embeddings with a single RVFL hidden
    layer. Output weights are solved analytically via the Moore-Penrose
    pseudoinverse — no backpropagation required, making inference extremely fast.

    Parameters
    ----------
    num_hidden_nodes : int
        Number of random hidden nodes. Default: 23 (as per paper).
    C : float
        Regularisation parameter (inverse of ridge penalty).
        Default: 0.03125 (as per paper).
    random_state : int or None
        Seed for reproducibility of random weight initialisation.
    """

    def __init__(self, num_hidden_nodes: int = 23, C: float = 0.03125,
                 random_state: int = 42):
        self.num_hidden_nodes = num_hidden_nodes
        self.C = C
        self.random_state = random_state
        self.W = None       # Random input weights  [input_dim × hidden_nodes]
        self.beta = None    # Output weights (solved analytically)

    def _init_random_weights(self, input_dim: int) -> None:
        rng = np.random.RandomState(self.random_state)
        self.W = rng.uniform(-1, 1, size=(input_dim, self.num_hidden_nodes))

    def _build_H(self, X: np.ndarray) -> np.ndarray:
        """
        Build the enhanced feature matrix H = [X | tanh(X @ W + bias)].
        Bias is re-sampled per call to introduce diversity across epochs
        (following the original paper's design).
        """
        rng = np.random.RandomState(self.random_state)
        bias = rng.uniform(-1, 1, size=(X.shape[0], 1))
        H_hidden = np.tanh(X @ self.W + bias)
        return np.hstack((X, H_hidden))   # shape: [n_samples, input_dim + hidden_nodes]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EPRVFL":
        """
        Fit the model using closed-form least-squares solution.

        Parameters
        ----------
        X : np.ndarray, shape [n_samples, embedding_dim]
            Pre-computed transformer embeddings.
        y : np.ndarray, shape [n_samples]
            Binary labels (0 = real, 1 = fake).

        Returns
        -------
        self
        """
        if self.W is None:
            self._init_random_weights(X.shape[1])

        H = self._build_H(X)
        # Closed-form: beta = (H^T H + (1/C) I)^{-1} H^T y
        reg = (1.0 / self.C) * np.eye(H.shape[1])
        self.beta = np.linalg.pinv(H.T @ H + reg) @ H.T @ y
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return continuous prediction scores."""
        H = self._build_H(X)
        return H @ self.beta

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.

        Parameters
        ----------
        X : np.ndarray
            Pre-computed transformer embeddings.
        threshold : float
            Decision boundary. Default: 0.5.

        Returns
        -------
        np.ndarray of int predictions.
        """
        return (self.predict_raw(X) >= threshold).astype(int)

    def fit_epochs(self, X_train: np.ndarray, y_train: np.ndarray,
                   epochs: int = 10) -> "EPRVFL":
        """
        Train for multiple epochs (refits output weights each epoch,
        matching the experimental setup in the paper).
        """
        if self.W is None:
            self._init_random_weights(X_train.shape[1])
        for _ in range(epochs):
            self.fit(X_train, y_train)
        return self


def evaluate(model: EPRVFL, X_test: np.ndarray,
             y_test: np.ndarray) -> dict:
    """
    Compute accuracy, precision, recall, and F1-score.

    Parameters
    ----------
    model : EPRVFL
    X_test : np.ndarray
    y_test : np.ndarray

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted",
                                     zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted",
                                  zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted",
                              zero_division=0),
    }
