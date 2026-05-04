"""
Baseline models used for comparison with EPRVFL in:

    R. K. Gurjwar, A. Kumar, U. P. Rao,
    "EPRVFL: A fast and scalable model for real-time fake news detection,"
    Pattern Recognition Letters, vol. 196, pp. 267-273, 2025.

Baselines implemented:
    Deep Learning  : GRU, BiGRU, LSTM, BiLSTM, CNN, FFNN
    Machine Learning: SVM, Logistic Regression, Naïve Bayes, Random Forest

All models receive the same pre-computed BERT embeddings as input,
ensuring a fair comparison with EPRVFL.
"""

import time
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# ── Shared utilities ──────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return accuracy, precision, recall, F1 (weighted)."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted",
                                     zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted",
                                  zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted",
                              zero_division=0),
    }


def _split(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """70 / 15 / 15 train-val-test split."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _timed_eval_keras(model, X_test, y_test) -> tuple:
    """Run inference, threshold at 0.5, return metrics + inference time."""
    t0 = time.time()
    y_prob = model.predict(X_test, verbose=0)
    inf_time = time.time() - t0
    y_pred = (y_prob > 0.5).astype(int).ravel()
    metrics = compute_metrics(y_test, y_pred)
    metrics["inference_time"] = inf_time
    return metrics


# ── Deep Learning baselines ───────────────────────────────────────────────────

def _requires_keras():
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Dense, GRU, LSTM,
                                             Bidirectional, Conv1D, Flatten)
        return Sequential, Dense, GRU, LSTM, Bidirectional, Conv1D, Flatten
    except ImportError as e:
        raise ImportError(
            "TensorFlow/Keras is required for deep learning baselines.\n"
            "Install with: pip install tensorflow"
        ) from e


def run_gru(X: np.ndarray, y: np.ndarray,
            epoch_list: list = None, bidirectional: bool = False) -> dict:
    """
    Train GRU (or BiGRU) baseline and return results per epoch.

    Parameters
    ----------
    X : np.ndarray  — BERT embeddings [n_samples, embedding_dim]
    y : np.ndarray  — binary labels
    epoch_list : list of int — epochs to evaluate at
    bidirectional : bool — use Bidirectional wrapper if True

    Returns
    -------
    dict  keyed by epoch count, each value is a metrics dict
    """
    if epoch_list is None:
        epoch_list = [3, 5, 10, 15, 20, 25, 30]

    Sequential, Dense, GRU, LSTM, Bidirectional, Conv1D, Flatten = \
        _requires_keras()

    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)

    # GRU expects 3-D input: (samples, timesteps, features)
    X_train_3d = X_train[:, np.newaxis, :]
    X_val_3d   = X_val[:,   np.newaxis, :]
    X_test_3d  = X_test[:,  np.newaxis, :]

    results = {}
    for epochs in epoch_list:
        label = "BiGRU" if bidirectional else "GRU"
        print(f"Training {label} — {epochs} epochs …")

        gru_layer = GRU(100, input_shape=(1, X_train.shape[1]))
        if bidirectional:
            gru_layer = Bidirectional(gru_layer)

        model = Sequential([gru_layer, Dense(1, activation="sigmoid")])
        model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=["accuracy"])

        t0 = time.time()
        model.fit(X_train_3d, y_train, epochs=epochs, batch_size=64,
                  validation_data=(X_val_3d, y_val), verbose=0, shuffle=False)
        train_time = time.time() - t0

        metrics = _timed_eval_keras(model, X_test_3d, y_test)
        metrics["training_time"] = train_time
        results[epochs] = metrics

    return results


def run_lstm(X: np.ndarray, y: np.ndarray,
             epoch_list: list = None, bidirectional: bool = False) -> dict:
    """Train LSTM (or BiLSTM) baseline."""
    if epoch_list is None:
        epoch_list = [3, 5, 10, 15, 20, 25, 30]

    Sequential, Dense, GRU, LSTM, Bidirectional, Conv1D, Flatten = \
        _requires_keras()

    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)
    X_train_3d = X_train[:, np.newaxis, :]
    X_val_3d   = X_val[:,   np.newaxis, :]
    X_test_3d  = X_test[:,  np.newaxis, :]

    results = {}
    for epochs in epoch_list:
        label = "BiLSTM" if bidirectional else "LSTM"
        print(f"Training {label} — {epochs} epochs …")

        lstm_layer = LSTM(100, input_shape=(1, X_train.shape[1]))
        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)

        model = Sequential([lstm_layer, Dense(1, activation="sigmoid")])
        model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=["accuracy"])

        t0 = time.time()
        model.fit(X_train_3d, y_train, epochs=epochs, batch_size=64,
                  validation_data=(X_val_3d, y_val), verbose=0, shuffle=False)
        train_time = time.time() - t0

        metrics = _timed_eval_keras(model, X_test_3d, y_test)
        metrics["training_time"] = train_time
        results[epochs] = metrics

    return results


def run_cnn(X: np.ndarray, y: np.ndarray,
            epoch_list: list = None) -> dict:
    """Train 1-D CNN baseline."""
    if epoch_list is None:
        epoch_list = [3, 5, 10, 15, 20, 25, 30]

    Sequential, Dense, GRU, LSTM, Bidirectional, Conv1D, Flatten = \
        _requires_keras()

    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)
    # Conv1D expects (samples, steps, channels)
    X_train_3d = X_train[:, :, np.newaxis]
    X_val_3d   = X_val[:,   :, np.newaxis]
    X_test_3d  = X_test[:,  :, np.newaxis]

    results = {}
    for epochs in epoch_list:
        print(f"Training CNN — {epochs} epochs …")

        model = Sequential([
            Conv1D(32, 2, activation="relu",
                   input_shape=(X_train.shape[1], 1)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(1,  activation="sigmoid"),
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=["accuracy"])

        t0 = time.time()
        model.fit(X_train_3d, y_train, epochs=epochs, batch_size=64,
                  validation_data=(X_val_3d, y_val), verbose=0, shuffle=False)
        train_time = time.time() - t0

        metrics = _timed_eval_keras(model, X_test_3d, y_test)
        metrics["training_time"] = train_time
        results[epochs] = metrics

    return results


def run_ffnn(X: np.ndarray, y: np.ndarray,
             epoch_list: list = None) -> dict:
    """Train Feed-Forward Neural Network baseline."""
    if epoch_list is None:
        epoch_list = [3, 5, 10, 15, 20, 25, 30]

    Sequential, Dense, *_ = _requires_keras()

    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)

    results = {}
    for epochs in epoch_list:
        print(f"Training FFNN — {epochs} epochs …")

        model = Sequential([
            Dense(100, input_dim=X_train.shape[1], activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=["accuracy"])

        t0 = time.time()
        model.fit(X_train, y_train, epochs=epochs, batch_size=64,
                  validation_data=(X_val, y_val), verbose=0, shuffle=False)
        train_time = time.time() - t0

        metrics = _timed_eval_keras(model, X_test, y_test)
        metrics["training_time"] = train_time
        results[epochs] = metrics

    return results


# ── Machine Learning baselines ────────────────────────────────────────────────

def run_ml_baselines(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Train SVM, Logistic Regression, Naïve Bayes, and Random Forest
    on the same train/test split and return their metrics.

    Parameters
    ----------
    X : np.ndarray  — BERT embeddings
    y : np.ndarray  — binary labels

    Returns
    -------
    dict  { model_name : metrics_dict }
    """
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y)

    classifiers = {
        "SVM":                SVC(kernel="linear", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                   random_state=42),
        "Naive Bayes":        GaussianNB(),
        "Random Forest":      RandomForestClassifier(n_estimators=100,
                                                     random_state=42),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"Training {name} …")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        y_pred = clf.predict(X_test)
        inf_time = time.time() - t1

        metrics = compute_metrics(y_test, y_pred)
        metrics["training_time"]  = train_time
        metrics["inference_time"] = inf_time
        results[name] = metrics
        print(f"  {name}: Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  "
              f"Train={train_time:.2f}s  Inf={inf_time:.4f}s")

    return results
