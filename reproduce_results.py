"""
Reproduce EPRVFL results from:

    R. K. Gurjwar, A. Kumar, U. P. Rao,
    "EPRVFL: A fast and scalable model for real-time fake news detection,"
    Pattern Recognition Letters, vol. 196, pp. 267-273, 2025.

Usage
-----
# Quick demo on synthetic data (no GPU or downloads needed):
    python reproduce_results.py --demo

# Full reproduction on a real dataset:
    python reproduce_results.py \
        --embeddings_path data/politifact_bert.npz \
        --epochs 3 5 10 15 20 25 30 \
        --run_baselines

# Generate BERT embeddings from a raw CSV first:
    python reproduce_results.py \
        --generate_embeddings \
        --csv_path data/politifact.csv \
        --save_path data/politifact_bert.npz

Dataset download links are in README.md.
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from model.eprvfl import EPRVFL, evaluate
from baselines.baselines import (run_gru, run_lstm, run_cnn,
                                 run_ffnn, run_ml_baselines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_metrics(label: str, metrics: dict) -> None:
    print(f"  {label:<30} "
          f"Acc={metrics['accuracy']:.4f}  "
          f"Pre={metrics['precision']:.4f}  "
          f"Rec={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}")


def run_eprvfl_epochs(X_train, y_train, X_val, y_val,
                      epoch_list, num_hidden=23, C=0.03125):
    """Run EPRVFL across multiple epoch counts and return per-epoch metrics."""
    results = {}
    for epochs in epoch_list:
        t0 = time.time()
        model = EPRVFL(num_hidden_nodes=num_hidden, C=C)
        model.fit_epochs(X_train, y_train, epochs=epochs)
        train_time = time.time() - t0

        t1 = time.time()
        metrics = evaluate(model, X_val, y_val)
        inf_time = time.time() - t1

        metrics["training_time"]  = train_time
        metrics["inference_time"] = inf_time
        results[epochs] = metrics
        print_metrics(f"EPRVFL (epochs={epochs})", metrics)

    return results


def build_comparison_table(eprvfl_results: dict,
                           ml_results: dict) -> None:
    """Print a summary comparison table matching Table in the paper."""
    print("\n" + "=" * 80)
    print(f"{'Model':<28} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'Inf Time':>10}")
    print("=" * 80)

    # Best EPRVFL epoch (highest accuracy)
    best_epoch = max(eprvfl_results,
                     key=lambda e: eprvfl_results[e]["accuracy"])
    m = eprvfl_results[best_epoch]
    print(f"  {'EPRVFL (best epoch)':<26} "
          f"{m['accuracy']*100:>8.2f}%  "
          f"{m['precision']*100:>8.2f}%  "
          f"{m['recall']*100:>7.2f}%  "
          f"{m['f1']*100:>6.2f}%  "
          f"{m['inference_time']:>9.4f}s")

    for name, metrics in ml_results.items():
        print(f"  {name:<26} "
              f"{metrics['accuracy']*100:>8.2f}%  "
              f"{metrics['precision']*100:>8.2f}%  "
              f"{metrics['recall']*100:>7.2f}%  "
              f"{metrics['f1']*100:>6.2f}%  "
              f"{metrics['inference_time']:>9.4f}s")

    print("=" * 80)
    print(f"  (* Best EPRVFL epoch: {best_epoch})")


# ── Demo mode ─────────────────────────────────────────────────────────────────

def demo_mode() -> None:
    """
    Smoke-test on synthetic embeddings.
    Validates that EPRVFL trains and evaluates without error.
    No downloads required.
    """
    print("\n" + "─" * 60)
    print("  DEMO MODE — synthetic BERT-sized embeddings (768-d)")
    print("─" * 60)

    rng = np.random.RandomState(0)
    n_samples, embedding_dim = 1000, 768
    X = rng.randn(n_samples, embedding_dim).astype(np.float32)
    # Fake news = 1 when first feature > 0 (linearly separable toy problem)
    y = (X[:, 0] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    epoch_list = [3, 5, 10, 15, 20]
    print(f"\nTraining EPRVFL on {len(X_train)} synthetic samples …\n")
    for epochs in epoch_list:
        model = EPRVFL(num_hidden_nodes=23, C=0.03125)
        model.fit_epochs(X_train, y_train, epochs=epochs)
        metrics = evaluate(model, X_test, y_test)
        print_metrics(f"EPRVFL (epochs={epochs})", metrics)

    print("\n✓ Demo complete. "
          "To run on real data, see --help for dataset instructions.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce EPRVFL results (Gurjwar et al., PRL 2025)")
    p.add_argument("--demo", action="store_true",
                   help="Quick smoke-test on synthetic data (no downloads).")
    p.add_argument("--generate_embeddings", action="store_true",
                   help="Generate BERT embeddings from a raw CSV.")
    p.add_argument("--csv_path", type=str,
                   help="Path to raw dataset CSV (with 'text' and 'label' cols).")
    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--embeddings_path", type=str,
                   help="Path to pre-computed .npz embeddings file.")
    p.add_argument("--save_path", type=str, default="data/embeddings.npz",
                   help="Where to save generated embeddings.")
    p.add_argument("--epochs", type=int, nargs="+",
                   default=[3, 5, 10, 15, 20, 25, 30],
                   help="Epoch counts to evaluate EPRVFL at.")
    p.add_argument("--run_baselines", action="store_true",
                   help="Also run ML baselines (SVM, LR, NB, RF).")
    p.add_argument("--run_dl_baselines", action="store_true",
                   help="Also run DL baselines (GRU, LSTM, CNN, FFNN). "
                        "Requires TensorFlow.")
    p.add_argument("--save_results", type=str, default=None,
                   help="Optional path to save results as JSON.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Demo ──────────────────────────────────────────────────────────────────
    if args.demo:
        demo_mode()
        return

    # ── Generate embeddings ───────────────────────────────────────────────────
    if args.generate_embeddings:
        if not args.csv_path:
            raise ValueError("--csv_path is required with --generate_embeddings")
        from embeddings import load_dataset, generate_bert_embeddings, save_embeddings
        texts, labels = load_dataset(args.csv_path, args.text_col, args.label_col)
        embeddings = generate_bert_embeddings(texts)
        save_embeddings(embeddings, labels, args.save_path)
        print(f"\n✓ Embeddings saved to {args.save_path}")
        return

    # ── Load embeddings ───────────────────────────────────────────────────────
    if not args.embeddings_path:
        print("No action specified. Run with --demo, --generate_embeddings, "
              "or --embeddings_path.\nUse --help for full usage.")
        return

    from embeddings import load_embeddings
    print(f"\nLoading embeddings from {args.embeddings_path} …")
    X, y = load_embeddings(args.embeddings_path)
    print(f"Shape: {X.shape}  |  Labels: {np.bincount(y)}\n")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42)

    all_results = {}

    # ── EPRVFL ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("EPRVFL")
    print("=" * 60)
    eprvfl_results = run_eprvfl_epochs(
        X_train, y_train, X_val, y_val, args.epochs)
    all_results["EPRVFL"] = eprvfl_results

    # ── ML Baselines ──────────────────────────────────────────────────────────
    ml_results = {}
    if args.run_baselines:
        print("\n" + "=" * 60)
        print("ML Baselines (SVM / LR / NB / RF)")
        print("=" * 60)
        ml_results = run_ml_baselines(X, y)
        all_results["ML_baselines"] = ml_results

    # ── DL Baselines ──────────────────────────────────────────────────────────
    if args.run_dl_baselines:
        for name, fn, kwargs in [
            ("GRU",    run_gru,  {"bidirectional": False}),
            ("BiGRU",  run_gru,  {"bidirectional": True}),
            ("LSTM",   run_lstm, {"bidirectional": False}),
            ("BiLSTM", run_lstm, {"bidirectional": True}),
            ("CNN",    run_cnn,  {}),
            ("FFNN",   run_ffnn, {}),
        ]:
            print(f"\n{'='*60}\n{name}\n{'='*60}")
            res = fn(X, y, epoch_list=args.epochs, **kwargs)
            all_results[name] = res

    # ── Comparison table ──────────────────────────────────────────────────────
    if ml_results:
        build_comparison_table(eprvfl_results, ml_results)

    # ── Save results ──────────────────────────────────────────────────────────
    if args.save_results:
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\n✓ Results saved to {args.save_results}")


if __name__ == "__main__":
    main()
