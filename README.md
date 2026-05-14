# EPRVFL: Fake News Detection

**Official replication code for:**

> R. K. Gurjwar, A. Kumar, U. P. Rao,  
> *"EPRVFL: A fast and scalable model for real-time fake news detection,"*  
> **Pattern Recognition Letters**, vol. 196, pp. 267–273, 2025.  
> DOI: [10.1016/j.patrec.2025.06.006](https://doi.org/10.1016/j.patrec.2025.06.006)

---

## What is EPRVFL?

EPRVFL (**E**mbedding-based **P**arallel **R**andom **V**ector **F**unctional **L**ink Network) is a hybrid model for real-time fake news detection that combines:

- **Pre-trained transformer embeddings** (BERT) for rich semantic representations
- **RVFL hidden layer** with a closed-form analytical solution for output weights

This design makes EPRVFL significantly faster than deep learning alternatives (GRU, LSTM, CNN) at inference time, while achieving competitive or superior accuracy on benchmark datasets.

### Key results (Pattern Recognition Letters 2025)

| Model | Accuracy | F1-Score | Inference Time |
|-------|----------|----------|----------------|
| **EPRVFL** | **91.77%** | **91.81%** | **fastest** |
| BERT + RVFL | 91.00% | 90.93% | fast |
| GRU | ~85% | ~84% | slow |
| LSTM | ~84% | ~83% | slow |
| SVM | ~78% | ~77% | medium |

*Results on PolitiFact dataset with BERT embeddings.*

---

## Architecture
<img width="1363" height="524" alt="Architecture of Proposed EPRVFL Model" src="https://github.com/user-attachments/assets/689497f3-4b7c-4666-9416-41d25080f6d6" />
```
Raw News Text
      │
      ▼
BERT Encoder (bert-base-uncased)
      │  mean-pool over tokens
      ▼
BERT Embeddings  [n × 768]
      │
      ├────────────────────────────┐
      │                            │
      ▼                            ▼
 Original X                 RVFL Hidden Layer
 [n × 768]              tanh(X @ W_random + b)
                              [n × 23]
      │                            │
      └──────────┬─────────────────┘
                 │  concatenate
                 ▼
         H = [X | H_hidden]   [n × 791]
                 │
                 ▼
     Output weights β (closed-form)
     β = (HᵀH + λI)⁻¹ Hᵀ y
                 │
                 ▼
           Prediction
```

**Why closed-form?** No gradient descent — output weights are solved analytically using the Moore-Penrose pseudoinverse. This makes EPRVFL extremely fast at inference.

---

## Repository structure

```
eprvfl/
├── model/
│   └── eprvfl.py           ← EPRVFL class (core model from the paper)
├── baselines/
│   └── baselines.py        ← GRU, BiGRU, LSTM, BiLSTM, CNN, FFNN, SVM, LR, NB, RF
├── embeddings.py           ← BERT embedding generation utilities
├── reproduce_results.py    ← Main script to reproduce paper results
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
git clone https://github.com/grjwr/eprvfl.git
cd eprvfl
pip install -r requirements.txt
```

### 2. Run demo (no downloads needed)

Verify the installation works using synthetic data:

```bash
python reproduce_results.py --demo
```

Expected output:
```
  EPRVFL (epochs=3)             Acc=0.9900  Pre=0.9900  Rec=0.9900  F1=0.9900
  EPRVFL (epochs=5)             Acc=0.9900  Pre=0.9900  Rec=0.9900  F1=0.9900
  ...
✓ Demo complete.
```

### 3. Generate BERT embeddings from your dataset

Your CSV should have at minimum a `text` column and a `label` column (0 = real, 1 = fake):

```bash
python reproduce_results.py \
    --generate_embeddings \
    --csv_path data/politifact.csv \
    --save_path data/politifact_bert.npz
```

### 4. Reproduce EPRVFL results

```bash
python reproduce_results.py \
    --embeddings_path data/politifact_bert.npz \
    --epochs 3 5 10 15 20 25 30
```

### 5. Run with all ML baselines

```bash
python reproduce_results.py \
    --embeddings_path data/politifact_bert.npz \
    --epochs 3 5 10 15 20 25 30 \
    --run_baselines \
    --save_results results/politifact_results.json
```

---

## Datasets

All datasets used in the paper are publicly available:

| Dataset | Source | Description |
|---------|--------|-------------|
| **PolitiFact** | [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) | Political news, fact-checked by PolitiFact |
| **BuzzFeed-Webis** | [Zenodo](https://zenodo.org/record/1239675) | News articles from 2016 US election |
| **LIAR-2** | [HuggingFace](https://huggingface.co/datasets/chengxuphd/liar2) | Extended LIAR dataset with 6-way labels (binarised) |
| **GossipCop** | [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) | Entertainment/celebrity news |

---

## Using EPRVFL in your own project

```python
import numpy as np
from model.eprvfl import EPRVFL, evaluate

# X: pre-computed BERT embeddings [n_samples, 768]
# y: binary labels [n_samples]

model = EPRVFL(num_hidden_nodes=23, C=0.03125)
model.fit_epochs(X_train, y_train, epochs=10)

metrics = evaluate(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_hidden_nodes` | 23 | RVFL random hidden nodes |
| `C` | 0.03125 | Regularisation (1/ridge penalty) |
| Transformer | `bert-base-uncased` | Embedding model |
| Max sequence length | 512 tokens | BERT input limit |
| Train/Val/Test split | 70/15/15 | Random state = 42 |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gurjwar2025eprvfl,
  title={EPRVFL: A fast and scalable model for real-time fake news detection},
  author={Gurjwar, Rajiv Kumar and Kumar, Alok and Rao, Udai Pratap},
  journal={Pattern Recognition Letters},
  volume={196},
  pages={267--273},
  year={2025},
  publisher={Elsevier},
  doi = {10.1016/j.patrec.2025.06.006},
}
```
## Related Work
- [LLM Benchmark: Mistral-7B LoRA vs EPRVFL](https://github.com/grjwr/LLM-FakeNews-Benchmark)
---

## Author

**Rajiv Kumar Gurjwar**  
PhD Research Fellow, SVNIT Surat  
[Google Scholar](https://scholar.google.com/citations?user=_3_1ExAAAAAJ) · [ORCID](https://orcid.org/0000-0002-2292-8972) · [LinkedIn](https://www.linkedin.com/in/rajiv-gurjwar)

> *This repository contains replication code for the published paper only.*  
> *Extended research code is not included.*

---

## Licence

MIT © Rajiv Kumar Gurjwar, 2025
