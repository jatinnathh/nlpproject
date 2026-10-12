# NLP Lab Project — Hindi Fake News Detection

Binary classification (0 = Real, 1 = Fake) on Hindi news articles using `google/muril-base-cased`.

## Setup

```bash
pip install torch transformers datasets accelerate scikit-learn pandas
```

## Training (requires GPU recommended)

```bash
cd backend
python3 train_muril.py
```

- Trains for up to 5 epochs with early stopping (patience=2)
- Best model saved to `backend/best_model/`
- Test predictions saved to `backend/best_model/test_predictions.csv`

## Expected Results
- Metric: Macro-F1 on validation set
- Baseline (TF-IDF + LogReg): run `03_Baselines.ipynb`
- Transformer fine-tune: `train_muril.py` or `05_Transformer_FineTuning.ipynb`

## Data
Processed splits are in `backend/data/processed/` (train / validation / test CSVs).
Raw data files are excluded from git (too large) — contact repo owner for access.
