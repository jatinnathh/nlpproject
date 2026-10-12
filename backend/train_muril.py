"""
MuRIL Fine-Tuning for Hindi Fake News Detection
================================================
Model  : google/muril-base-cased
Task   : Binary classification — 0 = Real, 1 = Fake
Metric : Macro-F1 (primary), Accuracy (secondary)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "google/muril-base-cased"
DATA_DIR     = "data/processed"
OUTPUT_DIR   = "./results"
BEST_DIR     = "./best_model"
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-5
WEIGHT_DECAY = 0.01
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"  Python  : {sys.executable}")
print(f"  PyTorch : {torch.__version__}")
print(f"  Device  : {DEVICE}")
print(f"{'='*60}\n")

# ── Load Data ────────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv").dropna(subset=["text"])
val_df   = pd.read_csv(f"{DATA_DIR}/validation.csv").dropna(subset=["text"])
test_df  = pd.read_csv(f"{DATA_DIR}/test.csv").dropna(subset=["text"])

print(f"  Train : {len(train_df):,}  |  Val : {len(val_df):,}  |  Test : {len(test_df):,}")
print(f"  Train label dist:\n{train_df['label'].value_counts().to_string()}\n")

# ── Tokenizer ────────────────────────────────────────────────────────────────
print(f"Loading tokenizer from '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize, batched=True)
test_ds  = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
val_ds.set_format  ("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
test_ds.set_format ("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

# ── Class Weights (handle imbalance) ─────────────────────────────────────────
counts = train_df["label"].value_counts().sort_index().values
class_weights = torch.tensor(
    (1.0 - counts / counts.sum()), dtype=torch.float32
).to(DEVICE)
print(f"Class weights: {class_weights.tolist()}\n")

# ── Model ────────────────────────────────────────────────────────────────────
print(f"Loading model from '{MODEL_NAME}'...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

# ── Custom Trainer with weighted loss ────────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)(
            logits.view(-1, model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1, "precision_macro": prec, "recall_macro": rec}

# ── Training Arguments ───────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=50,
    seed=SEED,
    report_to="none",
    fp16=DEVICE == "cuda",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── Train ────────────────────────────────────────────────────────────────────
print("Starting training...\n")
trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(BEST_DIR, exist_ok=True)
trainer.save_model(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)
print(f"\nBest model saved to '{BEST_DIR}'\n")

# ── Evaluate on Test Set ──────────────────────────────────────────────────────
print("Evaluating on test set...")
test_results = trainer.predict(test_ds)
test_preds   = np.argmax(test_results.predictions, axis=-1)
test_labels  = test_df["label"].values

print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)
print(classification_report(test_labels, test_preds, target_names=["Real (0)", "Fake (1)"]))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_preds))

# Save predictions
test_df["predicted_label"] = test_preds
test_df.to_csv(f"{BEST_DIR}/test_predictions.csv", index=False)
print(f"\nPredictions saved to '{BEST_DIR}/test_predictions.csv'")
