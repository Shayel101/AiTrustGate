"""Train and evaluate the AI Trust Gate baseline classifier.

Pipeline: TF-IDF (Term Frequency-Inverse Document Frequency) -> Logistic Regression -> benign/harmful prediction.

Usage:
    python src/train.py

Requires data/processed/jbb_behavior_classifier.csv to exist; run
src/prepare_data.py first if it does not.

Outputs:
    results/predictions.csv
    results/metrics.json
    results/evaluation.txt

Hyperparameters (sublinear_tf=True, C=0.3) were chosen via 5-fold stratified
cross-validation on the training split only (see
docs/ISE_A1_evidence_brief.md, "Negative results" section); the held-out test
set was evaluated exactly once with the selected config.
"""
from __future__ import annotations

import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from evaluate import build_evaluation_report, compute_metrics, find_misclassified
from prepare_data import OUTPUT_PATH, prepare_dataset

RANDOM_SEED = 42
TEST_SIZE = 0.2
TFIDF_KWARGS = {"sublinear_tf": True}
LOGREG_KWARGS = {"C": 0.3, "max_iter": 2000}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "predictions.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
EVALUATION_PATH = os.path.join(RESULTS_DIR, "evaluation.txt")


def load_processed_dataset() -> pd.DataFrame:
    """Load the processed dataset, generating it from raw CSVs if it is missing."""
    if not os.path.exists(OUTPUT_PATH):
        print(f"{OUTPUT_PATH} not found; running data preparation first.")
        return prepare_dataset()
    return pd.read_csv(OUTPUT_PATH)


def main() -> None:
    df = load_processed_dataset()

    # Stratified 80/20 split keeps the benign/harmful ratio balanced in both splits.
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])

    model = LogisticRegression(random_state=RANDOM_SEED, **LOGREG_KWARGS)

    # Cross-validate on the training split only (never touches the test set) to
    # report an honest estimate of variance alongside the single test-set score.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        LogisticRegression(random_state=RANDOM_SEED, **LOGREG_KWARGS),
        TfidfVectorizer(**TFIDF_KWARGS).fit_transform(train_df["text"]),
        train_df["label"],
        cv=cv,
    )

    model.fit(x_train, train_df["label"])

    y_pred = model.predict(x_test)
    y_true = test_df["label"]

    metrics = compute_metrics(y_true, y_pred)
    metrics_out = {
        "dataset_size": len(df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "random_seed": RANDOM_SEED,
        "test_split_fraction": TEST_SIZE,
        "model": "TF-IDF (sublinear_tf=True) + LogisticRegression (C=0.3)",
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "cv_accuracy_folds": cv_scores.tolist(),
        **metrics,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    report_text = build_evaluation_report(y_true, y_pred)
    header = (
        f"Dataset size: {len(df)} | Train size: {len(train_df)} | "
        f"Test size: {len(test_df)} | Random seed: {RANDOM_SEED}\n"
        f"Model: {metrics_out['model']}\n"
        f"Accuracy (held-out test set): {metrics['accuracy']:.4f}\n"
        f"5-fold CV accuracy on training split: {cv_scores.mean():.4f} "
        f"(+/- {cv_scores.std():.4f})\n\n"
    )
    with open(EVALUATION_PATH, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    predictions_out = test_df.copy()
    predictions_out["predicted_label"] = y_pred
    predictions_out["correct"] = predictions_out["label"] == predictions_out["predicted_label"]
    predictions_out.to_csv(PREDICTIONS_PATH, index=False)

    misclassified = find_misclassified(test_df, y_true, y_pred)

    print(header)
    print(report_text)
    print(f"Misclassified rows: {len(misclassified)} / {len(test_df)}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved evaluation report to {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
