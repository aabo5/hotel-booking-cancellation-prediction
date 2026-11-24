"""
evaluation.py - Comprehensive Model Evaluation and Performance Metrics

This module provides functions to:
1. Compute standard classification metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC
   across positive class (cancellations), macro-average, and weighted-average.
2. Extract exact Confusion Matrix components (TN, FP, FN, TP).
3. Execute Stratified K-Fold Cross-Validation across multiple metrics.
4. Format and save baseline performance summaries to results/metrics/baseline_results.csv.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


def run_cross_validation(
    model: Any,
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    sample_size: Optional[int] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Executes Stratified K-Fold Cross-Validation on the training set.
    Supports stratified sampling for computationally demanding lazy learners (KNN).
    
    Returns:
        dict: Mean and standard deviation for 'accuracy', 'roc_auc', and 'f1'.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    if sample_size is not None and len(X_train) > sample_size:
        X_cv, _, y_cv, _ = train_test_split(
            X_train, y_train,
            train_size=sample_size,
            stratify=y_train,
            random_state=random_state
        )
    else:
        X_cv, y_cv = X_train, y_train
        
    scores = cross_validate(
        model,
        X_cv,
        y_cv,
        cv=cv,
        scoring=['accuracy', 'roc_auc', 'f1'],
        n_jobs=-1
    )
    
    return {
        "accuracy": (float(np.mean(scores['test_accuracy'])), float(np.std(scores['test_accuracy']))),
        "roc_auc": (float(np.mean(scores['test_roc_auc'])), float(np.std(scores['test_roc_auc']))),
        "f1": (float(np.mean(scores['test_f1'])), float(np.std(scores['test_f1'])))
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    model_name: str = "Model",
    train_time_sec: float = 0.0,
    cv_metrics: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on a held-out test dataset, computing comprehensive metrics:
    - Accuracy
    - Precision, Recall, F1 (Positive class 1, Macro, Weighted)
    - ROC-AUC
    - Confusion Matrix (TN, FP, FN, TP)
    - Inference / Prediction time
    - Integrated 5-fold CV results
    
    Parameters:
        model: Trained scikit-learn compatible classifier
        X_test: Test feature matrix
        y_test: True test target vector (0 or 1)
        model_name: Algorithm name
        train_time_sec: Recorded training time in seconds
        cv_metrics: Output dictionary from run_cross_validation
        
    Returns:
        dict: Full dictionary of performance metrics and prediction arrays
    """
    y_test_arr = np.asarray(y_test).ravel()
    
    # Measure exact inference time
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time_sec = round(time.time() - start_pred, 4)
    
    # Obtain predicted probabilities for class 1
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        # Min-max scale decision function to [0, 1] for ROC calculation consistency
        y_prob = (df - df.min()) / (df.max() - df.min() + 1e-12)
    else:
        y_prob = y_pred.astype(float)
        
    acc = accuracy_score(y_test_arr, y_pred)
    
    # Positive class (1 = Canceled)
    prec_pos = precision_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
    rec_pos = recall_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_test_arr, y_pred, pos_label=1, zero_division=0)
    
    # Macro averages
    prec_macro = precision_score(y_test_arr, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test_arr, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test_arr, y_pred, average="macro", zero_division=0)
    
    # Weighted averages
    prec_weighted = precision_score(y_test_arr, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test_arr, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test_arr, y_pred, average="weighted", zero_division=0)
    
    roc_auc = roc_auc_score(y_test_arr, y_prob)
    
    # Confusion matrix extraction
    cm = confusion_matrix(y_test_arr, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "precision_pos": round(float(prec_pos), 4),
        "recall_pos": round(float(rec_pos), 4),
        "f1_pos": round(float(f1_pos), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc": round(float(roc_auc), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_time_sec": round(float(train_time_sec), 2),
        "pred_time_sec": round(float(pred_time_sec), 4),
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    
    if cv_metrics is not None:
        metrics["cv_mean_accuracy"] = round(cv_metrics["accuracy"][0], 4)
        metrics["cv_std_accuracy"] = round(cv_metrics["accuracy"][1], 4)
        metrics["cv_mean_roc_auc"] = round(cv_metrics["roc_auc"][0], 4)
        metrics["cv_std_roc_auc"] = round(cv_metrics["roc_auc"][1], 4)
        metrics["cv_mean_f1"] = round(cv_metrics["f1"][0], 4)
        metrics["cv_std_f1"] = round(cv_metrics["f1"][1], 4)
    else:
        metrics["cv_mean_accuracy"] = np.nan
        metrics["cv_std_accuracy"] = np.nan
        metrics["cv_mean_roc_auc"] = np.nan
        metrics["cv_std_roc_auc"] = np.nan
        metrics["cv_mean_f1"] = np.nan
        metrics["cv_std_f1"] = np.nan
        
    return metrics


def save_metrics_to_csv(
    metrics_list: List[Dict[str, Any]],
    output_path: str = "results/metrics/baseline_results.csv"
) -> pd.DataFrame:
    """
    Converts a list of evaluation metric dictionaries into a clean summary DataFrame
    and saves it to CSV in the results/metrics/ directory.
    Excludes large non-scalar artifacts (arrays, confusion matrices).
    """
    df_metrics = pd.DataFrame(metrics_list)
    excluded = ["confusion_matrix", "y_pred", "y_prob"]
    cols_to_save = [c for c in df_metrics.columns if c not in excluded]
    summary_df = df_metrics[cols_to_save]
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out, index=False)
    print(f"Saved evaluation metrics to: {out}")
    return summary_df
