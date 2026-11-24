"""
run_baseline_pipeline.py - Standalone Execution Pipeline for Baseline Models

This script executes the complete Phase 6 baseline evaluation:
1. Loads preprocessed training and test partitions (Parquet & CSV).
2. Runs 5-Fold Stratified Cross-Validation on the training partition.
3. Fits each of the 6 baseline models on the full training partition.
4. Evaluates performance on the held-out test partition (N = 23,842).
5. Generates and persists publication-grade figures in results/figures/.
6. Persists the consolidated metrics table to results/metrics/baseline_results.csv.
"""

import time
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

from src.evaluation import run_cross_validation, evaluate_model, save_metrics_to_csv
from src.plotting import (
    plot_baseline_confusion_matrices,
    plot_baseline_roc_curves,
    plot_baseline_metrics_comparison,
    plot_baseline_computation_time
)


def main():
    print("=" * 80)
    print("PHASE 6: BASELINE MACHINE LEARNING MODELS PIPELINE")
    print("=" * 80)

    # 1. Load Data
    print("\n[1/5] Loading preprocessed data from data/processed/...")
    data_dir = Path("data/processed")
    X_train_df = pd.read_parquet(data_dir / "X_train.parquet")
    X_test_df = pd.read_parquet(data_dir / "X_test.parquet")
    y_train_df = pd.read_csv(data_dir / "y_train.csv")
    y_test_df = pd.read_csv(data_dir / "y_test.csv")

    X_train = X_train_df.values
    X_test = X_test_df.values
    y_train = y_train_df.values.ravel()
    y_test = y_test_df.values.ravel()

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape},  y_test shape:  {y_test.shape}")
    print(f"Target distribution (train): Canceled={np.mean(y_train)*100:.2f}%, Non-Canceled={(1-np.mean(y_train))*100:.2f}%")
    print(f"Target distribution (test):  Canceled={np.mean(y_test)*100:.2f}%, Non-Canceled={(1-np.mean(y_test))*100:.2f}%")

    # 2. Define Model Configurations
    models = {
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "cv_sample": 20000,
            "paradigm": "Lazy Learner (Instance-Based)",
            "inductive_bias": "Locality assumption in Euclidean feature space"
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "cv_sample": None,
            "paradigm": "Eager Learner (Non-parametric Tree)",
            "inductive_bias": "Orthogonal axis-aligned recursive binary splits minimizing Gini impurity"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "cv_sample": None,
            "paradigm": "Eager Learner (Bagging Ensemble)",
            "inductive_bias": "Variance reduction via bootstrap aggregation and random subspace projection"
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
            "cv_sample": None,
            "paradigm": "Eager Learner (Gradient Boosting)",
            "inductive_bias": "Sequential gradient descent in function space with 2nd-order Taylor loss expansion"
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            "cv_sample": None,
            "paradigm": "Eager Learner (Histogram Boosting)",
            "inductive_bias": "Leaf-wise best-first growth with Gradient-based One-Side Sampling (GOSS)"
        },
        "Multilayer Perceptron": {
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True),
            "cv_sample": None,
            "paradigm": "Eager Learner (Parametric Neural Network)",
            "inductive_bias": "Smooth composition of affine transforms and non-linear ReLU activations"
        }
    }

    # 3. Model Training, Cross-Validation & Test Evaluation
    print("\n[2/5] Training, Cross-Validating, and Evaluating Baseline Models...")
    results_list = []
    trained_models = {}

    for name, config in models.items():
        print(f"\n---> Evaluating: {name} [{config['paradigm']}]")
        clf = config["model"]
        sample_sz = config["cv_sample"]

        # Stratified 5-Fold Cross-Validation
        print(f"  Running 5-Fold Stratified Cross-Validation (sample_size={sample_sz or 'Full 95,366'})...")
        t_cv_start = time.time()
        cv_metrics = run_cross_validation(clf, X_train, y_train, n_splits=5, random_state=42, sample_size=sample_sz)
        cv_elapsed = time.time() - t_cv_start
        print(f"  CV Complete ({cv_elapsed:.2f}s) | ROC-AUC: {cv_metrics['roc_auc'][0]:.4f} +/- {cv_metrics['roc_auc'][1]:.4f} | Acc: {cv_metrics['accuracy'][0]:.4f} | F1: {cv_metrics['f1'][0]:.4f}")

        # Training on full X_train
        print("  Fitting on full training dataset (N = 95,366)...")
        t_fit_start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t_fit_start
        print(f"  Fit Complete ({train_time:.2f}s)")

        # Evaluate on held-out X_test
        print("  Evaluating on held-out test partition (N = 23,842)...")
        eval_dict = evaluate_model(
            model=clf,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            train_time_sec=train_time,
            cv_metrics=cv_metrics
        )
        print(f"  Test Results: Acc={eval_dict['accuracy']:.4f} | Precision={eval_dict['precision_pos']:.4f} | Recall={eval_dict['recall_pos']:.4f} | F1={eval_dict['f1_pos']:.4f} | ROC-AUC={eval_dict['roc_auc']:.4f} | Pred Time={eval_dict['pred_time_sec']:.4f}s")
        print(f"  Confusion Matrix: TN={eval_dict['tn']}, FP={eval_dict['fp']}, FN={eval_dict['fn']}, TP={eval_dict['tp']}")

        results_list.append(eval_dict)
        trained_models[name] = clf

    # 4. Save Consolidated Metrics Table
    print("\n[3/5] Saving evaluation metrics to results/metrics/baseline_results.csv...")
    summary_df = save_metrics_to_csv(results_list, output_path="results/metrics/baseline_results.csv")
    print("\nSummary Metrics Table:")
    display_cols = ["model", "accuracy", "precision_pos", "recall_pos", "f1_pos", "roc_auc", "cv_mean_roc_auc", "train_time_sec", "pred_time_sec"]
    print(summary_df[display_cols].to_string(index=False))

    # 5. Generate Publication Figures
    print("\n[4/5] Generating Publication Figures in results/figures/...")
    
    # 5.1 Confusion Matrices Grid
    print("  Generating baseline_confusion_matrices.png...")
    plot_baseline_confusion_matrices(results_list, save_path="results/figures/baseline_confusion_matrices.png")

    # 5.2 ROC Curves
    print("  Generating baseline_roc_curves.png...")
    plot_baseline_roc_curves(results_list, y_test, save_path="results/figures/baseline_roc_curves.png")

    # 5.3 Metrics Comparison Grouped Bar Chart
    print("  Generating baseline_metrics_comparison.png...")
    plot_baseline_metrics_comparison(summary_df, save_path="results/figures/baseline_metrics_comparison.png")

    # 5.4 Computation Time Comparison (Lazy vs. Eager)
    print("  Generating baseline_computation_time.png...")
    plot_baseline_computation_time(summary_df, save_path="results/figures/baseline_computation_time.png")

    print("\n[5/5] Pipeline execution finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
