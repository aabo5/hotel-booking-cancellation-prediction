"""
plotting.py - Visualization Utilities for Model Evaluation and Publication Graphics

This module provides publication-quality plotting functions for:
1. Baseline Confusion Matrices (2x3 Grid with annotations and proportions)
2. Combined Baseline ROC Curves with AUC scoring
3. Baseline Model Performance Metrics Comparison (Grouped Bar Chart)
4. Baseline Computational Efficiency (Training vs Inference Time: Lazy vs Eager Learners)
5. Standalone single-model diagnostics and feature importance charts
6. Tuned Confusion Matrices (2x3 Grid)
7. Tuned ROC Curves
8. Tuning Performance Comparison (Baseline vs Tuned with delta annotations)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# Set publication-standard aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def plot_baseline_confusion_matrices(
    results_list: List[Dict[str, Any]],
    save_path: str = "results/figures/baseline_confusion_matrices.png"
) -> None:
    """
    Plots a 2x3 grid of confusion matrices for all 6 baseline classifiers.
    Includes cell counts, normalized percentages, and clear TP/TN/FP/FN designations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()
    
    cmap = sns.light_palette("navy", as_cmap=True)
    
    for idx, res in enumerate(results_list):
        ax = axes[idx]
        cm = np.array(res["confusion_matrix"])
        total = np.sum(cm)
        
        labels = [
            [f"TN\n{cm[0,0]:,}\n({cm[0,0]/total*100:.1f}%)", f"FP\n{cm[0,1]:,}\n({cm[0,1]/total*100:.1f}%)"],
            [f"FN\n{cm[1,0]:,}\n({cm[1,0]/total*100:.1f}%)", f"TP\n{cm[1,1]:,}\n({cm[1,1]/total*100:.1f}%)"]
        ]
        
        sns.heatmap(
            cm,
            annot=np.array(labels),
            fmt="",
            cmap=cmap,
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
            linewidths=1.2,
            linecolor="white"
        )
        
        ax.set_title(f"{res['model']}\n(Acc: {res['accuracy']:.3f} | F1: {res['f1_pos']:.3f})", 
                     fontsize=12, fontweight="bold", pad=8)
        ax.set_xticklabels(["Not Canceled (0)", "Canceled (1)"], fontsize=10)
        ax.set_yticklabels(["Not Canceled (0)", "Canceled (1)"], fontsize=10, rotation=0)
        ax.set_xlabel("Predicted Class", fontsize=10, labelpad=6)
        ax.set_ylabel("True Class", fontsize=10, labelpad=6)
        
    plt.suptitle("Baseline Model Evaluation: Confusion Matrices (Held-out Test Set, N = 23,842)",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved baseline confusion matrices figure: {out}")
    plt.close()


def plot_baseline_roc_curves(
    results_list: List[Dict[str, Any]],
    y_test: np.ndarray,
    save_path: str = "results/figures/baseline_roc_curves.png"
) -> None:
    """
    Plots overlaid Receiver Operating Characteristic (ROC) curves for all 6 models.
    Displays AUC in the legend, sorted by performance, with chance diagonal.
    """
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    
    sorted_results = sorted(results_list, key=lambda x: x["roc_auc"], reverse=True)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    linestyles = ["-", "-", "--", "-.", ":", "-"]
    
    y_test_arr = np.asarray(y_test).ravel()
    
    for idx, res in enumerate(sorted_results):
        y_prob = res["y_prob"]
        fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
        score = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=colors[idx % len(colors)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2.2,
            label=f"{res['model']} (AUC = {score:.4f})"
        )
        
    ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")
    
    ax.set_title("Baseline Classifiers: Receiver Operating Characteristic (ROC) Comparison", 
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95, facecolor="white")
    
    plt.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved baseline ROC curves figure: {out}")
    plt.close()


def plot_baseline_metrics_comparison(
    summary_df: pd.DataFrame,
    save_path: str = "results/figures/baseline_metrics_comparison.png"
) -> None:
    """
    Plots a grouped bar chart comparing Accuracy, Precision (Class 1), Recall (Class 1),
    F1-Score (Class 1), and ROC-AUC across all 6 baseline models.
    """
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    
    models = summary_df["model"].tolist()
    metrics = ["accuracy", "precision_pos", "recall_pos", "f1_pos", "roc_auc"]
    metric_labels = ["Accuracy", "Precision (Class 1)", "Recall (Class 1)", "F1-Score (Class 1)", "ROC-AUC"]
    colors = ["#2b5c8f", "#418ab3", "#e67e22", "#27ae60", "#8e44ad"]
    
    x = np.arange(len(models))
    n_metrics = len(metrics)
    total_width = 0.8
    bar_width = total_width / n_metrics
    
    for i, (m, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = summary_df[m].values
        offset = (i - (n_metrics - 1) / 2) * bar_width
        rects = ax.bar(x + offset, values, width=bar_width, label=label, color=color, edgecolor="black", linewidth=0.5, alpha=0.9)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, rotation=0, fontweight="bold"
            )
            
    ax.set_title("Baseline Machine Learning Models: Comprehensive Metric Comparison", 
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Metric Score", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim([0.0, 1.12])
    ax.axhline(0.80, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=5, fontsize=9.5, frameon=True)
    
    plt.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved baseline metrics comparison figure: {out}")
    plt.close()


def plot_baseline_computation_time(
    summary_df: pd.DataFrame,
    save_path: str = "results/figures/baseline_computation_time.png"
) -> None:
    """
    Plots a dual-panel bar chart illustrating training vs inference computation time.
    Visually demonstrates the fundamental distinction between Lazy (KNN) and Eager learners.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    models = summary_df["model"].tolist()
    train_times = summary_df["train_time_sec"].values
    pred_times = summary_df["pred_time_sec"].values
    
    palette = ["#e74c3c" if "Neighbors" in m else "#3498db" for m in models]
    
    # 1. Training Time (Fit)
    bars1 = ax1.bar(models, train_times, color=palette, edgecolor="black", linewidth=0.6, alpha=0.88)
    ax1.set_title("Training Execution Time (Seconds)\nEager vs. Lazy Construction", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Time (seconds)", fontsize=10)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(train_times)*0.02), f"{yval:.2f}s", 
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    ax1.annotate(
        "Lazy Learner:\nInstantaneous Fit ($O(1)$)",
        xy=(0, train_times[0]), xytext=(0.5, max(train_times)*0.6),
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
        fontsize=9, fontweight="bold", color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffdddd", ec="crimson", lw=1)
    )
    
    # 2. Inference Time (Predict on N=23,842 test queries)
    bars2 = ax2.bar(models, pred_times, color=palette, edgecolor="black", linewidth=0.6, alpha=0.88)
    ax2.set_title("Test Inference Time (Seconds, N = 23,842)\nOnline Query Latency", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Time (seconds)", fontsize=10)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(pred_times)*0.02), f"{yval:.2f}s", 
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    ax2.annotate(
        "Lazy Learner Bottleneck:\n$O(N \\cdot d)$ Query Cost!",
        xy=(0, pred_times[0]), xytext=(0.8, max(pred_times)*0.75),
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
        fontsize=9, fontweight="bold", color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffdddd", ec="crimson", lw=1)
    )
    
    plt.suptitle("Computational Efficiency Trade-Off: Training (Fit) vs. Inference (Predict)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved baseline computation time figure: {out}")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Canceled (0)", "Canceled (1)"],
                yticklabels=["Not Canceled (0)", "Canceled (1)"], ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
    ax.set_ylabel("True Label", fontsize=10, labelpad=8)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(models_dict: Dict[str, Any], X_test: Any, y_test: Any, save_path: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.50)")
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_score:.3f})")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, labelpad=8)
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importances(feature_names: List[str], importances: np.ndarray, model_name: str = "Random Forest",
                             top_n: int = 15, save_path: Optional[str] = None) -> None:
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices][::-1]
    top_scores = importances[indices][::-1]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.barh(range(len(top_names)), top_scores, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel("Relative Importance Score", fontsize=11, labelpad=8)
    ax.set_title(f"Top {top_n} Most Important Features ({model_name})", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tuned_confusion_matrices(
    results_list: List[Dict[str, Any]],
    save_path: str = "results/figures/tuned_confusion_matrices.png"
) -> None:
    """
    Plots a 2x3 grid of confusion matrices for all 6 hyperparameter-tuned classifiers.
    Displays counts, class proportions, and performance metrics.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()
    
    cmap = sns.light_palette("darkgreen", as_cmap=True)
    
    for idx, res in enumerate(results_list):
        ax = axes[idx]
        cm = np.array(res["confusion_matrix"])
        total = np.sum(cm)
        
        labels = [
            [f"TN\n{cm[0,0]:,}\n({cm[0,0]/total*100:.1f}%)", f"FP\n{cm[0,1]:,}\n({cm[0,1]/total*100:.1f}%)"],
            [f"FN\n{cm[1,0]:,}\n({cm[1,0]/total*100:.1f}%)", f"TP\n{cm[1,1]:,}\n({cm[1,1]/total*100:.1f}%)"]
        ]
        
        sns.heatmap(
            cm,
            annot=np.array(labels),
            fmt="",
            cmap=cmap,
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
            linewidths=1.2,
            linecolor="white"
        )
        
        ax.set_title(f"Tuned {res['model']}\n(Acc: {res['accuracy']:.4f} | F1: {res['f1_pos']:.4f} | AUC: {res['roc_auc']:.4f})", 
                     fontsize=11.5, fontweight="bold", pad=8)
        ax.set_xticklabels(["Not Canceled (0)", "Canceled (1)"], fontsize=10)
        ax.set_yticklabels(["Not Canceled (0)", "Canceled (1)"], fontsize=10, rotation=0)
        ax.set_xlabel("Predicted Class", fontsize=10, labelpad=6)
        ax.set_ylabel("True Class", fontsize=10, labelpad=6)
        
    plt.suptitle("Hyperparameter-Tuned Models: Confusion Matrices (Held-out Test Set, N = 23,842)",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved tuned confusion matrices figure: {out}")
    plt.close()


def plot_tuned_roc_curves(
    results_list: List[Dict[str, Any]],
    y_test: np.ndarray,
    save_path: str = "results/figures/tuned_roc_curves.png"
) -> None:
    """
    Plots overlaid Receiver Operating Characteristic (ROC) curves for all 6 tuned models.
    Sorted by ROC-AUC descending with chance diagonal.
    """
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    
    sorted_results = sorted(results_list, key=lambda x: x["roc_auc"], reverse=True)
    
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    linestyles = ["-", "-", "--", "-.", ":", "-"]
    
    y_test_arr = np.asarray(y_test).ravel()
    
    for idx, res in enumerate(sorted_results):
        y_prob = res["y_prob"]
        fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
        score = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=colors[idx % len(colors)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2.2,
            label=f"Tuned {res['model']} (AUC = {score:.4f})"
        )
        
    ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")
    
    ax.set_title("Tuned Classifiers: Receiver Operating Characteristic (ROC) Comparison", 
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, labelpad=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95, facecolor="white")
    
    plt.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved tuned ROC curves figure: {out}")
    plt.close()


def plot_tuning_performance_comparison(
    baseline_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    save_path: str = "results/figures/tuning_performance_comparison.png"
) -> None:
    """
    Plots a multi-panel comparative bar chart contrasting Baseline vs. Tuned performance
    across Accuracy, F1-Score (Positive Class), and ROC-AUC for all 6 models,
    with explicit delta annotations indicating performance improvement or degradation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    models = baseline_df["model"].tolist()
    
    metrics_to_plot = [
        ("accuracy", "Accuracy (Generalization)", axes[0]),
        ("f1_pos", "F1-Score (Positive Class: Cancellations)", axes[1]),
        ("roc_auc", "ROC-AUC (Discriminative Capability)", axes[2])
    ]
    
    x = np.arange(len(models))
    width = 0.36
    
    for metric_col, title, ax in metrics_to_plot:
        base_vals = baseline_df[metric_col].values
        tuned_vals = tuned_df[metric_col].values
        
        rects1 = ax.bar(x - width/2, base_vals, width, label="Baseline", color="#4682b4", edgecolor="black", linewidth=0.6, alpha=0.88)
        rects2 = ax.bar(x + width/2, tuned_vals, width, label="Tuned", color="#2e8b57", edgecolor="black", linewidth=0.6, alpha=0.88)
        
        for i, (b_val, t_val) in enumerate(zip(base_vals, tuned_vals)):
            delta = t_val - b_val
            delta_str = f"{delta:+.3f}" if abs(delta) >= 0.0005 else "0.000"
            delta_color = "#1b5e20" if delta > 0.001 else ("#b71c1c" if delta < -0.001 else "#333333")
            
            ax.annotate(
                f"{t_val:.3f}\n({delta_str})",
                xy=(x[i] + width/2, t_val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=delta_color
            )
            
            ax.annotate(
                f"{b_val:.3f}",
                xy=(x[i] - width/2, b_val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, color="#222222"
            )
            
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9, fontweight="bold")
        ax.set_ylim([0.70, 1.05])
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9.5, frameon=True)
        
    plt.suptitle("Hyperparameter Tuning Impact: Baseline vs. Tuned Model Performance Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved tuning performance comparison figure: {out}")
    plt.close()
