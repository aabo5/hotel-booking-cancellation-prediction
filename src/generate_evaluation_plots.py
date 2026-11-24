import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, average_precision_score
)

def run():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10

    X_test = pd.read_parquet('data/processed/X_test.parquet')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

    models_info = [
        ('knn', '03_knn', 'K-Nearest Neighbors (KNN)'),
        ('decision_tree', '04_decision_tree', 'Decision Tree'),
        ('random_forest', '05_random_forest', 'Random Forest'),
        ('xgboost', '06_xgboost', 'XGBoost'),
        ('lightgbm', '07_lightgbm', 'LightGBM'),
        ('mlp', '08_mlp', 'MLP Neural Network')
    ]

    all_tuned_probas = {}

    for key, folder, title in models_info:
        results_dir = f'results/{folder}'
        os.makedirs(results_dir, exist_ok=True)
        
        m_def = joblib.load(f'models/{key}_default.joblib')
        m_tune = joblib.load(f'models/{key}_tuned.joblib')
        
        y_pred_def = m_def.predict(X_test)
        y_prob_def = m_def.predict_proba(X_test)[:, 1]
        
        y_pred_tune = m_tune.predict(X_test)
        y_prob_tune = m_tune.predict_proba(X_test)[:, 1]
        all_tuned_probas[title] = (y_prob_tune, y_pred_tune)
        
        # Calculate evaluation metrics
        tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, y_pred_def).ravel()
        spec_d = tn_d / (tn_d + fp_d)
        
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_tune).ravel()
        spec_t = tn_t / (tn_t + fp_t)
        
        acc_d, acc_t = accuracy_score(y_test, y_pred_def), accuracy_score(y_test, y_pred_tune)
        prec_d, prec_t = precision_score(y_test, y_pred_def), precision_score(y_test, y_pred_tune)
        rec_d, rec_t = recall_score(y_test, y_pred_def), recall_score(y_test, y_pred_tune)
        f1_d, f1_t = f1_score(y_test, y_pred_def), f1_score(y_test, y_pred_tune)
        auc_d, auc_t = roc_auc_score(y_test, y_prob_def), roc_auc_score(y_test, y_prob_tune)
        
        comp_df = pd.DataFrame({
            'Metric': ['Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Specificity'],
            'Default (No Tuning)': [acc_d, auc_d, f1_d, prec_d, rec_d, spec_d],
            'Tuned': [acc_t, auc_t, f1_t, prec_t, rec_t, spec_t]
        })
        comp_df['Improvement'] = comp_df['Tuned'] - comp_df['Default (No Tuning)']
        comp_df.to_csv(f'{results_dir}/metrics_comparison.csv', index=False)
        
        # 1. Bar Chart: All evaluation metrics
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(comp_df))
        width = 0.35
        rects1 = ax.bar(x - width/2, comp_df['Default (No Tuning)'], width, label='Default (No Tuning)', color='#3498db', alpha=0.85)
        rects2 = ax.bar(x + width/2, comp_df['Tuned'], width, label='Tuned', color='#2ecc71', alpha=0.85)
        
        ax.set_title(f'{title}: Evaluation Metrics Comparison', fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel('Score (0.0 to 1.0)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df['Metric'], fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.legend(loc='lower right', frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in rects1:
            h = bar.get_height()
            ax.annotate(f'{h*100:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8.5)
        for bar in rects2:
            h = bar.get_height()
            ax.annotate(f'{h*100:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/metrics_comparison.png', dpi=300)
        plt.savefig(f'{results_dir}/comparison_barchart.png', dpi=300)
        plt.close()
        
        # 2. Side-by-side Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_def, ax=axes[0], cmap='Blues', colorbar=False)
        axes[0].set_title(f'Default (Accuracy: {acc_d*100:.1f}%)', fontsize=11, fontweight='bold')
        axes[0].grid(False)
        
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tune, ax=axes[1], cmap='Greens', colorbar=False)
        axes[1].set_title(f'Tuned (Accuracy: {acc_t*100:.1f}%)', fontsize=11, fontweight='bold')
        axes[1].grid(False)
        
        plt.suptitle(f'{title}: Confusion Matrices (Default vs. Tuned)', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also ensure individual confusion matrices exist
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_def, ax=ax, cmap='Blues')
        ax.set_title(f'{title} - Default Confusion Matrix', fontsize=10, fontweight='bold')
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/confusion_matrix_default.png', dpi=300)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tune, ax=ax, cmap='Greens')
        ax.set_title(f'{title} - Tuned Confusion Matrix', fontsize=10, fontweight='bold')
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/confusion_matrix_tuned.png', dpi=300)
        plt.close()
        
        # 3. ROC Curve
        fpr_d, tpr_d, _ = roc_curve(y_test, y_prob_def)
        fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_tune)
        
        plt.figure(figsize=(6.5, 5))
        plt.plot(fpr_d, tpr_d, color='#3498db', lw=2, label=f'Default (AUC = {auc_d:.4f})')
        plt.plot(fpr_t, tpr_t, color='#2ecc71', lw=2.2, label=f'Tuned (AUC = {auc_t:.4f})')
        plt.plot([0, 1], [0, 1], color='#7f8c8d', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.50)')
        plt.title(f'{title}: ROC Curve (Default vs. Tuned)', fontsize=12, fontweight='bold')
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
        plt.ylabel('True Positive Rate (Recall)', fontsize=10)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.legend(loc='lower right', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/roc_curve.png', dpi=300)
        plt.close()
        
        # 4. Precision-Recall Curve
        prec_curve_d, rec_curve_d, _ = precision_recall_curve(y_test, y_prob_def)
        prec_curve_t, rec_curve_t, _ = precision_recall_curve(y_test, y_prob_tune)
        ap_d = average_precision_score(y_test, y_prob_def)
        ap_t = average_precision_score(y_test, y_prob_tune)
        prevalence = y_test.mean()
        
        plt.figure(figsize=(6.5, 5))
        plt.plot(rec_curve_d, prec_curve_d, color='#3498db', lw=2, label=f'Default (AP = {ap_d:.4f})')
        plt.plot(rec_curve_t, prec_curve_t, color='#2ecc71', lw=2.2, label=f'Tuned (AP = {ap_t:.4f})')
        plt.axhline(prevalence, color='#e74c3c', linestyle='--', lw=1.5, label=f'Baseline Class Balance ({prevalence*100:.1f}%)')
        plt.title(f'{title}: Precision-Recall Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Recall', fontsize=10)
        plt.ylabel('Precision', fontsize=10)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.legend(loc='lower left', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/precision_recall_curve.png', dpi=300)
        plt.close()
        
        print(f'Successfully generated all plots for {title} in {results_dir}')

    # Combined PR Curve for 10_model_comparison
    res_comp_dir = 'results/10_model_comparison'
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for (title, (y_prob, _)), color in zip(all_tuned_probas.items(), colors):
        p_c, r_c, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(r_c, p_c, label=f'{title} (AP = {ap:.3f})', color=color, lw=2)
    plt.axhline(y_test.mean(), color='black', linestyle='--', label=f'Baseline ({y_test.mean()*100:.1f}%)')
    plt.title('All 6 Models: Precision-Recall Curves (Test Set)', fontsize=13, fontweight='bold')
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.legend(loc='lower left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{res_comp_dir}/combined_pr_curves.png', dpi=300)
    plt.close()
    print('Generated combined PR curves in results/10_model_comparison/!')

if __name__ == '__main__':
    run()
