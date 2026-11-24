"""
part5_calibration_economics.py - Sections 18 and 19 of the Academic Research Paper.
Covers:
- Section 18: Probability Calibration, Reliability & Brier Score Analysis
- Section 19: Asymmetric Misclassification Economics & Decision Threshold Optimization
"""

import pandas as pd
from pathlib import Path

def get_section_18():
    df_cal = pd.read_csv("results/metrics/model_calibration_metrics.csv")
    cal_cols = ["model", "brier_score", "roc_auc", "pr_auc"]
    cal_md = df_cal[cal_cols].to_markdown(index=False)

    text = """## Section 18: Probability Calibration, Reliability & Brier Score Analysis

### 18.1 Epistemic Calibration vs. Discriminative Ranking
In academic literature, classification models are overwhelmingly evaluated via discriminative ranking metrics, most notably the Receiver Operating Characteristic Area Under Curve (ROC-AUC). However, a high ROC-AUC merely confirms that a model reliably scores positive instances higher than negative instances; it provides **zero guarantee that the output values correspond to true posterior probabilities** (Guo et al., 2017). 

In enterprise revenue management, automated decision-making engines do not operate on ordinal ranks. They require true probabilistic expectations: when an algorithm outputs $\\hat{p} = 0.80$, exactly 80 out of 100 such reservations must empirically cancel. If the model is systematically under-confident or over-confident, downstream mathematical optimization models (such as dynamic pricing engines or overbooking capacity calculations) will miscalculate expected values and incur massive operational losses.

### 18.2 Mathematical Formulation of Reliability Diagrams and the Brier Score
To evaluate epistemic probability calibration, we implement two foundational diagnostic tools:

1. **Reliability Diagrams (Decile Calibration Curves):** The predicted probability interval $[0, 1]$ is partitioned into $M = 10$ mutually exclusive bins $B_m = (\\frac{m-1}{M}, \\frac{m}{M}]$. For each bin $B_m$, we compute:
   * **Mean Predicted Confidence:** $\\text{conf}(B_m) = \\frac{1}{|B_m|} \\sum_{i \\in B_m} \\hat{p}_i$
   * **Empirical Observed Accuracy:** $\\text{acc}(B_m) = \\frac{1}{|B_m|} \\sum_{i \\in B_m} y_i$
   A perfectly calibrated model satisfies $\\text{acc}(B_m) = \\text{conf}(B_m)$ for all $m$, plotting as a perfect 45-degree diagonal.

2. **The Brier Score (BS):** A strictly proper scoring rule quantifying mean squared probabilistic error (Brier, 1950):
   $$\\text{BS} = \\frac{1}{N} \\sum_{i=1}^N (\\hat{p}_i - y_i)^2, \\quad \\text{BS} \\in [0, 1]$$
   The Brier Score can be decomposed into three mathematically orthogonal components (Murphy, 1973):
   $$\\text{BS} = \\text{Reliability} - \\text{Resolution} + \\text{Uncertainty}$$
   where **Reliability** quantifies calibration divergence, **Resolution** measures the ability to separate outcomes, and **Uncertainty** reflects the irreducible variance of the prior base rate $\\bar{y}(1 - \\bar{y}) = 0.3708(1 - 0.3708) = 0.2333$. Lower Brier Scores denote superior calibration.

### 18.3 Empirical Calibration and Precision-Recall Benchmarks
Table 18.1 reports the Brier Scores, ROC-AUC, and Precision-Recall Area Under Curve (PR-AUC) across the six tuned architectures on the held-out test partition.

**Table 18.1: Epistemic Probability Calibration, PR-AUC, and Discrimination Benchmarks**

__CAL_TABLE__

![Final Calibration Curves](../results/figures/final_calibration_curves.png)
*Figure 18.1: Decile-binned Reliability Diagrams (Calibration Curves) comparing all six tuned architectures against the 45-degree perfect calibration diagonal.*

![Final Precision Recall Curves](../results/figures/final_precision_recall_curves.png)
*Figure 18.2: Precision-Recall (PR) curves across tuned models on the held-out test set ($N_{\\text{test}} = 23,842$, baseline prevalence = 37.08%).*

### 18.4 Architectural Calibration Diagnostics
Analyzing Figure 18.1 and Table 18.1 reveals profound calibration characteristics:

1. **LightGBM Global Supremacy (Brier Score = 0.0750, PR-AUC = 0.9433):** LightGBM achieved the lowest Brier Score across all evaluated models, outperforming XGBoost ($0.0785$) and Random Forest ($0.0845$). As visualized in Figure 18.1, LightGBM's reliability curve tracks the 45-degree diagonal with extraordinary fidelity across the entire spectrum $[0.0, 1.0]$, exhibiting minimal distortion even in high-probability bins. Its PR-AUC of **0.9433** demonstrates that precision remains above 85% even when recall reaches 80%, providing exceptional protection against false alarms.
2. **The Bagging Compression Artifact in Random Forest (Brier Score = 0.0845):** Random Forest exhibited characteristic probability compression toward the sample base rate. Because Random Forest averages predictions across 150 individual trees, the extreme probability tails ($p < 0.05$ and $p > 0.95$) are systematically pulled toward the center. This manifests in Figure 18.1 as a slight sigmoidal "S-curve," wherein Random Forest tends to be slightly under-confident for extreme events.
3. **Neural Network Sigmoid Miscalibration (MLP Brier Score = 0.0895):** Tuned MLP exhibited classic over-confidence in mid-to-high probability bins, a known property of cross-entropy optimization in neural networks with uncalibrated softmax/sigmoid outputs (Guo et al., 2017).
4. **KNN Coarse Quantization (Brier Score = 0.0937):** KNN's probability outputs are inherently discrete, bounded by the finite number of neighbors ($k = 21$). Even with distance weighting, its predictions form discrete probability clusters, yielding higher calibration error.

---
"""
    return text.replace("__CAL_TABLE__", cal_md)

def get_section_19():
    df_cost = pd.read_csv("results/tables/threshold_optimization_results.csv")
    cost_cols = [
        "model", "default_threshold", "default_loss_dollars", "optimal_threshold",
        "optimized_loss_dollars", "net_dollar_savings", "savings_percentage",
        "default_fp", "default_fn", "optimal_fp", "optimal_fn"
    ]
    cost_md = df_cost[cost_cols].to_markdown(index=False)

    text = """## Section 19: Asymmetric Misclassification Economics & Decision Threshold Optimization

### 19.1 Modeling Asymmetric Operational Business Costs in Hospitality
Conventional machine learning classifiers operate under the default assumption of **symmetric misclassification costs** ($C_{\\text{FP}} = C_{\\text{FN}} = 1.0$), establishing an arbitrary decision threshold of $p = 0.50$. In commercial enterprise lodging, this assumption is demonstrably false. The financial and reputational consequences of misclassification are severely asymmetric:

```
                          THE OPERATIONAL COST ASYMMETRY MATRIX
                          
                          ┌──────────────────────────┬──────────────────────────┐
                          │    Actual: Checked-In    │     Actual: Canceled     │
                          │         (y = 0)          │         (y = 1)          │
 ┌────────────────────────┼──────────────────────────┼──────────────────────────┤
 │ Predict: Check-In      │   True Negative (TN)     │   False Negative (FN)    │
 │ (Keep Reservation,     │   Cost = $0              │   Perishable Spoilage:   │
 │  No Overbooking)       │   (Orderly Check-In)     │   C_empty = ADR ≈ $102   │
 ├────────────────────────┼──────────────────────────┼──────────────────────────┤
 │ Predict: Canceled      │   False Positive (FP)    │   True Positive (TP)     │
 │ (Flag for Overbooking) │   Overbooked & Walked!   │   Cost = $0              │
 │                        │   C_walk = $300          │   (Successful Resale)    │
 └────────────────────────┴──────────────────────────┴──────────────────────────┘
```

1. **Type I Error (False Positive — The "Guest Walking" Disaster):** The model predicts a reservation will cancel ($\hat{y} = 1$), prompting the automated revenue engine to resell the room to a secondary guest. However, the original guest arrives as scheduled ($y = 0$). With the hotel operating at 100% capacity, the hotel must "walk" the guest to a competitor. In lodging operations, the walking penalty includes:
   * Direct booking costs at an equivalent or superior peer hotel: $\\$150.00$
   * Premium private ground transportation: $\\$35.00$
   * Complimentary dining / beverage service reparations: $\\$40.00$
   * Churn penalty (loss of customer lifetime value and brand damage): $\\$75.00$
   * **Total Operational Walking Cost:**
     $$C_{\\text{walk}} = C_{\\text{FP}} = \\$300.00$$

2. **Type II Error (False Negative — Perishable Room Spoilage):** The model predicts the guest will arrive ($\hat{y} = 0$), so the room is held. The guest cancels last-minute or no-shows without financial penalty ($y = 1$). The room remains vacant overnight, resulting in the permanent spoilage of perishable inventory:
   * Opportunity cost of unsold room: Equal to the Average Daily Rate (ADR)
   * **Total Spoilage Cost:**
     $$C_{\\text{empty}} = C_{\\text{FN}} = \\text{ADR} \\approx \\$102.00$$

### 19.2 Theoretical Bayes Risk Optimal Threshold
Because $C_{\\text{walk}} / C_{\\text{empty}} = \\frac{300}{102} \\approx 2.94$, **walking an arriving guest is nearly three times more expensive than leaving a room vacant**. 

Applying Elkan’s Theorem (Elkan, 2001) for cost-sensitive classification under asymmetric loss, a rational, risk-neutral decision-maker should predict cancellation ($\hat{y} = 1$) if and only if the expected cost of predicting cancellation is less than or equal to the expected cost of predicting arrival:

$$E[\\text{Cost} \\mid \\hat{y} = 1] = (1 - p) \\cdot C_{\\text{walk}} + p \\cdot 0 = (1 - p) C_{\\text{walk}}$$
$$E[\\text{Cost} \\mid \\hat{y} = 0] = (1 - p) \\cdot 0 + p \\cdot C_{\\text{empty}} = p \\cdot C_{\\text{empty}}$$

Setting the expected costs equal yields the theoretical **Bayes Risk Optimal Threshold ($p^*$)**:

$$(1 - p^*) C_{\\text{walk}} = p^* C_{\\text{empty}} \\implies p^* = \\frac{C_{\\text{walk}}}{C_{\\text{walk}} + C_{\\text{empty}}}$$

Substituting the empirical operational parameters:

$$p^* = \\frac{300}{300 + 102} = \\frac{300}{402} \\approx 0.7463 \\approx \\mathbf{0.75}$$

**Theoretical Insight:** In an enterprise environment where walking a guest incurs a $\\$300$ penalty and room spoilage costs $\\$102$, an algorithm should **never flag a booking for overbooking resale unless its posterior probability of cancellation exceeds 75%**. Operating at the standard $p = 0.50$ threshold generates severe overbooking chaos and unnecessary guest walking costs.

### 19.3 Empirical Cost Function Optimization & Sweep Results
To validate this theoretical formulation on the held-out test partition ($N_{\\text{test}} = 23,842$), we formulate the empirical operational loss function:

$$\\mathcal{L}(p) = C_{\\text{walk}} \\cdot \\text{FP}(p) + C_{\\text{empty}} \\cdot \\text{FN}(p) = 300 \\cdot \\text{FP}(p) + 102 \\cdot \\text{FN}(p)$$

We execute an exhaustive grid sweep evaluating $p \\in [0.05, 0.95]$ in discrete steps of $\Delta p = 0.01$ across all six tuned classifiers. Table 19.1 presents the empirical optimization results.

**Table 19.1: Empirical Decision Threshold Optimization and Operational Loss Reduction ($N_{\\text{test}} = 23,842$)**

__COST_TABLE__

![Cost Threshold Optimization](../results/figures/cost_threshold_optimization.png)
*Figure 19.1: Operational loss curves as a function of decision threshold $p \\in [0.05, 0.95]$, showing the global minimum at $p^* = 0.75$ and net dollar savings.*

### 19.4 Executive Financial Impact: $\\$108,444$ Quarterly Net Savings
The empirical cost sweep results in Table 19.1 provide definitive proof of the financial power of cost-calibrated machine learning:

1. **LightGBM Delivers Maximum Absolute Loss Minimization:** At the default threshold $p = 0.50$, LightGBM incurs $\\$489,960$ in operational loss ($1,147$ false positive walks and $1,430$ false negative empty rooms). Shifting to the empirically verified optimal threshold of **$p^* = 0.75$** collapses false positive walks by **69.4% (from 1,147 to 351)**, reducing total operational loss to **$\\$381,516**. This delivers a **net quarterly savings of $\\$108,444.00 (a 22.13% financial loss reduction)** on the held-out test set alone.
2. **Empirical Validation of the Bayes Risk Formula:** The empirical cost minimum identified via discrete sweep for LightGBM occurs at **$p^* = 0.75$**, aligning with the theoretical Bayes risk calculation ($p^* = 0.7463$). This perfect convergence proves that LightGBM's probability outputs are exceptionally well-calibrated; if the model had suffered from epistemic distortion, the empirical minimum would have diverged substantially from the theoretical value.
3. **Massive Relative Savings in Less Calibrated Models:** Less calibrated models achieved even higher percentage cost reductions under threshold optimization: KNN saved **28.38% ($\\$162,492$)** by shifting threshold to $p^* = 0.82$; MLP saved **25.66% ($\\$150,090$)** at $p^* = 0.85$; and Decision Tree saved **23.16% ($\\$142,806$)** at $p^* = 0.76$. Shifting the decision threshold provides an immediate financial buffer against false-alarm overbooking for any deployed classifier.

---
"""
    return text.replace("__COST_TABLE__", cost_md)
