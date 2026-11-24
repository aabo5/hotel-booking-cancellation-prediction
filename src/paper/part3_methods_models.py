"""
part3_methods_models.py - Sections 10 and 11 of the Academic Research Paper.
Covers:
- Section 10: Experimental Methodology & Evaluation Protocol
- Section 11: Machine Learning Algorithm Theoretical Formulations (All 6 Models)
"""

def get_section_10():
    return """## Section 10: Experimental Methodology & Evaluation Protocol

### 10.1 Validation Design: 5-Fold Stratified Cross-Validation
To guarantee reproducible, generalizable statistical inference, all architectural benchmarking and hyperparameter optimization were executed using a strict **5-Fold Stratified Cross-Validation** protocol on the training partition ($N_{\\text{train}} = 95,366$).

In stratified $K$-fold cross-validation, the training dataset $\\mathcal{D}_{\\text{train}}$ is partitioned into $K = 5$ disjoint, mutually exclusive folds $\\{F_1, F_2, \\dots, F_5\\}$ such that the class prevalence in each fold matches the global cancellation rate of $37.076\\%$ exactly:

$$\\frac{|\\{i \\in F_k : y_i = 1\\}|}{|F_k|} = \\frac{|\\{i \\in \\mathcal{D}_{\\text{train}} : y_i = 1\\}|}{|\\mathcal{D}_{\\text{train}}|} \\approx 0.37076, \\quad \\forall k \\in \\{1, \\dots, 5\\}$$

During each cross-validation iteration $k \\in \\{1, \\dots, 5\\}$, fold $F_k$ serves as the validation holdout, while the union of the remaining four folds $\\mathcal{D}_{\\text{train}} \\setminus F_k$ ($N \\approx 76,293$) is utilized for model fitting. Across all folds, the out-of-fold predictions are recorded, and validation metrics are computed as the sample mean $\\hat{\\mu}$ and standard deviation $\\hat{\\sigma}$. 

Once hyperparameter optimization and feature subset selection were concluded, the final model configurations were trained on the complete 95,366-sample training set and evaluated on the completely unseen, untouched held-out test partition ($N_{\\text{test}} = 23,842$).

### 10.2 Comprehensive Evaluation Metrics Suite
Evaluating binary classification models solely through global accuracy is deeply flawed in asymmetric class settings. A naive "dummy" classifier predicting all reservations will arrive ($y = 0$) achieves a 62.92% accuracy while failing to detect a single cancellation. To obtain a multi-dimensional characterization of performance, we implement eight rigorous evaluation metrics:

1. **Classification Accuracy:** Overall proportion of correct predictions:
   $$\\text{Accuracy} = \\frac{\\text{TP} + \\text{TN}}{\\text{TP} + \\text{TN} + \\text{FP} + \\text{FN}}$$

2. **Positive Class Precision:** The fidelity of positive cancellation alarms:
   $$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$$

3. **Positive Class Recall (Sensitivity / True Positive Rate):** The capture rate of actual cancellations:
   $$\\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$$

4. **Positive Class $F_1$-Score:** The harmonic mean balancing precision and recall:
   $$F_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}} = \\frac{2\\text{TP}}{2\\text{TP} + \\text{FP} + \\text{FN}}$$

5. **Macro-Averaged $F_1$-Score:** Unweighted arithmetic mean of $F_1$-scores across both negative and positive classes, penalizing models that sacrifice positive class performance:
   $$F_{1, \\text{macro}} = \\frac{F_{1, \\text{class 0}} + F_{1, \\text{class 1}}}{2}$$

6. **Receiver Operating Characteristic Area Under Curve (ROC-AUC):** Quantifies threshold-independent discrimination across all operational cutoffs:
   $$\\text{ROC-AUC} = \\int_0^1 \\text{TPR}(\\text{FPR}^{-1}(t)) \\, dt = P(\\hat{p}(x^+) > \\hat{p}(x^-))$$
   It measures the probability that a randomly chosen canceled booking receives a higher predicted cancellation score than a randomly chosen completed booking.

7. **Precision-Recall Area Under Curve (PR-AUC):** Computes the average precision across all recall levels via trapezoidal interpolation:
   $$\\text{PR-AUC} = \\sum_{k=1}^m (R_k - R_{k-1}) P_k$$
   In class-skewed regimes (37% prevalence), PR-AUC is far more sensitive to false positive surges than ROC-AUC because the denominator of precision is $\\text{TP} + \\text{FP}$ rather than the large true negative volume $\\text{FP} + \\text{TN}$ (Davis & Goadrich, 2006).

8. **Epistemic Brier Score:** Strictly proper scoring rule quantifying probabilistic calibration error:
   $$\\text{BS} = \\frac{1}{N} \\sum_{i=1}^N (\\hat{p}_i - y_i)^2, \\quad \\text{BS} \\in [0, 1]$$
   Lower scores indicate superior probability calibration.

9. **Computational Complexity & Latency:** Wall-clock training duration (seconds) and per-sample inference latency (microseconds, $\\mu s$).

### 10.3 Computational Environment & Reproducibility Specifications
All experiments were conducted on an enterprise-grade computing environment running 64-bit Windows OS, Python 3.12, Scikit-Learn 1.4+, LightGBM 4.3+, and XGBoost 2.0+. To enforce absolute deterministic reproducibility across all experiments, a global pseudo-random seed was fixed at `random_state = 42`.

---
"""

def get_section_11():
    return """## Section 11: Machine Learning Algorithm Theoretical Formulations (All 6 Models)

To address the diverse inductive biases required to discover non-linear patterns within tabular PMS data, we benchmark six foundational machine learning architectures spanning four distinct learning paradigms:

```
                            MACHINE LEARNING PARADIGM TAXONOMY
                                            │
        ┌───────────────────────────┬───────┴───────────────────┬───────────────────────────┐
        ▼                           ▼                           ▼                           ▼
[Tree-Based Symbolic]       [Ensemble Learning]         [Deep Connectionist]        [Instance-Based]
   • Decision Tree             • Random Forest (Bagging)   • Multi-Layer Perceptron   • K-Nearest Neighbors
     (CART / Entropy)          • XGBoost (Gradient Boost)    (Feedforward Neural Net)   (Metric Learning)
                               • LightGBM (Histogram Boost)
```

### 11.1 Model 1: Decision Tree (Classification and Regression Trees - CART)
Decision Trees recursively partition the $d$-dimensional feature space into axis-aligned hyper-rectangles using greedy heuristic splitting rules (Breiman et al., 1984). Given a node dataset $S$, the algorithm evaluates all available features $j \\in \\{1, \\dots, d\\}$ and candidate split thresholds $t \\in \\mathbb{R}$ to identify the split $s^* = (j^*, t^*)$ that maximizes the reduction in impurity (Information Gain).

Impurity at node $S$ is quantified using Shannon Entropy:
$$H(S) = -\\sum_{k=0}^1 p_k \\log_2(p_k)$$
where $p_k$ is the empirical proportion of class $k$ in node $S$. The Information Gain resulting from a candidate binary split of $S$ into subsets $S_L = \\{\\mathbf{x} \\in S : x_j \\le t\\}$ and $S_R = \\{\\mathbf{x} \\in S : x_j > t\\}$ is given by:

$$IG(S, j, t) = H(S) - \\left( \\frac{|S_L|}{|S|} H(S_L) + \\frac{|S_R|}{|S|} H(S_R) \\right)$$

To combat severe overfitting caused by unconstrained recursive expansion, we implement Cost-Complexity Pruning parameterized by $\\alpha \\ge 0$, which minimizes the penalized cost functional:

$$R_\\alpha(T) = R(T) + \\alpha |\\tilde{T}|$$

where $R(T)$ is the misclassification rate of tree $T$, and $|\\tilde{T}|$ denotes the number of terminal leaf nodes.

### 11.2 Model 2: Random Forest (Bagging Ensemble)
Random Forest (Breiman, 2001) mitigates the high variance and structural instability of individual decision trees by constructing an ensemble of $B$ decorrelated decision trees $\\{T_1, T_2, \\dots, T_B\\}$ trained on independent bootstrap samples drawn with replacement from $\\mathcal{D}_{\\text{train}}$.

The fundamental mathematical mechanism driving Random Forest is variance reduction. For an ensemble of $B$ identically distributed (but correlated) trees, each with variance $\\sigma^2$ and positive pairwise correlation $\\rho = \\text{corr}(T_i, T_j)$, the variance of the ensemble mean prediction $\\bar{T}(\\mathbf{x}) = \\frac{1}{B} \\sum_{b=1}^B T_b(\\mathbf{x})$ is:

$$\\text{Var}(\\bar{T}(\\mathbf{x})) = \\rho \\sigma^2 + \\frac{1 - \\rho}{B} \\sigma^2$$

As $B \\to \\infty$, the second term approaches zero, while the irreducible error is bounded by $\\rho \\sigma^2$. To aggressively minimize $\\rho$, Breiman introduced **Random Subspace Projection**: at each internal node split, the algorithm evaluates only a randomly selected subset of $m = \\lfloor \\sqrt{d} \\rfloor$ features rather than the full feature space $d$. The final ensemble class probability is aggregated via soft-voting:

$$\\hat{p}(\\mathbf{x}) = \\frac{1}{B} \\sum_{b=1}^B P_{T_b}(Y = 1 \\mid \\mathbf{x})$$

### 11.3 Model 3: Extreme Gradient Boosting (XGBoost)
XGBoost (Chen & Guestrin, 2016) is a scalable, regularized gradient boosted decision tree (GBDT) architecture. Rather than training independent trees in parallel, XGBoost builds an additive sequence of $M$ regression trees $f_m(\\mathbf{x})$ in a stagewise forward manner:

$$\\hat{y}_i^{(m)} = \\hat{y}_i^{(m-1)} + f_m(\\mathbf{x}_i)$$

The model minimizes a regularized objective function combining the empirical logistic loss with an explicit tree complexity penalty:

$$\\mathcal{L}^{(m)} = \\sum_{i=1}^N l(y_i, \\hat{y}_i^{(m-1)} + f_m(\\mathbf{x}_i)) + \\Omega(f_m), \\quad \\Omega(f) = \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^T w_j^2$$

where $T$ is the number of terminal leaves, $w_j$ is the weight of leaf $j$, $\\gamma$ penalizes leaf proliferation, and $\\lambda$ is an $L_2$ shrinkage parameter. Using a second-order Taylor expansion of the loss function around the previous iteration $\\hat{y}_i^{(m-1)}$:

$$\\mathcal{L}^{(m)} \\approx \\sum_{i=1}^N \\left[ l(y_i, \\hat{y}_i^{(m-1)}) + g_i f_m(\\mathbf{x}_i) + \\frac{1}{2} h_i f_m^2(\\mathbf{x}_i) \\right] + \\Omega(f_m)$$

where $g_i = \\frac{\\partial l(y_i, \\hat{y}_i^{(m-1)})}{\\partial \\hat{y}_i^{(m-1)}}$ and $h_i = \\frac{\\partial^2 l(y_i, \\hat{y}_i^{(m-1)})}{\\partial (\\hat{y}_i^{(m-1)})^2}$ represent the first- and second-order gradients. 

For a given tree structure with instance sets $I_j = \\{i : q(\\mathbf{x}_i) = j\\}$ allocated to leaf $j$, the optimal leaf weight $w_j^*$ is derived analytically as:

$$w_j^* = -\\frac{\\sum_{i \\in I_j} g_i}{\\sum_{i \\in I_j} h_i + \\lambda} = -\\frac{G_j}{H_j + \lambda}$$

The resulting optimal objective value is $\\mathcal{L}^*(q) = -\\frac{1}{2} \\sum_{j=1}^T \\frac{G_j^2}{H_j + \\lambda} + \\gamma T$. The gain of a candidate split into left ($L$) and right ($R$) branches is evaluated as:

$$\\mathcal{L}_{\\text{split}} = \\frac{1}{2} \\left[ \\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L + G_R)^2}{H_L + H_R + \\lambda} \\right] - \\gamma$$

### 11.4 Model 4: Light Gradient Boosting Machine (LightGBM)
LightGBM (Ke et al., 2017) introduces three fundamental architectural enhancements over traditional gradient boosting:

1. **Gradient-Based One-Side Sampling (GOSS):** Instances with small gradients (near convergence) contribute minimally to information gain. GOSS ranks instances by the absolute magnitude of their gradients $|g_i|$, retains the top $a \\times 100\\%$ instances with largest gradients, and randomly samples a fraction $b \\times 100\\%$ from the remaining small-gradient instances. To preserve the original data distribution, small-gradient instances are assigned an amplified weight factor of $\\frac{1 - a}{b}$ when computing information gain:
   $$\\tilde{G}_j = \\sum_{i \\in A \\cap I_j} g_i + \\frac{1 - a}{b} \\sum_{i \\in B \\cap I_j} g_i$$

2. **Exclusive Feature Bundling (EFB):** High-dimensional sparse categorical feature spaces (e.g., 231 one-hot encoded country dummies) contain many mutually exclusive features that rarely take non-zero values simultaneously. EFB bundles these exclusive features into dense composite feature bins, reducing histogram construction complexity from $O(\\text{data} \\times \\text{features})$ to $O(\\text{data} \\times \\text{bundles})$.

3. **Leaf-Wise (Best-First) Tree Growth:** Rather than expanding trees depth-wise (level-by-level) as in standard CART and XGBoost, LightGBM expands leaf-wise, greedily splitting the specific leaf node that produces the largest reduction in global loss:

```
    Level-Wise (Depth-Wise) Growth           Leaf-Wise (Best-First) Growth
              [Node]                                    [Node]
             /      \                                  /      \
        [Node]      [Node]                        [Node]      [Node]* (Max Gain)
       /     \     /     \                                   /      \
      O       O   O       O                             [Leaf]      [Leaf]
  (Symmetric, high overhead)                  (Asymmetric, maximizes loss drop)
```

To prevent deep leaf-wise trees from overfitting on small data partitions, LightGBM bounds leaf-wise growth via explicit `num_leaves` and `max_depth` regularization constraints.

### 11.5 Model 5: Multi-Layer Perceptron (MLP)
The Multi-Layer Perceptron (Rumelhart, Hinton, & Williams, 1986) represents a deep feedforward artificial neural network mapping continuous inputs through non-linear geometric latent spaces. We implement a multi-layer architecture parameterized by weight matrices $\\mathbf{W}^{[l]}$ and bias vectors $\\mathbf{b}^{[l]}$ across $L$ layers:

$$\\mathbf{z}^{[l]} = \\mathbf{W}^{[l]} \\mathbf{a}^{[l-1]} + \\mathbf{b}^{[l]}, \\quad \\mathbf{a}^{[l]} = g^{[l]}(\\mathbf{z}^{[l]})$$

where $\\mathbf{a}^{[0]} = \\mathbf{x} \\in \\mathbb{R}^{257}$ represents the normalized input vector. We implement the hyperbolic tangent ($\tanh$) non-linear activation function in hidden layers:
$$g(z) = \\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$$
and the logistic sigmoid function at the output layer $L$ to generate a valid posterior probability:
$$\\hat{p} = \\sigma(z^{[L]}) = \\frac{1}{1 + e^{-z^{[L]}}}$$

The network is trained end-to-end by minimizing the penalized Binary Cross-Entropy loss functional with $L_2$ weight regularization:

$$\\mathcal{J}(\\mathbf{W}, \\mathbf{b}) = -\\frac{1}{N} \\sum_{i=1}^N \\left[ y_i \\log \\hat{p}_i + (1 - y_i) \\log(1 - \\hat{p}_i) \\right] + \\frac{\\alpha}{2N} \\sum_{l=1}^L \\|\\mathbf{W}^{[l]}\\|_F^2$$

where $\\|\\mathbf{W}^{[l]}\\|_F = \\sqrt{\\sum_j \\sum_k (w_{jk}^{[l]})^2}$ denotes the Frobenius norm, and $\\alpha$ is the weight decay regularization parameter. Optimization is executed using the Adam adaptive moment estimation optimizer (Kingma & Ba, 2014) with backpropagation.

### 11.6 Model 6: K-Nearest Neighbors (KNN)
K-Nearest Neighbors (Cover & Hart, 1967) is a non-parametric, memory-based metric learning algorithm that makes no structural assumptions regarding the underlying data distribution. Given a query point $\\mathbf{x} \\in \\mathbb{R}^d$, the algorithm identifies the set $\\mathcal{N}_k(\\mathbf{x}) = \\{\\mathbf{x}_{(1)}, \\dots, \\mathbf{x}_{(k)}\\}$ of the $k$ closest training vectors under a chosen distance metric $D(\\mathbf{x}, \\mathbf{x}')$.

We evaluate both $L_2$ Euclidean distance and $L_1$ Manhattan distance (Minkowski $p=1$):
$$D_{\\text{Manhattan}}(\\mathbf{x}, \\mathbf{x}') = \\sum_{j=1}^d |x_j - x_j'|$$

In high-dimensional spaces ($d = 257$), standard uniform majority voting degrades rapidly due to the **curse of dimensionality**, wherein distances between points become approximately equidistant (Beyer et al., 1999). To counteract metric dispersion, we implement **Distance-Weighted Voting**, where each neighbor $\\mathbf{x}_{(i)}$ is assigned a weighting factor inversely proportional to its metric distance:

$$w_i = \\frac{1}{D(\\mathbf{x}, \\mathbf{x}_{(i)}) + \\epsilon}$$

The posterior probability of cancellation is then formulated as the normalized distance-weighted sum:

$$\\hat{p}(\\mathbf{x}) = \\frac{\\sum_{i \\in \\mathcal{N}_k(\\mathbf{x})} w_i y_i}{\\sum_{i \\in \\mathcal{N}_k(\\mathbf{x})} w_i}$$

---
"""
