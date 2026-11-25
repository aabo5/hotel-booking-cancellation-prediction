"""Adds a 'why this algorithm' cell and a 'Discussion' cell (numbers read from results/) to notebooks 03-08.
The code cells and their outputs are not touched. Safe to run again (old added cells are replaced).
Usage: python tools/add_model_discussions.py
"""
import ast
import json
import re
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAG = "added-discussion"

WHY = {
    "03_knn": ("**Why KNN for this problem?** KNN predicts a booking from the outcome of the most similar past bookings "
               "(Euclidean distance on the standardised features). It makes no assumption about the shape of the decision "
               "boundary, which suits our non-linear problem, and it is the simplest baseline of the project. Its weak points "
               "on this data are the 255 mostly binary dimensions (curse of dimensionality) and slow prediction, because "
               "every test booking is compared with all 95,366 training bookings."),
    "04_decision_tree": ("**Why a Decision Tree for this problem?** A tree learns if-then rules such as *deposit = Non Refund -> cancel* "
                         "or *lead time > 180 and no special request -> cancel*, which matches the patterns found in the EDA. It needs "
                         "no scaling, is robust to the outliers we kept and is easy to explain to hotel managers. Its weakness is high "
                         "variance: an unpruned tree memorises the training data, so depth and leaf size must be tuned."),
    "05_random_forest": ("**Why Random Forest for this problem?** Random Forest averages many decorrelated trees (bagging + random feature "
                         "subsets). It keeps the strengths of trees (non-linear rules, robustness to outliers and skew) while removing "
                         "most of their variance, and it gives feature importances. The price is a large model and slower prediction."),
    "06_xgboost": ("**Why XGBoost for this problem?** Gradient boosting adds trees one after another, each correcting the errors of the "
                   "previous ones, with L1/L2 regularisation and shrinkage. It is one of the strongest methods for tabular data with "
                   "mixed numerical and categorical features like ours; it has many hyper-parameters and trains sequentially."),
    "07_lightgbm": ("**Why LightGBM for this problem?** LightGBM is gradient boosting with histogram-based splits and leaf-wise growth. It "
                    "is very fast on large data with many sparse one-hot columns (our 95k x 255 matrix) and captures non-linear "
                    "interactions. Leaf-wise growth can overfit small datasets, which is not a problem with 95k training rows."),
    "08_mlp": ("**Why an MLP for this problem?** A multilayer perceptron can learn any non-linear function of the standardised inputs "
               "and represents the neural-network family in our comparison. It needs scaled inputs (done in notebook 02) and tuning, "
               "is a black box, and on tabular data it usually does not beat tree ensembles.\n\n"
               "*Note:* as 'default' (Part 1) the team used a baseline MLP with two hidden layers (64, 32), ReLU, Adam and early "
               "stopping instead of the scikit-learn default of one layer with 100 neurons."),
}

NAMES = {"03_knn": "KNN", "04_decision_tree": "Decision Tree", "05_random_forest": "Random Forest",
         "06_xgboost": "XGBoost", "07_lightgbm": "LightGBM", "08_mlp": "MLP Neural Net"}

EXPLAIN = {
    "03_knn": "Using 15 neighbours instead of 5 and weighting them by distance makes the vote less noisy: single, accidentally close bookings in the 255-dimensional space have less influence. KNN improves on every metric, but it remains the weakest model of the project because the many binary one-hot dimensions make distances less meaningful.",
    "04_decision_tree": "The default tree is grown until every leaf is pure, so it memorises the training data and its probabilities are only 0 or 1 - this is why its ROC-AUC is low. Limiting the depth and the leaf size gives smoother probabilities and a much better ranking (ROC-AUC). The F1 at threshold 0.5 drops slightly because the pruned tree is more conservative (higher precision, lower recall); the search optimised ROC-AUC, not F1.",
    "05_random_forest": "Random Forest is already very strong with default settings; the grid search only moved from 100 to 150 trees, which changes the results very little. It has the highest precision and specificity of all models, i.e. the fewest false alarms.",
    "06_xgboost": "XGBoost's defaults are already close to the optimum of the small grid, so tuning changed accuracy, F1 and ROC-AUC by less than 0.001 - within the noise level. It ranks third among the six tuned models.",
    "07_lightgbm": "LightGBM profits clearly from tuning: more leaves (127) and a higher learning rate make the model more expressive, which the 95k training rows support without overfitting. The tuned LightGBM has the best F1-score, ROC-AUC and recall of the whole project.",
    "08_mlp": "The selected stronger L2 penalty (alpha = 0.001) slightly changes the balance between precision and recall (precision down, recall up); accuracy and ROC-AUC stay almost the same. The MLP stays behind the tree ensembles, as is usual on tabular data.",
}


def best_params(nb):
    for c in nb.cells:
        for o in c.get("outputs", []):
            m = re.search(r"Best Parameters: (\{.*\})", "".join(o.get("text", "")))
            if m:
                return m.group(1)
    return "?"


lb = pd.read_csv(ROOT / "results/10_model_comparison/master_model_leaderboard.csv").set_index("Model")
for key, name in NAMES.items():
    path = ROOT / "notebooks" / f"{key}.ipynb"
    nb = nbformat.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if TAG not in c.get("metadata", {}).get("tags", [])]
    m = pd.read_csv(ROOT / "results" / key / "metrics_comparison.csv").set_index("Metric")
    rank = int(lb.index.get_loc(name)) + 1
    rows = "\n".join(f"| {k} | {m.loc[k, 'Default (No Tuning)']:.4f} | {m.loc[k, 'Tuned']:.4f} | {m.loc[k, 'Tuned'] - m.loc[k, 'Default (No Tuning)']:+.4f} |"
                     for k in m.index)
    disc = (f"## Discussion\n**Best parameters found by GridSearchCV (3-fold CV, ROC-AUC):** `{best_params(nb)}`\n\n"
            f"| Metric | Default | Tuned | Change |\n|---|---|---|---|\n{rows}\n\n"
            f"{EXPLAIN[key]}\n\n"
            f"**Rank among the six tuned models (by F1-score, notebook 10): {rank} of 6.**")
    why = nbformat.v4.new_markdown_cell(WHY[key])
    end = nbformat.v4.new_markdown_cell(disc)
    for c in (why, end):
        c.metadata["tags"] = [TAG]
    nb.cells.insert(1, why)
    nb.cells.append(end)
    if key == "08_mlp":
        for c in nb.cells:
            if c.cell_type == "markdown" and c.source.startswith("## Part 1"):
                c.source = ("## Part 1: Baseline Model (before tuning)\n"
                            "We train a baseline MLP with two hidden layers (64, 32) and early stopping (see the note above).")
    nbformat.write(nb, path)
    print("updated", key, "rank", rank)
