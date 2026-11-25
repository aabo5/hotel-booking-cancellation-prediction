"""Result-dependent sentences shared by make_report.py and make_slides.py (numbers read from results/)."""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
both = pd.read_csv(R / "10_model_comparison/default_vs_tuned_all_models.csv")
lb = pd.read_csv(R / "10_model_comparison/master_model_leaderboard.csv").set_index("Model")
cms = pd.read_csv(R / "10_model_comparison/confusion_matrices_all.csv").set_index("Model")
gain = pd.read_csv(R / "10_model_comparison/best_model_feature_importance_gain.csv", index_col=0).iloc[:, 0]
fsr = pd.read_csv(R / "09_feature_selection_and_pca/feature_selection_results.csv")
pcr = pd.read_csv(R / "09_feature_selection_and_pca/pca_results.csv")
pcn = pd.read_csv(R / "09_feature_selection_and_pca/pca_components_needed.csv", index_col=0).iloc[:, 0]
curve = pd.read_csv(R / "09_feature_selection_and_pca/n_features_curve_lightgbm.csv")
rank = pd.read_csv(R / "09_feature_selection_and_pca/feature_rankings_all_methods.csv", index_col=0)

d = both[both.Variant == "Default"].set_index("Model").drop(columns="Variant")
t = both[both.Variant == "Tuned"].set_index("Model").drop(columns="Variant")
D = t - d.reindex(t.index)[t.columns]
f4 = lambda x: f"{x:.4f}"
pc = lambda x, k=2: f"{x * 100:.{k}f}%"

# ---------- feature selection
F1 = fsr.pivot_table(index="Setting", columns="Model", values="F1-Score", sort=False)
AUC = fsr.pivot_table(index="Setting", columns="Model", values="ROC-AUC", sort=False)
ACC = fsr.pivot_table(index="Setting", columns="Model", values="Accuracy", sort=False)
TT = fsr.pivot_table(index="Setting", columns="Model", values="Train time (s)", sort=False)
base = F1.loc["All features"]
delta = F1.drop("All features") - base
best_cell = delta.stack().idxmax()           # (setting, model)
best_gain = delta.stack().max()
rf_best_set = delta["Random Forest"].idxmax()
knn_best_set = delta["KNN"].idxmax()
lgb_overall = F1.stack().idxmax()
rf_best_f1 = F1.loc[rf_best_set, "Random Forest"]
k255 = curve[(curve.k == curve.k.max())].set_index("Ranking")["F1-Score"]
noise = abs(k255.max() - base["LightGBM"])

# ---------- PCA
PF = pcr.pivot_table(index="Features", columns="Model", values="F1-Score")
PA = pcr.pivot_table(index="Features", columns="Model", values="ROC-AUC")
allf = PF.index.max()
n95 = int(pcn.loc[95])
n90 = int(pcn.loc[90])

HPO = (
    f"**Discussion.** Tuning helped most where the default model was clearly mis-configured. The **Decision Tree** default "
    f"is an unpruned tree that memorises the training data; limiting the depth (max_depth = 16) and requiring at least 8 "
    f"bookings per leaf raised its ROC-AUC by {D.loc['Decision Tree', 'ROC-AUC']:+.4f} (0.860 -> 0.939), because the "
    f"probabilities of a pruned tree are smoother and rank bookings much better. Its F1 changed by "
    f"{D.loc['Decision Tree', 'F1-Score']:+.4f}: the pruned tree is more conservative (precision up, recall down) - tuning "
    f"optimised ROC-AUC, not F1. **KNN** improved on every metric (F1 {D.loc['KNN', 'F1-Score']:+.4f}, ROC-AUC "
    f"{D.loc['KNN', 'ROC-AUC']:+.4f}) with 15 instead of 5 neighbours and distance weighting, which reduces the noise of single "
    f"neighbours in 255 dimensions. **LightGBM** gained the most among the ensembles (accuracy "
    f"{D.loc['LightGBM', 'Accuracy'] * 100:+.2f} pp, F1 {D.loc['LightGBM', 'F1-Score']:+.4f}) by using more and larger leaves "
    f"(num_leaves = 127) with a higher learning rate, i.e. a more complex model, which the 95k training rows can support. "
    f"**Random Forest, XGBoost and the MLP** were already close to their optimum with default settings (changes below 0.003 "
    f"in F1 and ROC-AUC): bagging and boosting ensembles are known to be robust to their default parameters, and the "
    f"small grids did not contain clearly better regions. For the MLP the selected L2 penalty (alpha = 0.001) traded some "
    f"precision for recall. Overall, hyper-parameter optimisation improved the ROC-AUC of all six models and the F1 of five of them."
)
HPO_SHORT = (f"Biggest gains: **Decision Tree** ROC-AUC {D.loc['Decision Tree', 'ROC-AUC']:+.3f} (pruning stops overfitting), "
             f"**KNN** F1 {D.loc['KNN', 'F1-Score']:+.3f}, **LightGBM** F1 {D.loc['LightGBM', 'F1-Score']:+.3f}. "
             "Random Forest, XGBoost and MLP were already near their optimum (< 0.003 change).")

lg, rf = lb.loc["LightGBM"], lb.loc["Random Forest"]
BEST = (
    f"**Best performing algorithm.** The tuned **LightGBM** is the best model: it has the highest F1-score ({f4(lg['F1-Score'])}), "
    f"ROC-AUC ({f4(lg['ROC-AUC'])}), PR-AUC ({f4(lg['PR-AUC (AP)'])}), recall ({pc(lg['Recall'])}) and the lowest Brier score "
    f"({f4(lg['Brier score'])}, i.e. the best calibrated probabilities), and it trains in about one second. Of the "
    f"{int(cms.loc['LightGBM', 'TP'] + cms.loc['LightGBM', 'FN']):,} cancellations in the test set it detects "
    f"{int(cms.loc['LightGBM', 'TP']):,} and misses {int(cms.loc['LightGBM', 'FN']):,}, while raising "
    f"{int(cms.loc['LightGBM', 'FP']):,} false alarms among {int(cms.loc['LightGBM', 'TN'] + cms.loc['LightGBM', 'FP']):,} kept bookings. "
    f"The tuned **Random Forest** is practically tied in accuracy ({pc(rf['Accuracy'])} vs {pc(lg['Accuracy'])}) and has the "
    f"highest precision ({pc(rf['Precision'])}) and specificity ({pc(rf['Specificity'])}) - it produces the fewest false alarms "
    f"({int(cms.loc['Random Forest', 'FP']):,}), which matters if the hotel uses the prediction for overbooking - but it misses "
    f"{int(cms.loc['Random Forest', 'FN'] - cms.loc['LightGBM', 'FN']):,} more cancellations than LightGBM. XGBoost is third; the "
    "single Decision Tree, the MLP and KNN are clearly weaker. The ranking confirms the expectation from the EDA: the problem is "
    "driven by non-linear rules and interactions of categorical attributes (deposit type x country x segment), which tree "
    "ensembles capture best, whereas the distance-based KNN suffers from the 255 mostly-binary dimensions and the MLP, although "
    "flexible, does not beat gradient boosting on tabular data."
)
BEST_SHORT = (f"LightGBM: best F1 ({lg['F1-Score']:.3f}), ROC-AUC ({lg['ROC-AUC']:.3f}), recall ({pc(lg['Recall'], 1)}), "
              f"best calibrated (Brier {lg['Brier score']:.3f}). Random Forest: same accuracy, highest precision "
              f"({pc(rf['Precision'], 1)}) -> fewest false alarms. Tree ensembles win; KNN weakest.")
g = gain.head(6)
GAIN = (
    f"The most important feature of LightGBM is by far **deposit type = Non Refund** ({g.iloc[0]:.1f}% of the total gain), followed "
    f"by **lead time** ({gain['num__lead_time']:.1f}%), the travel **agent** ({gain['num__agent']:.1f}%), **country = Portugal** "
    f"({gain['cat__country_PRT']:.1f}%), **market segment = Online TA** ({gain['cat__market_segment_Online TA']:.1f}%) and the number of "
    f"**special requests** ({gain['num__total_of_special_requests']:.1f}%). These are exactly the patterns found in the EDA "
    "(Section 3.5), so the model has learnt meaningful business rules rather than noise."
)
GAIN_SHORT = "Top drivers: Non Refund deposit (26% of gain), lead time, agent, Portugal, Online TA, special requests - same as in the EDA."

common_txt = ""
FS_COMMON = (
    "The techniques agree on the core signals: lead time, country = PRT, special requests, parking spaces and previous "
    "cancellations are in the top 12 of almost every ranking, and deposit type (Non Refund / No Deposit) leads the ANOVA, "
    "mutual information and L1 rankings. They differ on the rest: the ANOVA F-test only measures linear mean differences and "
    "prefers binary flags, mutual information also rewards non-linear numerical features (agent, ADR), and LightGBM's "
    "split-count importance favours numerical features with many possible thresholds (lead time, agent, ADR, arrival week) "
    "over binary dummies that need only one split. The variance threshold removes 140 near-constant one-hot columns (mostly "
    "rare countries) and keeps 115; the L1 logistic regression keeps 44 features with non-zero weights."
)
FS = (
    f"The two tables and the figure above show that feature selection kept or improved the performance in most cases while using only "
    f"50 of the 255 columns, i.e. **80% fewer features** and shorter training times (e.g. Random Forest "
    f"{TT.loc['All features', 'Random Forest']:.1f} s -> {TT.loc[rf_best_set, 'Random Forest']:.1f} s). The selections that **boosted** performance are:"
)
FS_ITEMS = [
    f"**{rf_best_set} with Random Forest** - the largest gain: F1 {f4(base['Random Forest'])} -> **{f4(rf_best_f1)}** "
    f"({delta.loc[rf_best_set, 'Random Forest']:+.4f}), accuracy {pc(ACC.loc['All features', 'Random Forest'])} -> "
    f"{pc(ACC.loc[rf_best_set, 'Random Forest'])}, ROC-AUC {f4(AUC.loc['All features', 'Random Forest'])} -> {f4(AUC.loc[rf_best_set, 'Random Forest'])}. "
    f"With only 50 features this Random Forest has the **highest F1-score of the whole project** (tuned LightGBM on all features: {f4(base['LightGBM'])}). "
    f"LightGBM importance (F1 {f4(F1.loc['LightGBM importance (k=50)', 'Random Forest'])}) and mutual information ({f4(F1.loc['Mutual information (k=50)', 'Random Forest'])}) give similar gains. "
    "Random Forest chooses a random subset of features at every split; when most of the 255 columns are rare, useless dummies, many "
    "splits are wasted on them. Removing them lets every tree use informative features more often.",
    f"**{knn_best_set} with KNN** - F1 {f4(base['KNN'])} -> {f4(F1.loc[knn_best_set, 'KNN'])} ({delta.loc[knn_best_set, 'KNN']:+.4f}); ANOVA and L1 selections "
    "also improve KNN. KNN treats every dimension equally in its distance, so dropping irrelevant dimensions directly improves the neighbourhoods (curse of dimensionality).",
    f"**Decision Tree** improves slightly only with mutual information ({delta.loc['Mutual information (k=50)', 'Decision Tree']:+.4f}); a single tree already selects its own features.",
    f"**LightGBM** does not profit from feature selection: the best change is {delta['LightGBM'].max():+.4f} (variance threshold) and the k = 50 "
    f"subsets cost between {abs(delta.loc['LightGBM importance (k=50)', 'LightGBM']):.4f} and {abs(delta.loc['ANOVA F-test (k=50)', 'LightGBM']):.4f} F1. Gradient boosting "
    "already ignores useless features (they are simply never chosen for a split), so removing them only removes a little information.",
    f"The **ANOVA F-test** and the **L1 logistic regression** subsets hurt the tree ensembles (Random Forest F1 {delta.loc['ANOVA F-test (k=50)', 'Random Forest']:+.4f} with ANOVA), "
    "because both methods rank features by *linear* relationships and drop features whose effect is non-linear (e.g. arrival week, ADR, agent ID).",
]
CURVE = (
    f"The figure above shows how many features LightGBM really needs. With the LightGBM-importance ranking, 30 features already reach "
    f"F1 {curve[(curve.k == 30) & (curve.Ranking == 'LightGBM importance')]['F1-Score'].iloc[0]:.3f} and 50 features "
    f"{curve[(curve.k == 50) & (curve.Ranking == 'LightGBM importance')]['F1-Score'].iloc[0]:.3f}, almost the full-feature value; the "
    f"ANOVA ranking needs about 100-150 features for the same level. Note that the same 255 features in a different column order give "
    f"F1 = {k255.max():.4f} instead of {base['LightGBM']:.4f}: differences smaller than about {noise:.3f} are within the natural "
    "variation of the training algorithm and should not be over-interpreted. The Random Forest gain above (+0.0075) is larger than this."
)
PCA = (
    f"PCA **reduced the performance of every model**. With {n95} components (95% of the variance) LightGBM drops from F1 "
    f"{f4(PF.loc[allf, 'LightGBM'])} to {f4(PF.loc[n95, 'LightGBM'])} and ROC-AUC from {f4(PA.loc[allf, 'LightGBM'])} to "
    f"{f4(PA.loc[n95, 'LightGBM'])}; Random Forest from {f4(PF.loc[allf, 'Random Forest'])} to {f4(PF.loc[n95, 'Random Forest'])} and the "
    f"Decision Tree from {f4(PF.loc[allf, 'Decision Tree'])} to {f4(PF.loc[n95, 'Decision Tree'])}. With only 10 components "
    f"(59% of the variance) the loss is much larger (LightGBM F1 {f4(PF.loc[10, 'LightGBM'])}). KNN is the most tolerant model: "
    f"with 30 components its F1 ({f4(PF.loc[30, 'KNN'])}) is close to the full-feature value ({f4(PF.loc[allf, 'KNN'])}) while each "
    "distance is computed in 30 instead of 255 dimensions, so PCA is a reasonable speed-up for KNN only."
)
PCA2 = (
    "Why does PCA hurt? (1) PCA is **unsupervised**: it keeps the directions with the largest variance, not the directions "
    "that separate cancelled from kept bookings - a rare but decisive dummy such as `deposit_type_Non Refund` has little "
    "variance and is spread over many components. (2) Each component is a linear **mixture** of many original features, so "
    "the clear thresholds that trees use ('Non Refund = 1', 'lead time > 180') disappear and the trees need many more splits to "
    "approximate them. (3) Variance was retained only up to 95%, so some information is lost by construction. **Feature "
    "selection is therefore the better way to reduce dimensionality for this dataset:** it keeps the original, interpretable "
    "features, reduces the dimension by 80% and even improves Random Forest and KNN, whereas PCA with a similar number of "
    "dimensions (25-35) loses 0.03-0.06 F1 for the tree models."
)
CONCL = (
    "We built an end-to-end pipeline that predicts hotel booking cancellations from information available at booking time. "
    "The EDA showed that deposit type, lead time, country, market segment and commitment signals (special requests, parking) "
    "drive cancellations. After cleaning errors, imputing missing values, analysing outliers with three methods, removing "
    "leakage, scaling and one-hot encoding, six algorithms were compared before and after hyper-parameter optimisation. "
    f"Tuning improved all models' ROC-AUC, especially the Decision Tree and KNN. The tuned **LightGBM** is the best overall model "
    f"(accuracy {pc(lg['Accuracy'])}, F1 {f4(lg['F1-Score'])}, ROC-AUC {f4(lg['ROC-AUC'])}), with Random Forest a close second "
    f"and the most precise. Feature selection reduced the 255 predictors to 50 without loss; RFE selection boosted Random Forest "
    f"to F1 {f4(rf_best_f1)} - the best F1 of the project - and mutual information boosted KNN. PCA reduced the performance of all "
    "models and is not recommended for this dataset. In practice the model can flag high-risk bookings (long lead time, "
    "non-refundable group blocks, no special requests) so that the hotel can overbook selectively and contact these guests early."
)
CONCL_BULLETS = [
    "EDA: **deposit type, lead time, country, segment and special requests** drive cancellations (Non Refund 99.4%, lead time >1 year 68%).",
    "Preprocessing: errors removed, missing values imputed with meaning, outliers kept (informative), leakage columns dropped, scaling + one-hot -> 255 predictors.",
    f"Best model: **tuned LightGBM** - accuracy {pc(lg['Accuracy'], 1)}, F1 {lg['F1-Score']:.3f}, ROC-AUC {lg['ROC-AUC']:.3f}; Random Forest close second with the highest precision.",
    "Tuning helped most for Decision Tree (ROC-AUC +0.08) and KNN; ensembles were already near-optimal.",
    f"Feature selection: 50 of 255 features are enough; **RFE boosted Random Forest to F1 {rf_best_f1:.3f}** (best of the project), mutual information boosted KNN.",
    "PCA lowered all models (mixes interpretable features, ignores the target) -> feature selection is preferred.",
]
FS_SHORT = (f"**Boosted:** Random Forest + RFE F1 {base['Random Forest']:.3f} -> **{rf_best_f1:.3f}** (best F1 of the project), KNN + mutual "
            f"information {base['KNN']:.3f} -> {F1.loc[knn_best_set, 'KNN']:.3f}. LightGBM unchanged (already ignores useless features); "
            "ANOVA/L1 (linear criteria) hurt the trees.")
PCA_SHORT = (f"PCA **lowers every model**: LightGBM F1 {PF.loc[allf, 'LightGBM']:.3f} -> {PF.loc[n95, 'LightGBM']:.3f} with {n95} components (95% variance). "
             "PCA is unsupervised and mixes clear one-hot rules into linear combinations. Only KNN keeps its level (30 PCs) -> feature selection is the better reduction.")
FS_ABSTRACT = (f"Feature selection reduced the predictors by 80% without loss and boosted Random Forest (RFE, F1 {f4(rf_best_f1)}) and KNN, "
               "while PCA reduced the performance of all models.")


def fill(doc, runs):
    """Replace the placeholder paragraphs of make_report.py with the texts above."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    rep = {"HPO_TEXT": [HPO], "BEST_TEXT": [BEST], "GAIN_TEXT": [GAIN], "FS_COMMON": [FS_COMMON],
           "FS_TEXT": [FS] + ["- " + x for x in FS_ITEMS], "CURVE_TEXT": [CURVE], "PCA_TEXT": [PCA, PCA2], "CONCL": [CONCL]}
    for par in list(doc.paragraphs):
        txt = par.text
        if "FS_ABSTRACT" in txt:
            for r in par.runs:
                r.text = r.text.replace("FS_ABSTRACT", FS_ABSTRACT)
        if txt.strip() in rep:
            items = rep[txt.strip()]
            par.text = ""
            runs(par, items[0])
            anchor = par
            for extra in items[1:]:
                style = "List Bullet" if extra.startswith("- ") else None
                # insert after anchor
                from copy import deepcopy
                new_el = deepcopy(anchor._p)
                for r in list(new_el):
                    if r.tag.endswith("}r"):
                        new_el.remove(r)
                anchor._p.addnext(new_el)
                from docx.text.paragraph import Paragraph
                np_ = Paragraph(new_el, anchor._parent)
                if style:
                    np_.style = doc.styles[style]
                runs(np_, extra[2:] if style else extra)
                np_.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                anchor = np_
