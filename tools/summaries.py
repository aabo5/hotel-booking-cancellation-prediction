"""Final markdown summaries of notebooks 01, 09 and 10 (written from the executed outputs)."""
SUMMARIES = {
    "SUMMARY_01": """## Summary of Exploration Findings
1. **Size:** 119,390 bookings x 32 attributes (20 numerical, 12 categorical/text); 31,994 rows are exact duplicates (common in PMS exports: several identical rooms booked together).
2. **Target:** 37.0% cancelled vs 63.0% not cancelled - moderately imbalanced, so we report F1, precision, recall and ROC-AUC besides accuracy.
3. **Hotel type:** the City Hotel cancels much more often (41.7%) than the Resort Hotel (27.8%).
4. **Deposit paradox:** 'Non Refund' bookings are cancelled 99.4% of the time (vs 28.4% for 'No Deposit'); 97% of them come from Portugal, 63% are Groups and their median lead time is 183 days (overall 69) - typical block bookings of tour operators that are later released.
5. **Lead time** is the strongest numerical signal (correlation +0.29): median 113 days for cancelled vs 45 days for kept bookings; the cancellation rate rises from 9.6% (0-7 days) to 67.7% (>1 year).
6. **Commitment signals lower the risk:** bookings with no special request cancel 47.7% of the time, those with requests ~22%; not a single booking that asked for a parking space was cancelled.
7. **Market segment / country:** Groups (61.1%) and guests from Portugal (56.6%, 40.7% of all bookings) cancel most; Direct and Corporate bookings least.
8. **Seasonality:** most bookings arrive in July-August; the cancellation rate is highest in spring/summer months.
9. **Missing values:** `company` (94.3%), `agent` (13.7%), `country` (0.41%) and `children` (4 rows) - handled in notebook 02.
10. **Leakage:** `reservation_status` maps 1-to-1 to the target and must be removed.""",
    "SUMMARY_09": """## Summary: feature selection and PCA
**Feature selection (before vs after)**
* Six techniques from three families were applied: variance threshold, ANOVA F-test and mutual information (filter), RFE with a Decision Tree (wrapper), L1 logistic regression and LightGBM importance (embedded).
* With only **50 of 255 features (-80%)** the F1-score improved in 10 of the 24 model/technique combinations and stayed within the noise level (-0.003) or better in 16 of them; training is faster.
* **Boosted performance:** RFE (k=50) raised the **Random Forest** F1 from 0.8499 to **0.8574** (accuracy 89.38% -> 89.77%, ROC-AUC 0.9594 -> 0.9615) - the highest F1 of the whole project; LightGBM importance (0.8572) and mutual information (0.8540) gave similar gains. **KNN** improved with mutual information (F1 0.8046 -> 0.8130), ANOVA and L1 selection, because irrelevant dimensions no longer distort its distances.
* **LightGBM** does not profit (best +0.0002 with the variance threshold): boosting already ignores useless features. ANOVA and L1 (linear criteria) hurt the tree ensembles.
* Re-ordering the same 255 columns changes LightGBM's F1 by ~0.003, so smaller differences are noise.

**PCA (before vs after)**
* 25 components keep 90% and 35 components 95% of the variance (one-hot columns spread the variance).
* PCA **lowered every model**: LightGBM F1 0.8539 -> 0.8230 (35 PCs), Random Forest 0.8499 -> 0.8220, Decision Tree 0.8151 -> 0.7618; with 10 PCs the loss is much larger. Only KNN stays close (0.8015 with 30 PCs vs 0.8046).
* Reason: PCA is unsupervised (keeps variance, not class separation) and mixes clear one-hot rules (e.g. Non Refund = 1) into linear combinations that trees cannot split cleanly.
* **Conclusion:** supervised feature selection is the better way to reduce dimensionality for this dataset.""",
    "SUMMARY_10": """## Conclusion
* **Best model: tuned LightGBM** - accuracy 89.37%, F1 0.8539, ROC-AUC 0.9623, PR-AUC 0.9435, recall 83.83% and the lowest Brier score (0.0749, best calibrated probabilities).
* **Random Forest** is practically tied in accuracy (89.38%) and has the highest precision (89.34%) and specificity (94.30%), i.e. the fewest false alarms (855) - useful if the predictions are used for overbooking - but it misses 247 more cancellations than LightGBM.
* Ranking by F1: LightGBM > Random Forest > XGBoost > MLP > Decision Tree > KNN. Tree ensembles win because cancellations depend on non-linear rules and interactions of categorical features.
* Hyper-parameter tuning improved the ROC-AUC of all six models; the largest gains were for the Decision Tree (+0.079 ROC-AUC, pruning stops overfitting), KNN (+0.031 F1) and LightGBM (+0.023 F1).
* Most important drivers of the best model: Non Refund deposit, lead time, agent, country = PRT, Online TA segment, special requests.
* **Practical use:** flag high-risk bookings (long lead time, non-refundable group blocks, no special requests) and overbook selectively / contact these guests early.""",
}
