"""
part6_synthesis_conclusion.py - Sections 20 through 23 + References of the Academic Research Paper.
Covers:
- Section 20: Empirical Synthesis & Resolution of Core Research Questions (RQ1 - RQ6)
- Section 21: Operational & Revenue Management Recommendations
- Section 22: Threats to Validity, Methodological Limitations & Future Work
- Section 23: Conclusion & Summary of Contributions
- References (Complete bibliography in APA format)
"""

import pandas as pd
from pathlib import Path

def get_section_20():
    df_rq = pd.read_csv("results/tables/final_research_questions_answers.csv")
    rq_cols = ["Research_Question", "Primary_Finding", "Statistical_Evidence", "Theoretical_Mechanism", "Curricular_Verdict"]
    rq_md = df_rq[rq_cols].to_markdown(index=False)

    df_lead = pd.read_csv("results/tables/final_model_leaderboard.csv")
    lead_cols = [
        "Global_Rank", "Model_Name", "Experiment_Phase", "Feature_Dimension",
        "Accuracy", "ROC_AUC", "PR_AUC", "Brier_Score", "F1_Pos", "Train_Time_Sec", "Inference_Time_Sec"
    ]
    lead_md = df_lead[lead_cols].to_markdown(index=False)

    text = """## Section 20: Empirical Synthesis & Resolution of Core Research Questions (RQ1 – RQ6)

### 20.1 Curricular Resolution of Core Research Questions (RQ1 – RQ6)
Table 20.1 synthesizes the empirical resolution of the six foundational research questions established in Section 3, cross-referencing parametric statistics, non-parametric tests, and cross-validated machine learning benchmarks.

**Table 20.1: Empirical Synthesis and Formal Resolution of Core Research Questions (RQ1 – RQ6)**

__RQ_TABLE__

### 20.2 The Global Master Model Leaderboard
To establish a definitive comparative record across all paradigms explored throughout this research lifecycle, Table 20.2 aggregates performance metrics across all 18 evaluated model configurations (spanning default baselines, hyperparameter-tuned models, feature selection subsets, and principal component projections).

**Table 20.2: Global Master Model Leaderboard Across All Evaluated Architectures and Paradigms**

__LEAD_TABLE__

![Final Model Leaderboard](../results/figures/final_model_leaderboard.png)
*Figure 20.1: Multi-panel comparative dashboard illustrating global performance across discrimination (ROC-AUC), calibration (Brier Score), operational loss, and inference latency.*

### 20.3 Multi-Criteria Pareto Optimality Analysis
Analyzing the multi-dimensional frontier across discrimination, calibration, operational cost, and inference latency reveals clear architectural hierarchy:

```
                          THE MULTI-CRITERIA PARETO FRONTIER
                          
    Discrimination (ROC-AUC) ▲
                             │                        ★ Tuned LightGBM (0.9621, 2.73s)
                      0.9600 ┼                       ▲
                             │                      ╱  ★ LightGBM (k=50) (0.9612, 3.08s)
                             │                     ╱   ★ Random Forest (k=50) (0.9609, 12.8s)
                      0.9550 ┼                    ╱    ★ Tuned XGBoost (0.9588, 7.68s)
                             │                   ╱     ★ Tuned Random Forest (0.9560, 17.7s)
                             │                  ╱
                      0.9500 ┼                 ╱
                             │                ╱
                      0.9450 ┼               ╱         ★ Tuned MLP (0.9466, 56.25s)
                             │              ╱
                      0.9350 ┼             ╱           ★ Tuned Decision Tree (0.9373, 2.36s)
                             │            ╱            ★ Tuned KNN (0.9363, 0.16s fit / 444s pred)
                             │           ╱
                             └──────────┴──────────────────────────────────────►
                               Fast (< 5s)             Slow (> 50s)
                                            Training Time (Wall-Clock)
```

1. **The Undisputed Champion (Tuned LightGBM):** Occupies the global Pareto optimum. It achieves Rank 1 in ROC-AUC ($0.9621$), Rank 1 in PR-AUC ($0.9433$), Rank 1 in Accuracy ($89.19\\%$), Rank 1 in positive F1-score ($0.8519$), Rank 1 in Brier calibration ($0.0750$), Rank 1 in minimal operational loss ($381,516 dollars), and trains in just **$2.73$ seconds**.
2. **The Production Efficiency Champion (LightGBM at $k=50$ Features):** Occupies the ideal production deployment point. By retaining only the top 50 features, it delivers **$0.9612$ ROC-AUC (99.91% retention)** and **$89.17\\%$ accuracy**, while reducing feature pipeline maintenance overhead by **$80.5\\%$**.
3. **The Unviable Production Architecture (Tuned KNN):** Despite achieving a respectable ROC-AUC of $0.9363$, KNN requires **$444.61$ seconds** to score 23,842 instances ($18,648 \\, \\mu s$ per prediction). In live hospitality distribution environments where APIs must return booking confirmations within 100 milliseconds, uncompressed KNN is computationally non-viable.

---
"""
    return text.replace("__RQ_TABLE__", rq_md).replace("__LEAD_TABLE__", lead_md)

def get_section_21():
    return """## Section 21: Operational & Revenue Management Recommendations

To translate empirical machine learning findings into tangible business value, we formulate an executive revenue management framework comprising four actionable operational strategies:

### 21.1 Dynamic Cost-Calibrated Overbooking ($p^* = 0.75$)
* **Current Industry Practice:** Most hotels implement static, property-wide overbooking buffers (e.g., arbitrarily overselling 5% to 10% of total physical room capacity across all dates). This static approach ignores booking-specific risk and generates costly walking incidents during low-cancellation periods.
* **Algorithmic Strategy:** Revenue management systems should deploy the tuned LightGBM model to calculate individualized cancellation probabilities $\\hat{p}_i$ for every confirmed booking. 
* **Operational Execution:** Rooms should be flagged for speculative overbooking resale **if and only if** the posterior probability exceeds the cost-optimal Bayes threshold:
  $$\\hat{p}_i \\ge p^* = 0.75$$
  Reservations with $\\hat{p}_i \\in [0.50, 0.74]$ must **not** be overbooked; while likely to cancel under symmetric loss, the severe $\\$300$ walking penalty makes overbooking them an expected-loss decision. Deploying this policy eliminates **69.4% of guest walking incidents**, delivering $\\$108,444$ in quarterly operational savings per property.

### 21.2 Reforming Wholesale & Group Deposit Structures
* **Current Industry Practice:** Hoteliers frequently mandate "Non-Refundable" deposit clauses for wholesale group blocks, assuming this guarantees financial protection. As proved in Section 6, these non-refundable blocks suffer an astonishing **99.36% cancellation rate** as tour operators dump uncommitted inventory.
* **Algorithmic Strategy:** Eliminate monolithic non-refundable group contracts in favor of **Dynamic Staggered Attrition Contracts**:
  * Implement progressive milestone release dates (e.g., 90, 60, and 30 days prior to arrival).
  * Require escalating earnest deposits tied directly to lead time (e.g., 20% at booking, 30% at 60 days, remaining 50% non-refundable at 30 days).
  * Enforce strict attrition caps permitting wholesale tour operators to release no more than 15% of reserved block inventory without forfeiting accrued deposits.

### 21.3 Automated High-Lead-Time Customer Engagement Workflows
* **Behavioral Insight:** Bookings secured with lead times $> 100$ days cancel at rates exceeding 50%. However, Section 9 demonstrated that submitting special requests collapses cancellation rates from **47.77% to 21.76% (-26.01%)**.
* **Algorithmic Strategy:** Hospitality CRM systems should establish automated, personalized outreach sequences for reservations identified as high-risk (lead time $> 90$ days and $\\hat{p}_i > 0.60$):
  * **Day 30 Post-Booking:** Automated outreach inviting the guest to select room preferences (e.g., pillow choice, floor preference, quiet zone).
  * **Day 60 Post-Booking:** Curated dining and local attraction packages.
  * **Day 14 Pre-Arrival:** Exclusive promotional offers for discounted on-site car parking. Because requesting a parking space empirically yields a **0% cancellation rate**, incentivizing parking reservations effectively secures guest arrival.
  * *Mechanism:* Transforming a passive reservation into an active, customized itinerary introduces psychological commitment and endowment effects, neutralizing rate-shopping churn.

### 21.4 Strategic Room Reassignment as a Retention Mechanism
* **Behavioral Insight:** Operational room reassignments (`room_type_changed`) collapse cancellation rates from **41.56% down to 5.41% (an absolute reduction of 36.15%)**.
* **Algorithmic Strategy:** PMS room allocation algorithms should systematically leverage complimentary upgrades as targeted retention interventions:
  * When a high-value reservation (high ADR, long stay) is predicted to have an elevated cancellation risk ($\hat{p}_i \\approx 0.60 - 0.70$), the front-desk system should pre-assign an upgraded room category (e.g., Standard to Deluxe).
  * Send an immediate confirmation notification: *"Congratulations, your booking has been upgraded complimentary to our Deluxe Suite."*
  * *Mechanism:* The perceived unexpected surplus eliminates the incentive to rate-shop competing properties and creates immediate brand loyalty.

---
"""

def get_section_22():
    return """## Section 22: Threats to Validity, Methodological Limitations & Future Work

### 22.1 Threats to Validity
1. **Threats to Internal Validity:**
   * *Proxy Variable Confounding:* The variable `deposit_type_Non_Refund` acts as an empirical proxy for wholesale group blocks rather than retail consumer commitment. While highly predictive in this specific historical dataset, changes in property management contracting practices could alter this association over time.
   * *Omitted Exogenous Variables:* The dataset lacks exogenous macroeconomic indicators, such as real-time jet fuel prices, airline ticket fluctuations, currency exchange rates (e.g., GBP/EUR for British tourists in the Algarve), and local weather anomalies. Incorporating external time-series data could further refine predictive precision.
2. **Threats to External Validity:**
   * *Geographical and Regional Specificity:* The data was collected from two hotels located exclusively in Portugal (Algarve and Lisbon). While the dual-property structure captures both leisure resort and urban metropolitan dynamics, caution must be exercised when generalizing findings to radically different lodging markets, such as North American suburban motels, Asian luxury casino resorts, or global mega-chains.
   * *Temporal Horizon Stability:* The observations span July 2015 to August 2017. Post-pandemic consumer booking behaviors (e.g., heightened hygiene concerns, remote work flexibility, altered OTA policies) may exhibit structural shifts requiring model retraining and domain adaptation.

### 22.2 Methodological Limitations
1. **Static Cross-Sectional Formulation:** This research models cancellation prediction as a static cross-sectional classification task evaluated at a single point in time. In production, a booking is an evolving dynamic process: cancellation probability is not fixed, but fluctuates as days-to-arrival elapses and modifications occur.
2. **Property-Specific Calibration:** Although the models were trained across both properties simultaneously, individual properties possess unique ADR distributions and walking penalties. In multi-property enterprise deployments, cost parameters ($C_{\\text{walk}}, C_{\\text{empty}}$) must be parameterized dynamically per property and season.

### 22.3 Promising Directions for Future Research
1. **Survival Analysis & Dynamic Hazard Modeling:** Formulating cancellation forecasting within a time-to-event survival analysis framework (e.g., Cox Proportional Hazards or DeepSurv). Estimating the continuous hazard rate $\\lambda(t)$ as a function of elapsed time would allow revenue managers to dynamically track cancellation risk day-by-day across the entire booking lifecycle.
2. **Deep Sequential Architectures on PMS Audit Logs:** Modern enterprise PMS systems record event logs detailing every micro-interaction (date changes, payment attempts, room inquiries). Applying Recurrent Neural Networks (RNNs) or Transformer-based sequence models to raw PMS event streams could capture temporal behavioral signatures prior to explicit cancellation.
3. **Reinforcement Learning for Joint Pricing and Overbooking:** Integrating calibrated probability predictions into Deep Reinforcement Learning (DRL) agents that simultaneously optimize real-time dynamic room pricing and probabilistic overbooking buffers under stochastic arrival constraints.
4. **Federated Hospitality Learning:** Investigating privacy-preserving Federated Learning (FL) across multi-brand lodging chains, enabling global model training on millions of bookings without centralizing sensitive proprietary hotel guest data.

---
"""

def get_section_23():
    return """## Section 23: Conclusion & Summary of Contributions

### 23.1 Summary of Methodological & Empirical Contributions
This academic research paper has delivered an exhaustive, publication-grade investigation into predictive machine learning for hotel booking cancellations, establishing critical empirical milestones across the data science lifecycle:

1. **Rigorous Data Hygiene & Leakage Elimination:** Audited $N = 119,390$ raw records, eliminated 180 ghost bookings and 2 extreme pricing typos, resolved missing data through domain imputation, and permanently excised post-outcome target leakage features (`reservation_status`, `reservation_status_date`), creating a clean, verified analytical corpus of $N = 119,208$ bookings across $d = 257$ dimensions.
2. **Inferential Hypothesis Testing & Domain Discovery:** Conducted comprehensive parametric and non-parametric testing on large samples ($N > 100,000$). We proved that `deposit_type` ($V = 0.4815$) and `lead_time` ($d = 0.6342$, canceled bookings planned 68 days earlier) are the dominant drivers of cancellation. Crucially, we exposed the "ADR Significance Fallacy," demonstrating that room rate is not an isolated churn catalyst ($d = 0.0986$).
3. **Multi-Model Architectural Benchmarking:** Benchmarked six diverse machine learning architectures under standardized 5-fold stratified cross-validation and holdout testing ($N_{\\text{test}} = 23,842$). We established that **LightGBM** is the decisive global champion, achieving **0.9621 ROC-AUC**, **0.9433 PR-AUC**, **89.19% Accuracy**, **0.8519 positive F1-score**, and an elite **Brier Score of 0.0750**, while requiring only **2.73 seconds for full-dataset training**.
4. **Bias-Variance Tuning Dynamics:** Quantified architectural responsiveness to hyperparameter optimization. Unconstrained weak learners (Decision Tree, KNN) achieved dramatic performance gains (+7.63% and +3.38% ROC-AUC), while bagging ensembles (Random Forest) demonstrated mathematical optimality under default configurations.
5. **The 50-Feature Pareto Frontier:** Proved that supervised feature selection preserves **99.91% of baseline discrimination** ($0.9612$ vs. $0.9621$ ROC-AUC) while reducing dimensionality by **80.5%**, showing that pruning over 100 near-zero variance categorical dummies acts as structural regularization.
6. **Theoretical Characterization of "Diagonal Boundary Destruction":** Discovered and formalized why unsupervised linear projection (PCA) systematically degrades non-linear tree ensembles on tabular data (LightGBM dropping to $0.9478$ at PCA-50) due to diagonal hyperplane rotation, while accelerating instance-based KNN inference by **41.5-fold**.
7. **Cost-Sensitive Optimization & Executive Impact:** Integrated epistemic probability calibration with asymmetric operational costs ($C_{\\text{walk}} = \\$300$, $C_{\\text{empty}} = \\$102$). We proved mathematically and verified empirically that shifting the decision threshold to $p^* = 0.75$ reduces total misclassification losses from $\\$489,960$ to $\\$381,516$, delivering a **quarterly net financial savings of $\\$108,444 (a 22.13% reduction)**.

### 23.2 Concluding Remarks
By bridging the gap between advanced machine learning theory and the operational realities of revenue management, this research demonstrates that modern data science can transform hospitality operations from reactive, heuristic-driven risk management into a proactive, mathematically calibrated competitive advantage.

---

## References

1. Antonio, N., de Almeida, A., & Nunes, L. (2017). Predicting hotel booking cancellation with a machine learning classification model. In *Proceedings of the 2017 4th International Conference on e-Democracy & e-Government (ICEDEG)* (pp. 149–154). IEEE. https://doi.org/10.1109/ICEDEG.2017.7962527
2. Antonio, N., de Almeida, A., & Nunes, L. (2019a). Big data in hotel revenue management: Exploring cancellation drivers to gain competitive advantage. *Current Issues in Tourism*, 22(8), 980–995. https://doi.org/10.1080/13683500.2017.1368350
3. Antonio, N., de Almeida, A., & Nunes, L. (2019b). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
4. Belobaba, P. P. (1987). Air travel demand and airline seat inventory management. Doctoral dissertation, Flight Transportation Laboratory, Massachusetts Institute of Technology, Cambridge, MA.
5. Belobaba, P. P. (1989). Application of a probabilistic decision model to airline seat inventory control. *Operations Research*, 37(2), 183–197. https://doi.org/10.1287/opre.37.2.183
6. Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? In *International Conference on Database Theory* (pp. 217–235). Springer, Berlin, Heidelberg.
7. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
8. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC Press, Boca Raton, FL.
9. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
10. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785
11. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates, Hillsdale, NJ.
12. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964
13. Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press, Princeton, NJ.
14. Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 233–240). ACM. https://doi.org/10.1145/1143844.1143874
15. Drummond, C., & Holte, R. C. (2006). Cost curves: An improved alternative to ROC curves for evaluating binary classifiers. *Machine Learning*, 65(1), 95–130. https://doi.org/10.1007/s10994-006-8199-5
16. Elkan, C. (2001). The foundations of cost-sensitive learning. In *Proceedings of the 17th International Joint Conference on Artificial Intelligence* (Vol. 1, pp. 973–978). Morgan Kaufmann Publishers.
17. Falk, M., & Vieru, M. (2018). Modelling the cancellation behaviour of hotel guests. *International Journal of Contemporary Hospitality Management*, 30(10), 3100–3116. https://doi.org/10.1108/IJCHM-08-2017-0509
18. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
19. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems*, 35, 507–520.
20. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *International Conference on Machine Learning* (pp. 1321–1330). PMLR.
21. Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3(Mar), 1157–1182.
22. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer, New York.
23. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 374(2065), 20150202. https://doi.org/10.1098/rsta.2015.0202
24. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(4), 1–21. https://doi.org/10.1145/2382577.2382579
25. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.
26. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
27. Littlewood, K. (1972). Forecasting and control of passenger bookings. In *12th AGIFORS Symposium Proceedings* (pp. 95–117).
28. Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *The Annals of Mathematical Statistics*, 18(1), 50–60. https://doi.org/10.1214/aoms/1177730491
29. Morales, D. R., & Wang, J. (2010). Forecasting cancellation rates for airline passenger bookings. *European Journal of Operational Research*, 202(2), 554–562. https://doi.org/10.1016/j.ejor.2009.05.034
30. Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology and Climatology*, 12(4), 595–600.
31. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430
32. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
33. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31, 6638–6648.
34. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536. https://doi.org/10.1038/323533a0
35. Sanchez-Medina, A. J., & C-Sanchez, E. (2020). Using machine learning and big data for efficient forecasting of hotel booking cancellations. *International Journal of Hospitality Management*, 89, 102546. https://doi.org/10.1016/j.ijhm.2020.102546
36. Talluri, K. T., & van Ryzin, G. J. (2004). *The Theory and Practice of Revenue Management*. Springer Science & Business Media, New York. https://doi.org/10.1007/b139000
37. Weatherford, L. R., & Kimes, S. E. (2003). A comparison of forecasting methods for hotel revenue management. *International Journal of Forecasting*, 19(3), 401–415. https://doi.org/10.1016/S0169-2070(02)00011-0
38. Welch, B. L. (1947). The generalization of 'Student's' problem when several different population variances are involved. *Biometrika*, 34(1/2), 28–35. https://doi.org/10.2307/2332510

---
"""
    return text.replace("__RQ_TABLE__", rq_md).replace("__LEAD_TABLE__", lead_md)
