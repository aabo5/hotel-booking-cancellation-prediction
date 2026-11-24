"""
part1_intro.py - Sections 1 through 5 of the Academic Research Paper.
Covers:
- Section 1: Title, Abstract, Keywords
- Section 2: Introduction & Background
- Section 3: Problem Statement & Research Objectives
- Section 4: Literature Review / Related Work
- Section 5: Dataset Description & Provenance
"""

def get_section_1():
    return """# Predicting Hotel Booking Cancellations: An Empirical Investigation of Machine Learning Architectures, Feature Selection, Probability Calibration, and Asymmetric Operational Economics

**Lead Author & Senior Data Scientist**: University Data Science Research Team  
**Institution**: Faculty of Engineering and Natural Sciences, Computer Engineering & Data Science Program  
**Course Project**: FSMVU Data Science Research Project  
**Repository**: `hotel-booking-cancellation-prediction`  
**Date**: Academic Year 2025–2026  

---

## Section 1: Title, Abstract, Keywords

### Abstract
Hotel booking cancellations represent one of the most critical structural challenges in modern hospitality revenue management, destabilizing occupancy forecasts, creating perishable inventory spoilage, and driving inefficient overbooking practices. Historically, lodging establishments have relied on aggregate historical averages or blunt heuristic overbooking buffers to mitigate cancellation uncertainty. This paper presents an exhaustive, publication-grade academic investigation into predicting individualized booking cancellations using supervised machine learning, advanced feature selection, probability calibration, and asymmetric operational cost optimization. 

Utilizing an empirical corpus of $N = 119,208$ rigorously validated real-world reservations from two contrasting Portuguese hotel properties (a coastal resort and an urban metropolitan hotel), we formulate a 36-variable observational feature space expanded into $d = 257$ dimensions via strictly leak-free preprocessing. We benchmark six foundational machine learning architectures spanning distinct inductive learning paradigms: Decision Trees, Random Forests, Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LightGBM), Multi-Layer Perceptrons (MLP), and K-Nearest Neighbors (KNN). Through rigorous 5-fold stratified cross-validation and held-out test partitioning ($N_{\\text{test}} = 23,842$), we conduct four sequential experiments: baseline benchmarking, Bayesian/grid hyperparameter optimization, supervised feature subset selection ($k \\in [10, 257]$), and unsupervised dimensionality reduction via Principal Component Analysis ($k \\in [10, 100]$). 

Our empirical results demonstrate that **LightGBM** is the decisive global champion, achieving a test **ROC-AUC of 0.9621**, **PR-AUC of 0.9433**, **Accuracy of 89.19%**, **Positive F1-Score of 0.8519**, and a state-of-the-art **Brier Score of 0.0750**, while requiring only $2.73$ seconds for full-dataset training. We uncover that tree-based ensembles are remarkably resilient under supervised feature selection; selecting the top 50 features retains **99.91%** of baseline discrimination power ($0.9612$ vs. $0.9621$ ROC-AUC) while reducing feature dimensionality by **80.5%**. Conversely, we discover and mathematically characterize the "Diagonal Boundary Destruction" phenomenon under PCA: rotating sparse binary dummy features into continuous principal components systematically degrades tree ensemble discrimination (LightGBM dropping to $0.9478$ at PCA-50), while accelerating non-parametric KNN inference by **41.5-fold**. Finally, integrating epistemic probability calibration with an empirical revenue management loss function reflecting asymmetric operational costs ($C_{\\text{walk}} = \\$300$ for walking an overbooked arriving guest vs. $C_{\\text{empty}} = \\$102$ for room spoilage) reveals that shifting the decision threshold from the standard $p = 0.50$ to the cost-optimal threshold of $p^* = 0.75$ reduces total misclassification losses from $\\$489,960$ to $\\$381,516$, delivering a **quarterly net financial savings of $\\$108,444$ (a 22.13% reduction)**. We synthesize our findings into an actionable executive framework for dynamic, revenue-optimal hospitality management.

### Keywords
Hotel Revenue Management; Booking Cancellation Prediction; Machine Learning; LightGBM; Gradient Boosting; Supervised Feature Selection; Principal Component Analysis; Probability Calibration; Brier Score; Asymmetric Misclassification Loss; Decision Threshold Optimization; Hospitality Analytics.

---
"""

def get_section_2():
    return """## Section 2: Introduction & Background

The modern hospitality industry is defined by an extreme economic vulnerability: perishable inventory combined with high fixed capital investment and negligible marginal servicing costs (Talluri & van Ryzin, 2004; Weatherford & Kimes, 2003). Unlike manufactured goods that can be inventoried, warehoused, and sold across subsequent financial quarters, an unsold hotel room on a given night represents a complete, irreversible loss of revenue potential. The marginal cost of servicing an occupied room (linen laundering, housekeeping labor, utility consumption) typically ranges between 10% and 25% of the Average Daily Rate (ADR), meaning that the vast majority of room revenue flows directly toward amortizing fixed property debt, real estate financing, and foundational operational overhead. Consequently, maximizing occupancy while optimizing room rates—the foundational mandate of Revenue Management (RM)—is paramount to solvency and long-term profitability.

Over the past two decades, the proliferation of digital travel intermediaries, notably Online Travel Agencies (OTAs) such as Booking.com, Expedia, and Agoda, has fundamentally altered consumer booking dynamics. To capture market share and eliminate consumer friction, OTAs introduced and aggressively promoted flexible "Free Cancellation" policies, allowing prospective travelers to reserve multiple accommodations months in advance with zero financial commitment or downside risk (Antonio, de Almeida, & Nunes, 2017). This structural shift precipitated a dramatic surge in customer churn, speculative multi-property reservations, and strategic "rate-shopping," wherein consumers book a tentative room and continuously monitor algorithmic price fluctuations, canceling immediately if a cheaper alternative emerges prior to the arrival date (Falk & Vieru, 2018).

In high-demand European leisure and business destinations, aggregate booking cancellation rates frequently fluctuate between 30% and 45% (Antonio, de Almeida, & Nunes, 2019a). This immense volume of cancellation volatility introduces severe systemic operational frictions across lodging organizations:
1. **Perishable Inventory Spoilage:** Last-minute cancellations leave rooms vacant during peak demand periods when replacement guests can no longer be acquired, directly eroding gross operating margins.
2. **Operational and Labor Inefficiencies:** Departmental scheduling—including housekeeping staff allocation, front-desk staffing rosters, and food and beverage (F&B) provisioning—is calibrated weeks in advance based on projected occupancy numbers. Massive cancellation spikes lead to acute labor over-allocation, food waste, and inflated operating expenses.
3. **Overbooking Dilemmas and Guest "Walking" Penalties:** To counteract expected cancellations, hotels historically implemented blunt, heuristic overbooking buffers (e.g., overselling total capacity by 5% to 15%). However, when cancellations fail to materialize as projected, hotels experience catastrophic capacity breaches. Arriving guests must be "walked" to competing properties, obligating the hotel to pay for alternative accommodations, luxury ground transportation, complimentary meals, and brand reparations, while inflicting permanent damage on customer lifetime value and brand reputation.

Historically, hotel revenue managers approached cancellation management using aggregated, property-level heuristics, such as static cancellation rate averages, historical pickup curves, or basic linear regression based on seasonality and days-to-arrival (Morales & Wang, 2010). These classical operational models treat reservations as homogeneous statistical aggregates, failing completely to capture the rich, individualized behavioral footprints embedded within contemporary Property Management Systems (PMS). Today, every booking contains multifaceted behavioral signals: lead time, intermediation channel, room modification histories, special requests, historical cancellation records, and deposit structures. 

The transition from aggregate heuristic buffers to individualized, machine learning-driven probabilistic classification offers a transformative paradigm shift for hospitality operations. By training supervised algorithms to evaluate the multi-dimensional feature space of each incoming reservation, hotel management systems can quantify the precise probability of cancellation at any stage of the booking lifecycle. However, successfully deploying machine learning within enterprise hospitality requires addressing critical methodological challenges: preventing target data leakage, optimizing high-dimensional feature representations, handling asymmetric class distributions, evaluating epistemic probability calibration, and explicitly aligning algorithmic decision thresholds with real-world financial cost structures.

This research paper provides a comprehensive, rigorous academic and empirical investigation into predicting hotel booking cancellations. Drawing upon real-world enterprise PMS records from Portugal spanning July 2015 to August 2017, we formulate and execute an end-to-end data science lifecycle. We demonstrate how advanced gradient boosting architectures, combined with rigorous leakage prevention, supervised feature selection, probability calibration, and asymmetric economic threshold optimization, can convert latent PMS data into millions of dollars in actionable operational value.

---
"""

def get_section_3():
    return """## Section 3: Problem Statement & Research Objectives

### 3.1 Mathematical Problem Formulation
We formalize the hotel booking cancellation prediction task as a supervised binary classification problem on tabular observational data. Let $\\mathcal{D} = \\{(\\mathbf{x}_i, y_i)\\}_{i=1}^N$ denote a dataset of $N$ independent historical reservation records, where each observation $\\mathbf{x}_i \\in \\mathcal{X} \\subset \\mathbb{R}^d$ represents a $d$-dimensional feature vector of pre-arrival booking attributes, and $y_i \\in \\mathcal{Y} = \\{0, 1\\}$ represents the ground-truth terminal cancellation status:

$$y_i = \\begin{cases} 1 & \\text{if the reservation is canceled or results in a no-show} \\\\ 0 & \\text{if the guest arrives, checks in, and completes the stay} \\end{cases}$$

The primary objective is to learn a parameterized scoring hypothesis $f_\\theta: \\mathcal{X} \\to [0, 1]$ from the hypothesis space $\\mathcal{H}$ that maps input features to a well-calibrated posterior probability of cancellation:

$$\\hat{p}_i = f_\\theta(\\mathbf{x}_i) = P(Y = 1 \\mid \\mathbf{X} = \\mathbf{x}_i)$$

A binary decision rule $h_p(\\mathbf{x}_i) \\in \\{0, 1\\}$ is then induced by comparing the estimated posterior probability against an operational classification threshold $p \\in (0, 1)$:

$$\\hat{y}_i = h_p(\\mathbf{x}_i) = \\mathbb{I}(\\hat{p}_i \\ge p)$$

where $\\mathbb{I}(\\cdot)$ denotes the indicator function. The learning objective is to optimize $f_\\theta$ to maximize threshold-independent discrimination (ROC-AUC, PR-AUC), minimize probabilistic calibration error (Brier Score), and minimize real-world financial loss under asymmetric misclassification costs.

### 3.2 The Core Research Questions (RQ1 – RQ6)
To ensure curricular alignment and academic rigor, this research investigation is structured around resolving six fundamental, interconnected research questions:

* **Research Question 1 (RQ1): Which behavioral, operational, and market attributes are significantly associated with hotel booking cancellations?**  
  *Objective:* Conduct rigorous parametric and non-parametric statistical hypothesis testing and feature importance decomposition to isolate the primary operational, demographic, and contractual drivers of cancellation, distinguishing genuine domain drivers from statistical noise.

* **Research Question 2 (RQ2): Which machine learning algorithm delivers superior predictive performance across discrimination, calibration, and computational efficiency?**  
  *Objective:* Benchmark six diverse machine learning architectures representing four foundational learning paradigms (linear/symbolic trees, bagging ensembles, gradient boosting, feedforward neural networks, and instance-based metric learning) under standardized cross-validation protocols.

* **Research Question 3 (RQ3): What is the empirical magnitude of hyperparameter optimization across distinct model families?**  
  *Objective:* Quantify the performance delta ($\Delta$) induced by systematic Bayesian and randomized hyperparameter search across model families, evaluating the bias-variance trade-off and determining which architectures possess near-optimal defaults versus those requiring extensive tuning.

* **Research Question 4 (RQ4): Can supervised feature selection reduce dimensionality without degrading predictive discrimination and accuracy?**  
  *Objective:* Implement multi-criterion feature importance ranking (LightGBM gain, Random Forest Gini, ANOVA $F$-statistic) and evaluate nested subsets ($k \\in [10, 257]$) to identify the optimal sparsity-performance Pareto frontier.

* **Research Question 5 (RQ5): How does unsupervised linear dimensionality reduction via Principal Component Analysis (PCA) impact model performance relative to supervised feature selection?**  
  *Objective:* Evaluate the mathematical and empirical consequences of projecting sparse, high-dimensional tabular data onto continuous orthogonal eigenvectors across tree-based, neural, and distance-based architectures, testing the hypothesis of "Diagonal Boundary Destruction."

* **Research Question 6 (RQ6): How robustly do the developed models generalize to unseen holdout data, and has target data leakage been completely eradicated?**  
  *Objective:* Formulate an airtight leakage-prevention pipeline, analyze cross-validation versus held-out test generalization gaps, and evaluate out-of-sample stability to verify external validity for production deployment.

### 3.3 Scope and Methodological Delimitations
This study strictly restricts predictive features to information observable at or prior to the booking event, systematically stripping all post-outcome operational updates (e.g., `reservation_status`, `reservation_status_date`) that introduce fatal target leakage. Furthermore, our evaluation extends beyond traditional symmetric loss metrics by incorporating epistemic probability calibration curves, Brier score decomposition, and empirical dollar-cost loss matrices reflecting asymmetric hospitality economics ($C_{\\text{walk}} = \\$300$ vs. $C_{\\text{empty}} = \\$102$).

---
"""

def get_section_4():
    return """## Section 4: Literature Review / Related Work

### 4.1 Foundations of Hospitality Revenue Management & Overbooking Theory
The systematic discipline of Revenue Management (RM), originating in the deregulated airline industry under the moniker of Yield Management, was formally extended to lodging by Littlewood (1972) and Belobaba (1987, 1989) via Expected Marginal Seat Revenue (EMSR) heuristics. In their seminal treatise, Talluri and van Ryzin (2004) characterized hotel revenue management as the science of allocating perishable room inventory across heterogeneous customer segments over time to maximize total expected contribution. Weatherford and Kimes (2003) conducted comprehensive taxonomy studies comparing forecasting methodologies, concluding that unconstrained demand estimation and accurate cancellation forecasting are the indispensable precursors to dynamic pricing and optimal inventory control.

In classical RM literature, cancellation management was predominantly treated through overbooking models formulated as classic newsvendor inventory problems. Under the standard single-period static overbooking formulation, the optimal overbooking level $B^*$ satisfies the critical fractile condition:

$$P(C \\le B^*) = \\frac{C_u}{C_u + C_o}$$

where $C$ is the total number of cancellations, $C_u$ is the underage cost (opportunity cost of an empty room), and $C_o$ is the overage cost (penalty of walking an arriving guest). However, as Morales and Wang (2010) demonstrated, static critical fractile formulations break down catastrophically in multi-period, highly volatile environments where cancellation distributions vary dynamically by lead time, market segment, and distribution channel.

### 4.2 Machine Learning in Hotel Cancellation Forecasting
The transition from property-level aggregate statistical forecasting to booking-level supervised machine learning was pioneered by Antonio, de Almeida, and Nunes (2017). Evaluating a prototype dataset of Portuguese hotel bookings, the authors applied Support Vector Machines (SVM), Multi-Layer Perceptrons (MLP), and Naïve Bayes classifiers, reporting that non-linear machine learning models substantially outperformed traditional logistic regression and historical averages. In subsequent investigations, Antonio, de Almeida, and Nunes (2019a, 2019b) introduced the comprehensive `hotel_bookings` dataset to the open-source community, benchmarking automated machine learning (AutoML) frameworks and highlighting lead time, country of origin, and deposit types as primary determinants of cancellation behavior.

Simultaneously, Sanchez-Medina and C-Sanchez (2020) investigated decision tree and ensemble approaches for cancellation forecasting in the Canary Islands, demonstrating that cancellation probabilities evolve dynamically across the booking horizon. Falk and Vieru (2018) conducted econometric analyses on Austrian and Finnish alpine destinations, concluding that OTAs systematically inflate cancellation propensities due to zero-cost cancellation guarantees. Despite these pioneering works, early literature in hospitality analytics suffered from notable methodological shortcomings:
1. **Target Data Leakage:** Multiple early studies inadvertently retained downstream operational fields (such as reservation status dates or post-arrival modification logs), reporting artificially inflated accuracies exceeding 98% that collapsed in live production.
2. **Symmetric Loss Assumption:** Existing studies evaluated classifiers almost exclusively using symmetric accuracy, precision, and ROC-AUC, ignoring the severe cost asymmetry between false positive overbooking errors and false negative vacant room errors.
3. **Absence of Calibration Analysis:** Hospitality researchers consistently overlooked whether predicted probability scores were epistemically calibrated, rendering probabilistic outputs unsuitable for risk-neutral mathematical programming engines.

### 4.3 Gradient Boosted Decision Trees in Tabular Machine Learning
In the broader domain of statistical learning, tabular data domains have remained stubbornly dominated by decision tree ensembles rather than deep neural architectures. Breiman (2001) revolutionized ensemble theory by introducing Random Forests, proving that bagging unpruned decision trees with random feature sub-spacing achieves substantial variance reduction without increasing asymptotic bias.

Chen and Guestrin (2016) subsequently formalized Extreme Gradient Boosting (XGBoost), incorporating second-order Taylor expansions of the loss function, weighted quantile sketches, and explicit tree regularization parameters ($\lambda, \\gamma$) that established new benchmarks across empirical predictive tasks. Building upon this, Ke et al. (2017) introduced LightGBM, which introduced two foundational algorithmic innovations: Gradient-based One-Side Sampling (GOSS), which filters instances based on gradient magnitudes, and Exclusive Feature Bundling (EFB), which bundles mutually exclusive sparse features into dense representations. Furthermore, LightGBM replaced traditional depth-wise level-by-level tree expansion with leaf-wise (best-first) tree expansion, substantially improving objective loss reduction per split. Recent extensive empirical benchmarks by Grinsztajn, Oyallon, and Varoquaux (2022) confirm that tree-based gradient boosting models consistently outperform deep neural networks on tabular benchmarks due to their rotational invariance, robustness to unscaled features, and ability to construct step-function decision boundaries across uncorrelated categorical dummies.

### 4.4 Cost-Sensitive Learning, Probability Calibration, and Dimensionality Reduction
The theoretical foundations of cost-sensitive classification were established by Elkan (2001), who proved that under known, asymmetric misclassification costs $C(\\hat{y}, y)$, the optimal decision threshold $p^*$ shifting from $p = 0.50$ is governed by the Bayes risk ratio:

$$p^* = \\frac{C(1, 0) - C(0, 0)}{C(1, 0) - C(0, 0) + C(0, 1) - C(1, 1)}$$

However, Elkan’s theorem relies fundamentally on the assumption that predicted probabilities $\\hat{p}_i$ reflect true posterior probabilities $P(Y = 1 \\mid \\mathbf{X} = \\mathbf{x}_i)$. Brier (1950) formulated the Brier Score as a strictly proper scoring rule to quantify epistemic calibration. Niculescu-Mizil and Caruana (2005) and Guo et al. (2017) demonstrated that while modern gradient boosting and neural architectures achieve exceptional discrimination (ROC-AUC), their raw probability outputs frequently exhibit systemic miscalibration (under-confidence or over-confidence) due to aggressive loss optimization and parameter regularization.

Finally, the tension between supervised feature selection and unsupervised dimensionality reduction remains a vibrant area of machine learning inquiry. While Principal Component Analysis (Jolliffe & Cadima, 2016) is mathematically optimal for global variance maximization in continuous, Gaussian-distributed domains, Guyon and Elisseeff (2003) noted that unsupervised linear projections frequently discard discriminative information present in low-variance sparse features.

### 4.5 Identified Literature Gaps & Contributions of this Study
Our review reveals four critical deficiencies in the existing body of hospitality machine learning literature:
1. **Methodological Rigor and Target Leakage Auditing:** Lack of transparent, mathematically verified preprocessing pipelines that guarantee zero information contamination between pre-arrival features and post-outcome operational status.
2. **Direct Comparison of Supervised Sparsity vs. Unsupervised PCA:** Absence of systematic empirical comparisons evaluating how linear dimensionality reduction (PCA) impacts non-linear tree-based gradient boosters versus metric learners (KNN) in tabular settings.
3. **Rigorous Epistemic Calibration Profiling:** Absence of decile-binned reliability diagrams and Brier score decomposition across multiple machine learning families in hotel cancellation tasks.
4. **Integration of Real-World Cost Asymmetries:** Total disconnect between statistical classification metrics and operational revenue management loss functions reflecting real-world guest walking costs ($C_{\\text{walk}}$) versus perishable room spoilage ($C_{\\text{empty}}$).

This paper resolves each of these deficiencies systematically, establishing an empirical benchmark for the field.

---
"""

def get_section_5():
    return """## Section 5: Dataset Description & Provenance

### 5.1 Dataset Provenance & Collection Methodology
The empirical dataset utilized in this research was curated by Antonio, de Almeida, and Nunes (2019) and deposited in *Data in Brief*. The raw corpus was directly extracted from the Property Management System (PMS) databases of two distinct hotel properties located in Portugal, capturing all historical bookings scheduled to arrive between **July 1, 2015, and August 31, 2017** (a longitudinal observation window of 26 consecutive months). The PMS database extraction was fully anonymized prior to public release to strip all Personally Identifiable Information (PII), such as customer names, credit card numbers, phone numbers, and physical addresses, in strict compliance with the European Union General Data Protection Regulation (GDPR).

The dual-property structure of the dataset provides an exceptional empirical testbed for evaluating machine learning generalization across highly divergent hospitality environments:
* **Property H1: Resort Hotel (Algarve, Portugal):** Located in the southern coastal resort region of the Algarve. This property contains 40,060 raw booking records characterized by long average lengths of stay, marked seasonal demand peaks during summer months, leisure and family demographics, and substantial lead times.
* **Property H2: City Hotel (Lisbon, Portugal):** Located in the metropolitan capital city of Lisbon. This property contains 79,330 raw booking records characterized by short business and city-break stays, year-round corporate and conference demand, high OTA intermediation, and rapid booking turnover.

The combined raw dataset consists of $N_{\\text{raw}} = 119,390$ individual reservation records and $32$ raw attributes capturing demographic, contractual, temporal, and operational dimensions.

### 5.2 Attribute Taxonomy & Domain Characterization
The 32 raw attributes can be organized into six coherent functional categories, providing a granular taxonomy of the hospitality reservation lifecycle:

| Category | Attributes Included | Domain Operational Description |
| :--- | :--- | :--- |
| **1. Target Variable** | `is_canceled` | Binary ground-truth indicator: 1 if the reservation was canceled or no-showed; 0 if the stay was successfully checked in and completed. |
| **2. Temporal & Lead Dynamics** | `lead_time`, `arrival_date_year`, `arrival_date_month`, `arrival_date_week_number`, `arrival_date_day_of_month` | Quantifies the elapsed planning horizon (days between booking and arrival) and precise temporal arrival coordinates for seasonal modeling. |
| **3. Stay Architecture** | `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`, `children`, `babies`, `meal` | Reflects physical party composition, duration of occupancy, and booked dining board packages (e.g., Bed & Breakfast, Full Board). |
| **4. Intermediation & Contract** | `market_segment`, `distribution_channel`, `agent`, `company`, `deposit_type` | Captures commercial intermediation (Direct, Corporate, Online TA, Groups), travel agency ID, corporate billing ID, and financial guarantee contracts (No Deposit, Non-Refundable, Refundable). |
| **5. Behavioral History & Loyalty**| `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled` | Historical track record of customer interactions, quantifying past cancellation delinquency versus established loyalty. |
| **6. Operational Allocation & Preferences** | `reserved_room_type`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`, `customer_type`, `adr`, `required_car_parking_spaces`, `total_of_special_requests`, `reservation_status`, `reservation_status_date` | Financial rate metrics (Average Daily Rate), customer classification (Transient, Group), operational resource requests (parking, baby cribs), and post-booking room assignment dynamics. |

### 5.3 Complete Variable Specification Table
To establish exhaustive documentation for reproducibility, Table 5.1 details all 32 raw variables, their underlying data types, permissible domains, and operational interpretations.

**Table 5.1: Comprehensive Raw Attribute Dictionary & Domain Descriptions**

| Attribute Name | Storage Type | Permissible Domain / Range | Operational & Behavioral Meaning |
| :--- | :--- | :--- | :--- |
| `hotel` | Categorical (str) | {'Resort Hotel', 'City Hotel'} | Flag distinguishing coastal resort property from metropolitan business hotel. |
| `is_canceled` | Integer (binary) | {0, 1} | Target variable: 1 = Canceled/No-Show, 0 = Checked-in and completed. |
| `lead_time` | Integer | [0, 737] days | Elapsed days between transaction booking date and scheduled arrival date. |
| `arrival_date_year` | Integer | [2015, 2017] | Calendar year of scheduled check-in. |
| `arrival_date_month` | Categorical (str) | {January, ..., December} | Calendar month of scheduled check-in. |
| `arrival_date_week_number` | Integer | [1, 53] | ISO week number of scheduled check-in. |
| `arrival_date_day_of_month` | Integer | [1, 31] | Day of the month of scheduled check-in. |
| `stays_in_weekend_nights` | Integer | [0, 19] nights | Number of weekend nights (Saturday, Sunday) reserved. |
| `stays_in_week_nights` | Integer | [0, 50] nights | Number of weekday nights (Monday through Friday) reserved. |
| `adults` | Integer | [0, 55] guests | Number of adult occupants listed on the reservation. |
| `children` | Float (numeric) | [0.0, 10.0] guests | Number of paying child occupants. |
| `babies` | Integer | [0, 10] guests | Number of infant occupants (crib-eligible). |
| `meal` | Categorical (str) | {BB, FB, HB, SC, Undefined} | Board package: BB (Bed & Breakfast), HB (Half Board), FB (Full Board), SC (Self-Catering). |
| `country` | Categorical (str) | ISO 3166-1 alpha-3 (177 codes) | Country of origin of the primary guest. |
| `market_segment` | Categorical (str) | {Direct, Corporate, Online TA, Offline TA/TO, Complementary, Groups, Aviation} | Market booking channel designation. |
| `distribution_channel` | Categorical (str) | {Direct, Corporate, TA/TO, Undefined, GDS} | Commercial intermediary infrastructure used to transmit reservation. |
| `is_repeated_guest` | Integer (binary) | {0, 1} | PMS loyalty flag: 1 if guest has a completed prior stay on record; 0 otherwise. |
| `previous_cancellations` | Integer | [0, 26] events | Count of historical bookings canceled by this customer prior to the current booking. |
| `previous_bookings_not_canceled` | Integer | [0, 72] stays | Count of historical bookings successfully completed by this customer prior to current booking. |
| `reserved_room_type` | Categorical (str) | {A, B, C, D, E, F, G, H, L, P} | Anonymized room category code selected by guest at booking. |
| `assigned_room_type` | Categorical (str) | {A, B, C, D, E, F, G, H, I, K, L, P} | Anonymized room category code allocated by PMS/front-desk at check-in. |
| `booking_changes` | Integer | [0, 21] events | Number of modifications/amendments made to the reservation prior to arrival. |
| `deposit_type` | Categorical (str) | {No Deposit, Non Refund, Refundable} | Financial deposit condition securing the booking. |
| `agent` | Float (numeric) | Categorical ID [1.0, 535.0] | Anonymized identifier of the travel agency that made the booking. |
| `company` | Float (numeric) | Categorical ID [6.0, 543.0] | Anonymized identifier of the corporate account entity paying/booking. |
| `days_in_waiting_list` | Integer | [0, 391] days | Number of days the booking sat on a waiting list before confirmation. |
| `customer_type` | Categorical (str) | {Transient, Transient-Party, Contract, Group} | Customer relationship classification. |
| `adr` | Float (numeric) | [-6.38, 5400.0] USD/EUR | Average Daily Rate: Total room revenue divided by total nights stayed. |
| `required_car_parking_spaces` | Integer | [0, 8] vehicles | Number of on-site parking spaces requested by the guest. |
| `total_of_special_requests` | Integer | [0, 5] requests | Count of non-standard operational requests (e.g., high floor, twin beds, late check-in). |
| `reservation_status` | Categorical (str) | {Check-Out, Canceled, No-Show} | Terminal PMS operational status (**Target Leakage - Dropped**). |
| `reservation_status_date` | Categorical (str) | YYYY-MM-DD | Date on which terminal status was recorded (**Target Leakage - Dropped**). |

---
"""
