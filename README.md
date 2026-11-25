# Hotel Booking Cancellation Prediction

SEN22436E / BLM22436E **Introduction to Data Science** – final project, Fatih Sultan Mehmet Vakıf University.

**Team:** Yunus Emre Coşkun (2021221055) · Muhammed Eren Koçkan (2121221022) · Abdelfatah M A Alhoot (2321051372)
**Instructor:** Dr. Öğr. Üyesi Abdulkerim Mohammed Yibre

We predict whether a hotel booking will be cancelled (`is_canceled`) from the information known when the booking is made, compare **six machine-learning algorithms** before and after **hyper-parameter optimisation**, and study **feature selection** and **PCA**.

| Deliverable | File |
|---|---|
| Final report (sections I–VI of the guideline) | `report/Final_Report.pdf` / `.docx` |
| Final presentation | `report/Final_Presentation.pdf` / `.pptx` |
| Code | `notebooks/01 … 10` (executed, with outputs) |
| Live demo | `demo/app.py` |

---

## 1. Dataset
* **Hotel Booking Demand** – Antonio, de Almeida & Nunes (2019), *Data in Brief* 22, 41-49; Kaggle (jessemostipak). File: `data/raw/hotel_bookings.csv`.
* 119,390 bookings × 32 attributes of a City Hotel (Lisbon) and a Resort Hotel (Algarve), arrivals July 2015 – August 2017.
* Target: `is_canceled` – 37.0% cancelled, 63.0% not cancelled.

## 2. Project structure
```text
├── data/raw/                 hotel_bookings.csv
├── data/processed/           80/20 train/test matrices written by notebook 02
├── models/                   preprocessor + default and tuned models (created by notebooks 02-08)
├── notebooks/                01-10, run in order
├── results/<notebook>/       every table (CSV) and chart (PNG) of each notebook
├── report/                   final report + final presentation (+ progress presentation)
├── demo/app.py               desktop demo: cancellation probability of a single booking
├── src/                      helper modules (src/preprocessing.py reproduces notebook 02 exactly)
└── tools/                    notebook builder/runner, report & slide generators
```

## 3. Workflow and results (all numbers on the 20% test set, 23,842 bookings)

### 01 – Exploratory data analysis
Dimensionality, feature dictionary (meaning, dtype, distinct values) and descriptive statistics of **all 32 attributes**, target distribution, missing values and **10 charts**.
Key findings: City Hotel 41.7% vs Resort 27.8% cancelled · **Non Refund deposit 99.4%** cancelled (97% Portuguese group blocks) · median lead time 113 days (cancelled) vs 45 (kept) · no special request 47.7% vs ~22% · parking request 0% · Portugal 56.6%.

### 02 – Preprocessing
* **Missing values** (company 94.31%, agent 13.69%, country 0.41%, children 4 rows; all other attributes 0%) → filled with 0 / 'Unknown' because NaN means "no company / no agency".
* **Errors removed:** 180 ghost bookings (0 guests), 2 impossible prices (−6.38 and 5,400 EUR) → 119,208 rows.
* **Outliers** detected with the **IQR rule**, the **Z-score rule** and **Isolation Forest**; real rare values are kept (they are strong cancellation signals, trees are robust).
* **Leakage:** `reservation_status` and its date dropped. 5 engineered features.
* Stratified 80/20 split → **StandardScaler** (24 numerical) + **OneHotEncoder** (10 categorical) → **255 predictors**.

### 03–08 – Six algorithms, default vs tuned (`GridSearchCV`, 3-fold CV, ROC-AUC)

| Model | Best parameters | Accuracy (default → tuned) | F1 (default → tuned) | ROC-AUC (default → tuned) |
|---|---|:-:|:-:|:-:|
| KNN | k=15, weights=distance | 83.61% → 86.03% | 0.7738 → 0.8046 | 0.8998 → 0.9309 |
| Decision Tree | max_depth=16, min_samples_leaf=8 | 86.52% → 86.71% | 0.8198 → 0.8151 | 0.8597 → 0.9387 |
| Random Forest | 150 trees, max_depth=None | 89.20% → 89.38% | 0.8472 → 0.8499 | 0.9590 → 0.9594 |
| XGBoost | lr=0.2, max_depth=6, 150 trees | 88.12% → 88.11% | 0.8352 → 0.8353 | 0.9539 → 0.9542 |
| **LightGBM** | lr=0.2, num_leaves=127, 150 trees | 87.81% → **89.37%** | 0.8306 → **0.8539** | 0.9516 → **0.9623** |
| MLP | (64, 32), alpha=0.001 | 87.12% → 87.01% | 0.8178 → 0.8206 | 0.9443 → 0.9447 |

### 10 – Final leaderboard (tuned models)

| Rank | Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | **LightGBM** | 89.37% | 87.01% | **83.83%** | 92.63% | **0.8539** | **0.9623** |
| 2 | Random Forest | **89.38%** | **89.34%** | 81.04% | **94.30%** | 0.8499 | 0.9594 |
| 3 | XGBoost | 88.11% | 85.85% | 81.33% | 92.10% | 0.8353 | 0.9542 |
| 4 | MLP | 87.01% | 84.07% | 80.15% | 91.05% | 0.8206 | 0.9447 |
| 5 | Decision Tree | 86.71% | 84.19% | 79.00% | 91.25% | 0.8151 | 0.9387 |
| 6 | KNN | 86.03% | 83.55% | 77.59% | 91.00% | 0.8046 | 0.9309 |

### 09 – Feature selection & PCA (tuned models, F1-score)

| Setting | Features | Decision Tree | KNN | Random Forest | LightGBM |
|---|:-:|:-:|:-:|:-:|:-:|
| All features | 255 | 0.8151 | 0.8046 | 0.8499 | 0.8539 |
| Variance threshold (filter) | 115 | 0.8149 | 0.8039 | 0.8540 | 0.8541 |
| ANOVA F-test (filter) | 50 | 0.8023 | 0.8116 | 0.8339 | 0.8305 |
| Mutual information (filter) | 50 | 0.8170 | **0.8130** | 0.8540 | 0.8487 |
| RFE – Decision Tree (wrapper) | 50 | 0.8134 | 0.8043 | **0.8574** | 0.8511 |
| L1 logistic regression (embedded) | 44 | 0.8030 | 0.8113 | 0.8459 | 0.8422 |
| LightGBM importance (embedded) | 50 | 0.8117 | 0.8078 | 0.8572 | 0.8526 |
| PCA 25 comp. (90% var.) | 25 | 0.7595 | 0.7951 | 0.8181 | 0.8176 |
| PCA 35 comp. (95% var.) | 35 | 0.7618 | 0.8001 | 0.8220 | 0.8230 |

* **Boosted:** RFE raised Random Forest to F1 **0.8574** (best F1 of the project) with 80% fewer features; mutual information / ANOVA / L1 boosted KNN. LightGBM does not need feature selection.
* **PCA lowered every model** (unsupervised, mixes clear one-hot rules) → feature selection is the better reduction for this data.

## 4. How to run
```bash
pip install -r requirements.txt
jupyter notebook            # run notebooks/01 … 10 in order
python demo/app.py          # demo (needs the models written by notebooks 02-08)
```
Or headless: `python tools/run_notebooks.py 01 02 03 04 05 06 07 08 09 10`.
The report and slides are regenerated from `results/` with `python tools/make_report.py`, `python tools/make_slides.py` and `tools/export_office.ps1` (PDF export, needs MS Office).
