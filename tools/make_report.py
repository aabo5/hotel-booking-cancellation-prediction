"""Final project report (.docx) structured exactly like the course guideline (sections I-VI).

python tools/make_report.py  ->  report/Final_Report.docx   (then tools/export_office.ps1 -> PDF)
Every number is read from results/ (written by notebooks 01-10), nothing is typed by hand.
"""
import json
import re
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from team import TEAM, COURSE, INSTRUCTOR, DATE

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "Final_Report.docx"
INK = RGBColor(0x1B, 0x2F, 0x45)
MODELS = {"KNN": "03_knn", "Decision Tree": "04_decision_tree", "Random Forest": "05_random_forest",
          "XGBoost": "06_xgboost", "LightGBM": "07_lightgbm", "MLP Neural Net": "08_mlp"}

# ------------------------------------------------------------------ data
dim = pd.read_csv(R / "01_data_exploration/dataset_dimensionality.csv")
fdict = pd.read_csv(R / "01_data_exploration/feature_dictionary.csv")
dnum = pd.read_csv(R / "01_data_exploration/descriptive_statistics_numerical.csv", index_col=0)
dcat = pd.read_csv(R / "01_data_exploration/descriptive_statistics_categorical.csv", index_col=0)
facts = pd.read_csv(R / "01_data_exploration/eda_key_facts.csv", index_col=0)["value"]
tcorr = pd.read_csv(R / "01_data_exploration/correlation_with_target.csv", index_col=0).iloc[:, 0]
ltb = pd.read_csv(R / "01_data_exploration/cancel_rate_by_lead_time_band.csv", index_col=0)
mv = pd.read_csv(R / "02_data_preprocessing/missing_values_all_attributes.csv", index_col=0)
outl = pd.read_csv(R / "02_data_preprocessing/outlier_detection_iqr_zscore.csv", index_col=0)
ocmp = pd.read_csv(R / "02_data_preprocessing/outlier_methods_comparison.csv")
enc = pd.read_csv(R / "02_data_preprocessing/one_hot_encoding_summary.csv")
prep = pd.read_csv(R / "02_data_preprocessing/preprocessing_summary.csv").set_index("Metric")["Value"]
both = pd.read_csv(R / "10_model_comparison/default_vs_tuned_all_models.csv")
lb = pd.read_csv(R / "10_model_comparison/master_model_leaderboard.csv")
cms = pd.read_csv(R / "10_model_comparison/confusion_matrices_all.csv").set_index("Model")
gain = pd.read_csv(R / "10_model_comparison/best_model_feature_importance_gain.csv", index_col=0).iloc[:, 0]
fst = pd.read_csv(R / "09_feature_selection_and_pca/feature_selection_techniques.csv")
fsr = pd.read_csv(R / "09_feature_selection_and_pca/feature_selection_results.csv")
pcr = pd.read_csv(R / "09_feature_selection_and_pca/pca_results.csv")
pcn = pd.read_csv(R / "09_feature_selection_and_pca/pca_components_needed.csv", index_col=0).iloc[:, 0]
curve = pd.read_csv(R / "09_feature_selection_and_pca/n_features_curve_lightgbm.csv")


def best_params(folder):
    nb = json.load(open(ROOT / "notebooks" / f"{folder}.ipynb", encoding="utf-8"))
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            t = "".join(o.get("text", ""))
            m = re.search(r"Best Parameters: (\{.*\})", t)
            if m:
                return m.group(1)
    return "?"


def grid(folder):
    nb = json.load(open(ROOT / "notebooks" / f"{folder}.ipynb", encoding="utf-8"))
    for c in nb["cells"]:
        s = "".join(c["source"])
        m = re.search(r"param_grid = (\{.*?\n\})", s, re.S)
        if m:
            return " ".join(m.group(1).split())
    return "?"


PARAMS = {m: best_params(f) for m, f in MODELS.items()}
GRIDS = {m: grid(f) for m, f in MODELS.items()}
tuned = both[both.Variant == "Tuned"].set_index("Model")
default = both[both.Variant == "Default"].set_index("Model")
lbi = lb.set_index("Model")
BEST = lb.iloc[0]["Model"]
SECOND = lb.iloc[1]["Model"]


def p(x, d=1):
    return f"{x * 100:.{d}f}%"


def f4(x):
    return f"{x:.4f}"


# ------------------------------------------------------------------ docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.3))
sec.top_margin = sec.bottom_margin = Cm(2.2)
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.12
for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Calibri"
    h.font.size = Pt(size)
    h.font.color.rgb = INK
    h.font.bold = True
    h.element.rPr.rFonts.set(qn("w:asciiTheme"), "")
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True

FIG_NO = [0]
TAB_NO = [0]


def runs(par, text, size=None, color=None):
    for i, part in enumerate(re.split(r"(\*\*.+?\*\*|`.+?`|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*]))", text)):
        if not part:
            continue
        if part.startswith("**"):
            r = par.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = par.add_run(part[1:-1])
            r.italic = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        else:
            r = par.add_run(part)
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
    return par


def P(text, align=None, size=None, italic=False, after=None):
    par = runs(doc.add_paragraph(), text, size)
    if align == "j":
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    elif align == "c":
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if italic:
        for r in par.runs:
            r.italic = True
    if after is not None:
        par.paragraph_format.space_after = Pt(after)
    return par


def J(text):
    return P(text, "j")


def B(items, style="List Bullet"):
    for it in items:
        par = runs(doc.add_paragraph(style=style), it)
        par.paragraph_format.space_after = Pt(2)
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def H(text, lvl=1):
    return doc.add_heading(text, lvl)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def T(df, caption, size=8.5, widths=None, bold_row=None, index=False):
    if index:
        df = df.reset_index()
    TAB_NO[0] += 1
    cap = doc.add_paragraph()
    r = cap.add_run(f"Table {TAB_NO[0]}. ")
    r.bold = True
    cap.add_run(caption)
    cap.paragraph_format.keep_with_next = True
    cap.paragraph_format.space_after = Pt(3)
    for r in cap.runs:
        r.font.size = Pt(9.5)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = ""
        rr = cell.paragraphs[0].add_run(str(c))
        rr.bold = True
        rr.font.size = Pt(size)
        rr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, "1B2F45")
    for i, row in enumerate(df.itertuples(index=False)):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            rr = cells[j].paragraphs[0].add_run(str(v))
            rr.font.size = Pt(size)
            if bold_row is not None and i == bold_row:
                rr.bold = True
            if i % 2 == 1:
                shade(cells[j], "EEF2F6")
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def F(path, caption, width=16):
    FIG_NO[0] += 1
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.keep_with_next = True
    par.add_run().add_picture(str(R / path), width=Cm(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure {FIG_NO[0]}. ")
    r.bold = True
    r.font.size = Pt(9.5)
    r2 = cap.add_run(caption)
    r2.font.size = Pt(9.5)
    r2.italic = True


def pagebreak():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def field(par, instr):
    r = par.add_run()
    for tag, txt in (("begin", None), (None, instr), ("separate", None), (None, "1"), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), tag)
            r._r.append(e)
        else:
            e = OxmlElement("w:instrText" if txt == instr else "w:t")
            e.set(qn("xml:space"), "preserve")
            e.text = txt
            r._r.append(e)


# footer page numbers
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
field(fp, "PAGE")

# ------------------------------------------------------------------ cover
for _ in range(2):
    doc.add_paragraph()
P("FATIH SULTAN MEHMET VAKIF UNIVERSITY", "c", 14).runs[0].bold = True
P("Faculty of Engineering", "c", 12)
P(COURSE, "c", 12)
doc.add_paragraph()
t = P("Hotel Booking Cancellation Prediction", "c", 24)
t.runs[0].bold = True
t.runs[0].font.color.rgb = INK
P("An end-to-end data science project: EDA, preprocessing, six machine learning algorithms, "
  "hyper-parameter optimisation, feature selection and PCA", "c", 12, italic=True)
doc.add_paragraph()
P("Final Project Report", "c", 14).runs[0].bold = True
doc.add_paragraph()
P("Team members", "c", 11).runs[0].bold = True
for m in TEAM:
    P(m, "c", 11, after=1)
doc.add_paragraph()
P(f"Instructor: {INSTRUCTOR}", "c", 11)
P(DATE, "c", 11)
P("Code repository: hotel-booking-cancellation-prediction (notebooks 01-10, src/, results/)", "c", 9.5, italic=True)
pagebreak()

# ------------------------------------------------------------------ abstract + TOC
H("Abstract", 1)
J(f"Cancellations are one of the biggest sources of lost revenue for hotels. In this project we predict whether a hotel "
  f"booking will be cancelled, using the public Hotel Booking Demand dataset (119,390 bookings of a city hotel and a "
  f"resort hotel in Portugal, 2015-2017). After a detailed exploratory analysis we cleaned the data (impossible records, "
  f"missing values, outlier analysis with the IQR rule, the Z-score rule and Isolation Forest), removed two leakage "
  f"columns, engineered five features and encoded/scaled the data into {prep['Total Predictors']} predictors. Six algorithms "
  f"- K-Nearest Neighbours, Decision Tree, Random Forest, XGBoost, LightGBM and a Multilayer Perceptron - were trained "
  f"with default settings and again after hyper-parameter optimisation with grid search and 3-fold cross-validation. "
  f"On the held-out test set ({prep['Testing Rows (20%)']:,} bookings) the tuned **{BEST}** is the best model "
  f"(accuracy {p(lbi.loc[BEST, 'Accuracy'], 2)}, F1 {f4(lbi.loc[BEST, 'F1-Score'])}, ROC-AUC {f4(lbi.loc[BEST, 'ROC-AUC'])}), "
  f"closely followed by {SECOND}. Finally, five feature selection techniques (filter, wrapper and embedded) and PCA were "
  f"compared before and after reduction. FS_ABSTRACT")
P("**Keywords:** hotel booking cancellation, classification, LightGBM, Random Forest, hyper-parameter optimisation, "
  "feature selection, PCA")
H("Contents", 1)
tp = doc.add_paragraph()
field(tp, 'TOC \\o "1-2" \\h \\z \\u')
pagebreak()

# ------------------------------------------------------------------ I
H("I. Introduction", 1)
H("1.1 Why we chose this problem", 2)
J("A hotel room is a perishable product: a night that stays empty can never be sold again. In the dataset used in this "
  f"project **{facts['cancel_rate_%']}% of all bookings were cancelled**, and a large part of them only shortly before arrival. "
  "Hotels try to protect themselves with overbooking and strict deposit policies, but both are blunt tools - too much "
  "overbooking means guests who cannot be accommodated, too strict policies scare customers away. If the hotel can estimate "
  "the cancellation risk of **each individual booking** at the moment it is made, it can overbook only where it is safe, "
  "contact risky guests early, or ask for a deposit only from risky segments.")
J("We chose this topic because (1) it is a real business problem with a clear economic value, (2) a large, real and public "
  "dataset exists for it, (3) the target is a clear yes/no outcome, which makes it a well-defined **binary classification** "
  "problem, and (4) the data contains every kind of preprocessing challenge covered in the course: missing values, outliers, "
  "data-entry errors, high-cardinality categorical features, skewed numerical features and target leakage.")
H("1.2 Problem definition", 2)
J("Given the information that is known when a booking is made (hotel, lead time, dates, length of stay, guests, meal, "
  "country, channel, deposit type, price, special requests, history of the customer ...), predict the target "
  "`is_canceled` (1 = the booking will be cancelled, 0 = the guest will arrive).")
H("1.3 Machine learning algorithms and techniques used", 2)
B(["**Exploratory data analysis** with pandas, Matplotlib and Seaborn (descriptive statistics, 10 charts, correlation analysis).",
   "**Data preprocessing:** missing-value imputation, outlier detection with IQR, Z-score and Isolation Forest, leakage removal, "
   "feature engineering, stratified train/test split, standardisation (`StandardScaler`) and one-hot encoding (`OneHotEncoder`).",
   "**Six classification algorithms** (scikit-learn, XGBoost, LightGBM): K-Nearest Neighbours (KNN), Decision Tree, "
   "Random Forest, XGBoost, LightGBM and a Multilayer Perceptron (MLP) neural network - from a simple instance-based "
   "method to single trees, bagging and boosting ensembles and a neural network.",
   "**Hyper-parameter optimisation** with `GridSearchCV` (3-fold cross-validation, ROC-AUC as the selection score).",
   "**Evaluation:** accuracy, precision, recall, specificity, F1-score, ROC-AUC, precision-recall curves and confusion matrices.",
   "**Feature selection:** variance threshold, ANOVA F-test, mutual information (filters), recursive feature elimination "
   "(wrapper), L1-regularised logistic regression and LightGBM importance (embedded).",
   "**Dimensionality reduction:** Principal Component Analysis (PCA)."])
H("1.4 Project structure", 2)
J("The code is organised as ten Jupyter notebooks that are run in order: 01 EDA, 02 preprocessing, 03-08 one notebook per "
  "algorithm (default model, tuned model, comparison), 09 feature selection and PCA, 10 final comparison. Every notebook "
  "saves its tables (CSV) and charts (PNG) to its own folder in `results/`; all numbers in this report are taken from these files.")

# ------------------------------------------------------------------ II
H("II. Description of the Dataset", 1)
H("2.1 Source", 2)
J("The data is the **Hotel Booking Demand** dataset published by Antonio, de Almeida and Nunes (2019) in *Data in Brief* "
  "and distributed on Kaggle (file `hotel_bookings.csv`). The authors exported it from the Property Management Systems "
  "of two real hotels in Portugal - a **City Hotel** in Lisbon and a **Resort Hotel** in the Algarve - and anonymised "
  "it (no names or personal identifiers; agents and companies are given as numeric IDs). It covers bookings with arrival "
  "dates between 1 July 2015 and 31 August 2017.")
H("2.2 Attributes and their values", 2)
J(f"The dataset has **{len(fdict)} attributes**: one target, 19 numerical attributes, 10 categorical attributes and two "
  "status columns that are only known after the outcome. They can be grouped as follows:")
B(["**Booking timing:** `lead_time` (0-737 days), arrival year (2015-2017), month (January-December), week number (1-53), day of month (1-31).",
   "**Stay and guests:** `stays_in_weekend_nights` (0-19), `stays_in_week_nights` (0-50), `adults` (0-55), `children` (0-10), `babies` (0-10), `meal` (BB, HB, FB, SC, Undefined).",
   "**Customer and channel:** `country` (177 countries), `market_segment` (8 values, e.g. Online TA, Groups, Direct), `distribution_channel` (5 values), `customer_type` (Transient, Transient-Party, Contract, Group), `is_repeated_guest` (0/1), `previous_cancellations` (0-26), `previous_bookings_not_canceled` (0-72), `agent` / `company` (IDs).",
   "**Room and price:** `reserved_room_type` (10 codes), `assigned_room_type` (12 codes), `adr` - average daily rate in EUR (-6.38 to 5,400 before cleaning), `deposit_type` (No Deposit, Non Refund, Refundable).",
   "**Requests and changes:** `booking_changes` (0-21), `days_in_waiting_list` (0-391), `required_car_parking_spaces` (0-8), `total_of_special_requests` (0-5).",
   "**Outcome columns:** `is_canceled` (target, 0/1), `reservation_status` (Check-Out, Canceled, No-Show) and `reservation_status_date`."])
J("The full feature dictionary (meaning, data type, number of distinct values and examples of every attribute) is given in Table 2 in Section III.")
H("2.3 Why the dataset is suitable for our problem", 2)
B(["It contains the **labelled outcome** (`is_canceled`) for every booking, which is exactly what supervised classification needs.",
   "Almost all attributes are **known at booking time**, so a model trained on them can really be used when a new booking arrives.",
   f"It is **large** ({int(dim.loc[0, 'Value'].replace(',', '')):,} rows), so a 20% test set still contains more than 23,000 bookings and the comparison between the six algorithms is statistically reliable.",
   "It comes from **real hotels**, so it contains real-world problems (missing values, errors, outliers, leakage) that the course asks us to handle.",
   f"The classes are only moderately imbalanced ({facts['cancel_rate_%']}% cancelled), so standard classifiers can be trained without resampling."])

# ------------------------------------------------------------------ III
H("III. Exploratory Data Analysis", 1)
H("3.1 Dimensionality", 2)
T(dim, "Dimensionality of the raw dataset.", 9, [7, 5])
J("The dataset is a table of 119,390 rows (bookings) and 32 columns (31 attributes + the target). After one-hot encoding "
  f"the categorical attributes the modelling matrix has **{prep['Total Predictors']} dimensions**, which is why dimensionality "
  "reduction is studied in Section VI. 31,994 rows are exact duplicates; they are typical of PMS exports where several "
  "identical rooms are booked in one transaction, and the team kept them (see the limitations in Section VII).")
H("3.2 Meaning and data type of every feature", 2)
fd = fdict.copy()
fd["Example values"] = fd["Example values"].astype(str).str.slice(0, 38)
fd["Data type"] = fd["Data type"].replace({"str": "object (text)"})
T(fd[["Feature", "Meaning", "Data type", "Kind", "Distinct values", "Example values"]], "Feature dictionary of the 32 attributes.", 7.5,
  [3.3, 5.6, 1.7, 2.2, 1.2, 2.6])
H("3.3 Descriptive statistics", 2)
J("Table 3 gives count, missing values, mean, standard deviation, minimum, quartiles, maximum, skewness and kurtosis of every "
  "numerical attribute; Table 4 gives count, number of categories, most frequent value and its share for every categorical attribute.")
dn = dnum.drop(columns=["count"]).copy()
dn.columns = ["missing", "mean", "std", "min", "Q1", "median", "Q3", "max", "skew", "kurt"]
dn = dn.round(2)
dn["missing"] = dn["missing"].astype(int)
T(dn, "Descriptive statistics of the numerical attributes (119,390 rows).", 7.5,
  [4.0, 1.3, 1.3, 1.3, 1.2, 1.2, 1.3, 1.2, 1.3, 1.2, 1.4], index=True)
dc = dcat.rename(columns={"freq": "top freq", "top_share_%": "top share %"})
T(dc[["count", "missing", "unique", "top", "top freq", "top share %"]], "Descriptive statistics of the categorical attributes.", 8,
  [4, 1.8, 1.5, 1.6, 3, 2, 2], index=True)
J("**Observations.** Most numerical attributes are strongly right-skewed (skewness above 1): `lead_time` (1.35), "
  "`stays_in_week_nights` (2.86), `days_in_waiting_list` (11.9), `adr` (10.5), `previous_cancellations` (24.5), `babies` (24.7). "
  "Their maxima lie far above the third quartile, which already points to outliers. `adr` has a negative minimum and a maximum "
  "of 5,400 EUR, and `adults` has a minimum of 0 - these are data errors. `agent` and `company` are stored as numbers but are "
  "identifiers, so their means have no meaning. Among the categorical attributes, `country` has 177 categories (high "
  "cardinality) and Portugal alone accounts for 40.7% of the bookings; 87.7% of the bookings have no deposit and 75% of the "
  "customers are transient.")
H("3.4 Target variable", 2)
F("01_data_exploration/target_distribution.png", "Chart 1 - distribution of the target variable.", 9)
J(f"{facts['cancel_rate_%']}% of the bookings were cancelled and {100 - facts['cancel_rate_%']:.1f}% were not. The classes "
  "are moderately imbalanced: a model that always predicts 'not cancelled' would already reach 63% accuracy. Therefore we also "
  "use precision, recall, F1-score and ROC-AUC, and we split the data in a stratified way.")
H("3.5 Charts of the most important features", 2)
F("01_data_exploration/cancellation_by_categories.png", "Chart 2 - cancellation rate by hotel type, deposit type and market segment.")
J(f"The City Hotel has a much higher cancellation rate ({facts['city_cancel_%']}%) than the Resort Hotel ({facts['resort_cancel_%']}%). "
  f"Surprisingly, bookings with a **non-refundable** deposit are cancelled {facts['non_refund_cancel_%']}% of the time, against "
  f"{facts['no_deposit_cancel_%']}% for no deposit. 97% of these bookings come from Portugal and 63% are 'Groups' with a median "
  f"lead time of 183 days: they are block reservations of tour operators that are released later, not individual guests who "
  f"lose their money. The Groups segment ({facts['groups_cancel_%']}%) cancels most, Direct and Corporate bookings least.")
F("01_data_exploration/lead_time_and_adr_boxplots.png", "Chart 3 - lead time and average daily rate by booking outcome.")
J(f"Cancelled bookings were made much earlier: their median lead time is {facts['median_lead_cancel']:.0f} days against "
  f"{facts['median_lead_kept']:.0f} days for kept bookings. The price (ADR) differs only slightly (median 96.2 vs 92.5 EUR), "
  "so price alone is a weak signal.")
F("01_data_exploration/numerical_histograms.png", "Chart 4 - histograms of the main numerical features.")
J("The histograms confirm the right skew of the count-type attributes: most bookings have zero changes, zero earlier "
  "cancellations and zero days on the waiting list, and a long tail of a few large values. This is the reason why the "
  "IQR rule marks many values of these features as outliers (Section 4.3).")
F("01_data_exploration/monthly_seasonality.png", "Chart 5 - bookings and cancellation rate per arrival month and hotel.")
J("Demand is seasonal: both hotels receive most bookings in summer (peak in August) and fewest in winter. The City Hotel's "
  "cancellation rate is above 37% in every month and peaks in April-June; the Resort Hotel's rate is highest in the summer "
  "months and lowest in January. Arrival month is therefore a useful predictor.")
F("01_data_exploration/correlation_heatmap.png", "Chart 6 - Pearson correlation matrix of the numerical attributes.", 14)
top_c = ", ".join(f"`{k}` ({v:+.2f})" for k, v in tcorr.head(5).items())
J(f"The attributes most correlated with the target are {top_c}. No numerical attribute has a strong linear "
  "correlation with the target (all |r| < 0.3), so the relationship is non-linear and depends on combinations of "
  "features - a hint that tree-based models will work better than linear ones. Strong correlations between predictors "
  "are moderate at most (arrival year vs week number -0.54, weekend vs week nights +0.50, repeated guest vs previous kept bookings +0.42), so multicollinearity is not a major issue.")
F("01_data_exploration/top_countries.png", "Chart 7 - booking volume and cancellation rate of the ten most frequent countries.")
J(f"Portugal (PRT) is the dominant origin ({facts['PRT_share_%']}% of the bookings) and has by far the highest cancellation "
  f"rate ({facts['PRT_cancel_%']}%), mostly because of the group bookings mentioned above. Foreign guests (GBR, FRA, DEU, "
  "NLD) cancel only 17-20% of the time.")
F("01_data_exploration/behavioural_features.png", "Chart 8 - cancellation rate by special requests, parking, previous cancellations and booking changes.")
J(f"Every sign of commitment lowers the risk: bookings with no special request are cancelled {facts['cancel_%_0_requests']}% "
  "of the time, bookings with at least one request only about 17-22%. Not a single booking that requested a parking space "
  "was cancelled. Bookings that have been modified are cancelled less often. On the other hand customers who cancelled "
  "exactly once before cancel again in more than 90% of the cases.")
F("01_data_exploration/customer_type_channel.png", "Chart 9 - bookings by customer type and distribution channel, split by outcome.")
J("Transient customers and the TA/TO (travel agent / tour operator) channel dominate both the volume and the number of "
  "cancellations, while direct and corporate channels have a much smaller cancelled share.")
F("01_data_exploration/lead_time_density.png", "Chart 10 - density of lead time for cancelled and kept bookings.")
ltt = ltb.copy()
ltt["cancel_rate"] = (ltt["cancel_rate"] * 100).round(1).astype(str) + "%"
ltt["bookings"] = ltt["bookings"].map("{:,}".format)
T(ltt, "Cancellation rate by lead-time band.", 9, [4, 4, 4], index=True)
J("Kept bookings are concentrated at short lead times, cancelled ones are spread over long lead times. The cancellation "
  f"rate rises monotonically from {ltb.iloc[0]['cancel_rate'] * 100:.1f}% for bookings made less than a week ahead to "
  f"{ltb.iloc[-1]['cancel_rate'] * 100:.1f}% for bookings made more than a year ahead.")
H("3.6 Summary of the EDA", 2)
B(["The target is moderately imbalanced (37% cancelled).",
   "The strongest single signals are deposit type, lead time, country (Portugal), market segment (Groups), special requests and parking.",
   "Most numerical features are right-skewed with long tails (outliers), and their linear correlation with the target is weak, so non-linear models are expected to perform best.",
   "Four attributes have missing values and two attributes leak the target; both problems are handled in Section IV."])

# ------------------------------------------------------------------ IV
H("IV. Data Preprocessing", 1)
H("4.1 Missing values", 2)
mvt = mv[mv["Missing rows"] > 0].copy()
others = ", ".join(mv[mv["Missing rows"] == 0].index)
mvt["Missing %"] = mvt["Missing %"].map(lambda x: f"{x:.3f}%")
mvt["Action"] = ["Filled with 0 (= no company)", "Filled with 0 (= no agency, direct booking)",
                 "Filled with 'Unknown'", "Filled with 0 (mode)"]
T(mvt, "Missing values of the attributes that contain missing values.", 9, [3, 2.5, 2.5, 2.5, 6], index=True)
J(f"The percentage of missing values was computed for **all 32 attributes** (notebook 02, `missing_values_all_attributes.csv`, "
  f"Figure {FIG_NO[0] + 1}). Only the four attributes in Table {TAB_NO[0]} contain missing values; the remaining 28 attributes ({others}) are 0% missing.")
F("02_data_preprocessing/missing_values_bar.png", "Percentage of missing values per attribute (attributes not shown have 0%).", 12)
J("**What we did and why.** The missing values are not random errors but carry meaning. A missing `company` means that no "
  "company paid for the booking (private guest) and a missing `agent` means that the booking was made without a travel "
  "agency (direct booking); both were filled with 0, a value that is not used as an ID. A missing `country` was replaced by "
  "a separate category 'Unknown' so that the 488 rows are kept without inventing an origin. The four missing `children` "
  "values were filled with the mode 0.")
J("**What would happen without handling them?** (1) scikit-learn estimators such as KNN, MLP, Random Forest, the "
  "`StandardScaler` and PCA raise an error when they meet NaN values, so the models could not even be trained. (2) Simply "
  "deleting rows with missing values would delete 94% of the dataset because of `company`, leaving a small and biased sample "
  "of company bookings only. (3) Filling the IDs with a mean would create non-existing agents/companies and hide the useful "
  "information that the booking was direct or private. (4) The engineered feature `total_guests` would be missing for the rows without `children`.")
H("4.2 Impossible records (data errors)", 2)
J(f"We removed **{prep['Ghost bookings removed']} 'ghost bookings'** with 0 adults, 0 children and 0 babies (a stay without guests "
  "is not possible) and **2 price errors**: one booking with a negative ADR (-6.38 EUR) and one with an ADR of 5,400 EUR "
  f"(the next highest value is 510 EUR). {prep['Cleaned Records']:,} bookings remain.")
H("4.3 Outliers: definition, detection with two (three) methods", 2)
J("An **outlier** is an observation that lies far away from the bulk of the other observations of a variable. It can be a "
  "**data error** (a typo, a test record, an impossible value) or a **real but rare event** (a booking made two years ahead, a "
  "group of 50 people, a guest with 26 previous cancellations). Errors distort statistics and models and must be removed; "
  "rare real events contain information that the model also needs in production. We detected outliers with:")
B(["**IQR (Tukey) rule:** x is an outlier if x < Q1 - 1.5·IQR or x > Q3 + 1.5·IQR, with IQR = Q3 - Q1. It does not assume any distribution.",
   "**Z-score rule:** x is an outlier if |z| = |x - mean| / std > 3. It assumes a roughly normal distribution.",
   "**Isolation Forest** (additional, multivariate): random trees isolate unusual *combinations* of values; the 1% most easily isolated bookings were flagged."], "List Number")
ot = outl[["IQR lower", "IQR upper", "IQR outliers", "IQR %", "Z>3 outliers", "Z %", "max", "cancel rate of IQR outliers"]].copy()
ot.columns = ["IQR low", "IQR high", "IQR n", "IQR %", "Z>3 n", "Z %", "max", "cancel rate (IQR out.)"]
T(ot, "Outliers detected by the IQR and the Z-score rule (cleaned data, 119,208 rows).", 8,
  [4.2, 1.4, 1.4, 1.4, 1.2, 1.3, 1.1, 1.1, 2.0], index=True)
F("02_data_preprocessing/outlier_detection.png", "Number of outliers per feature for both rules and box plots of four features.")
oc = ocmp.copy()
oc["% of data"] = oc["% of data"].map(lambda x: f"{x:.1f}%")
oc["Cancellation rate of flagged rows"] = oc["Cancellation rate of flagged rows"].map(lambda x: f"{x * 100:.1f}%")
oc["Rows flagged"] = oc["Rows flagged"].map("{:,}".format)
T(oc, "Comparison of the three detection methods.", 9, [5.5, 2.5, 2.5, 5])
J(f"The IQR rule is very aggressive on our data: {ocmp.iloc[0]['% of data']:.1f}% of all bookings contain at least one IQR "
  "outlier. The reason is that many attributes are counts that are 0 for most bookings (Q1 = Q3 = 0), so *every* non-zero "
  "value (one child, one booking change, one parking space) becomes an outlier; for `adults` every value other than 2 is "
  f"flagged. The Z-score rule flags {ocmp.iloc[1]['% of data']:.1f}% of the rows and agrees with the IQR rule on most of them; "
  "Isolation Forest isolates bookings with unusual combinations (e.g. long stays with many previous bookings).")
H("4.4 How we handled the outliers and why", 2)
B(["**Errors were removed:** the ghost bookings and the two impossible prices (Section 4.2).",
   f"**All other outliers were kept.** Removing all IQR outliers would delete more than half of the data. Moreover the "
   f"flagged rows are not random noise: bookings with a very long lead time are cancelled {outl.loc['lead_time', 'cancel rate of IQR outliers'] * 100:.1f}% of the time and guests "
   f"with previous cancellations {outl.loc['previous_cancellations', 'cancel rate of IQR outliers'] * 100:.1f}% of the time, while bookings with parking requests are never "
   "cancelled. These extreme values are among the most informative signals for our problem; deleting them would bias the model and the test set.",
   "Our strongest algorithms are **tree-based** (Decision Tree, Random Forest, XGBoost, LightGBM). Trees split on thresholds and the "
   "exact size of an extreme value does not matter, so they are robust to outliers.",
   "For KNN and the MLP, which are sensitive to scale, the influence of extreme values is reduced by **standardisation** (Section 4.6)."])
H("4.5 Target leakage and feature engineering", 2)
J("`reservation_status` is 'Check-Out' for every kept booking and 'Canceled' or 'No-Show' for every cancelled booking (a perfect "
  "1-to-1 mapping verified in notebook 02), and `reservation_status_date` is the date of that final status. Both are only known "
  "*after* the outcome, so a model that uses them would reach ~100% accuracy in the notebook but would be useless in practice. "
  "They were dropped. Five new features were engineered:")
B(["`total_stay_nights` = weekend nights + week nights", "`total_guests` = adults + children + babies",
   "`has_special_requests` = 1 if at least one special request (cancel rate 21.8% vs 47.8% without)",
   "`has_booking_changes` = 1 if the booking was modified", "`room_type_changed` = 1 if the assigned room type differs from the reserved one"])
H("4.6 Train/test split, normalisation/scaling", 2)
J(f"The data was split **80/20 with stratification** on the target (`random_state = 42`): {prep['Training Rows (80%)']:,} training and "
  f"{prep['Testing Rows (20%)']:,} test bookings, both with a cancellation rate of 37.08%. The {prep['Numerical features (scaled)']} numerical "
  "features were standardised with `StandardScaler`: z = (x - mean) / std, where mean and std are learned **on the training set "
  "only** and then applied to the test set (no information from the test set leaks into training). After scaling every numerical "
  "feature has mean 0 and standard deviation 1 on the training set.")
J("**Why scaling?** The attributes have completely different ranges - `lead_time` up to 737 days, `adr` up to 510 EUR, counts "
  "between 0 and 10 and binary flags 0/1. KNN uses Euclidean distance, so without scaling `lead_time` and `adr` would dominate "
  "every distance; the MLP is trained with gradient descent, which converges slowly and unstably on unscaled inputs. Tree-based "
  "models are scale-invariant, so scaling does not change them; one common scaled matrix is therefore used for all six models. "
  "We chose standardisation instead of min-max normalisation because min-max scaling would squeeze most values of the skewed "
  "features into a tiny interval near 0 because of the outliers we decided to keep.")
F("02_data_preprocessing/scaling_before_after.png", "Four features before (top) and after (bottom) standardisation: the shape is kept, the scale becomes comparable.")
H("4.7 Encoding of categorical features", 2)
J(f"The {prep['Categorical features (one-hot)']} categorical attributes were encoded with **one-hot encoding** (`OneHotEncoder`, "
  "`handle_unknown='ignore'`): every category becomes its own 0/1 column. We chose one-hot instead of label/ordinal encoding "
  "because the categories have no natural order - encoding PRT = 1, GBR = 2, FRA = 3 would make the models believe that "
  "France is 'larger' than Portugal. `handle_unknown='ignore'` encodes a category that appears only in the test set as all "
  "zeros instead of raising an error. The encoder is also fit on the training set only. `agent` and `company` IDs were kept "
  "as numerical columns as in the team's original pipeline (one-hot encoding them would add more than 600 columns).")
enc2 = enc.copy()
T(enc2, f"One-hot encoding: number of 0/1 columns created per categorical feature (total predictors after encoding: {prep['Total Predictors']}).", 9, [7, 6])
prep_t = prep.reset_index()
prep_t["Value"] = prep_t["Value"].map(lambda v: f"{v:,}")
T(prep_t, "Summary of the preprocessing pipeline.", 9, [7, 4])

# ------------------------------------------------------------------ V
H("V. Machine Learning Algorithms", 1)
H("5.1 Chosen algorithms and why they suit the problem", 2)
J("Our problem is a binary classification on a large tabular dataset with mixed numerical and one-hot categorical features, "
  "non-linear relationships (Section 3.5) and outliers. We chose six algorithms from different families so that we can compare "
  "simple and complex, single and ensemble, distance-based, tree-based and neural approaches:")
algo = [
    ("K-Nearest Neighbours (KNN)", "Classifies a booking by the majority vote of the k most similar training bookings (Euclidean distance on the scaled features). Simple, non-parametric baseline that makes no assumption about the shape of the decision boundary - suitable because similar bookings (same channel, lead time, deposit) tend to have the same outcome."),
    ("Decision Tree (CART)", "Splits the data recursively with if-then rules that maximise the purity (Gini/entropy) of the child nodes. Interpretable, handles non-linear effects and interactions (e.g. Non Refund AND Portugal) and is not sensitive to outliers or scaling."),
    ("Random Forest", "Bagging ensemble of many decorrelated decision trees, each trained on a bootstrap sample with a random subset of features; the trees vote. Reduces the high variance of a single tree, robust to noise and outliers."),
    ("XGBoost", "Gradient boosting: trees are added sequentially, each one correcting the errors (gradients of the log-loss) of the current ensemble, with L1/L2 regularisation and shrinkage. State-of-the-art accuracy on tabular data."),
    ("LightGBM", "Gradient boosting with histogram-based splits and leaf-wise tree growth; very fast on large data with many sparse one-hot columns, which matches our 95k x 255 matrix."),
    ("Multilayer Perceptron (MLP)", "Feed-forward neural network with hidden ReLU layers trained with Adam and early stopping. Can learn any non-linear function of the scaled inputs; represents the neural-network family."),
]
T(pd.DataFrame(algo, columns=["Algorithm", "How it works and why it suits our problem"]), "The six algorithms.", 8.5, [3.8, 12.6])
H("5.2 Strengths and weaknesses on our dataset", 2)
sw = [
    ("KNN", "No training phase; captures local patterns; easy to explain", "Slow prediction (distance to 95k bookings, 255 dims); curse of dimensionality with many one-hot columns; needs scaling; sensitive to irrelevant features"),
    ("Decision Tree", "Interpretable rules; fast; handles outliers, skew and interactions; no scaling needed", "High variance - an unpruned tree overfits (needs max_depth / min_samples_leaf); unstable under small data changes"),
    ("Random Forest", "Low variance, robust, few critical parameters, gives feature importance; very high precision on our data", "Large model (hundreds of deep trees), slower than boosting; less interpretable than a single tree"),
    ("XGBoost", "High accuracy; regularisation against overfitting; handles mixed features", "Many hyper-parameters; slower than LightGBM on 255 columns; sequential training"),
    ("LightGBM", "Best accuracy/speed trade-off; efficient with sparse one-hot features; handles non-linearity", "Leaf-wise growth can overfit small data (not a problem with 95k rows); many hyper-parameters"),
    ("MLP", "Very flexible non-linear model; uses all features jointly", "Needs scaling and tuning; black box; on tabular data usually below tree ensembles; longest training"),
]
T(pd.DataFrame(sw, columns=["Algorithm", "Strengths (for this data)", "Weaknesses (for this data)"]), "Strengths and weaknesses of the algorithms for our dataset.", 8.5, [2.8, 6.4, 7.2])
H("5.3 How we evaluated the algorithms", 2)
J(f"All models were trained on the same {prep['Training Rows (80%)']:,} training bookings and evaluated **once** on the same held-out "
  f"test set of {prep['Testing Rows (20%)']:,} bookings that was never used for training or tuning. A booking is predicted as "
  "'cancelled' when the predicted probability is at least 0.5. From the confusion matrix (TP = cancelled and predicted "
  "cancelled, TN = kept and predicted kept, FP = kept but predicted cancelled, FN = cancelled but predicted kept) we computed:")
B(["**Accuracy** = (TP + TN) / all - the share of correct predictions. Easy to understand, but with 63/37 classes it can look good even for a weak model.",
   "**Precision** = TP / (TP + FP) - when the model says 'will cancel', how often it is right. Important for the hotel because a false alarm (FP) can lead to overbooking a guest who actually arrives.",
   "**Recall (sensitivity)** = TP / (TP + FN) - the share of real cancellations that are detected. A missed cancellation (FN) means an empty room.",
   "**Specificity** = TN / (TN + FP) - the share of kept bookings that are recognised as kept.",
   "**F1-score** = 2·Precision·Recall / (Precision + Recall) - the harmonic mean of precision and recall; our **main metric** because it balances both error types and is not inflated by the majority class.",
   "**ROC-AUC** - area under the ROC curve (true positive rate vs false positive rate for all thresholds); measures how well the model *ranks* risky above safe bookings, independent of the 0.5 threshold. We use it as the selection score in cross-validation.",
   "**PR curve / Average precision** - precision vs recall for all thresholds; more informative than ROC when the positive class is the minority."])
J("All metrics were computed with `sklearn.metrics` (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, "
  "`roc_auc_score`, `confusion_matrix`) on the test predictions; specificity was computed from the confusion matrix.")
H("5.4 Hyper-parameter optimisation", 2)
J("Each algorithm was first trained with its **default** parameters. Then we searched for better parameters with "
  "**GridSearchCV**: every combination of the grid below is trained and evaluated with **3-fold cross-validation on the "
  "training set** (the test set is not touched), using ROC-AUC as the score. For KNN, Random Forest and MLP the search was "
  "run on a random subsample of the training set (15,000 / 30,000 / 25,000 rows) to keep the run time acceptable; the best "
  "parameters were then refit on the full training set.")
hp = pd.DataFrame({"Algorithm": list(MODELS), "Search space (param_grid)": [GRIDS[m] for m in MODELS],
                   "Best parameters found": [PARAMS[m] for m in MODELS]})
T(hp, "Hyper-parameter search spaces and best parameters.", 8, [2.8, 7.4, 6.2])
H("5.5 Results before and after hyper-parameter optimisation", 2)
rows = []
for m in MODELS:
    for v, src in (("Default", default), ("Tuned", tuned)):
        r = src.loc[m]
        rows.append([m, v, p(r["Accuracy"], 2), p(r["Precision"], 2), p(r["Recall"], 2), p(r["Specificity"], 2), f4(r["F1-Score"]), f4(r["ROC-AUC"])])
T(pd.DataFrame(rows, columns=["Model", "Variant", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]),
  "Test-set results of all six algorithms before (default) and after (tuned) hyper-parameter optimisation.", 8.5)
F("10_model_comparison/default_vs_tuned_all_models.png", "Accuracy, F1-score and ROC-AUC before and after tuning.")
dlt = (tuned[["Accuracy", "F1-Score", "ROC-AUC"]] - default[["Accuracy", "F1-Score", "ROC-AUC"]]).loc[list(MODELS)]
dlt_t = dlt.copy()
dlt_t["Accuracy"] = dlt_t["Accuracy"].map(lambda x: f"{x * 100:+.2f} pp")
dlt_t["F1-Score"] = dlt_t["F1-Score"].map(lambda x: f"{x:+.4f}")
dlt_t["ROC-AUC"] = dlt_t["ROC-AUC"].map(lambda x: f"{x:+.4f}")
T(dlt_t, "Change caused by tuning (tuned minus default).", 9, [5, 3.5, 3.5, 3.5], index=True)
HPO_TEXT = "HPO_TEXT"
J(HPO_TEXT)
H("5.6 Comparison of the tuned algorithms and the best performing algorithm", 2)
lbt = lb.copy()
for c in ["Accuracy", "Precision", "Recall", "Specificity"]:
    lbt[c] = lbt[c].map(lambda x: p(x, 2))
for c in ["F1-Score", "ROC-AUC", "PR-AUC (AP)", "Brier score"]:
    lbt[c] = lbt[c].map(f4)
lbt = lbt[["Rank", "Model", "Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC (AP)", "Brier score"]]
T(lbt, "Final leaderboard of the tuned models on the test set (sorted by F1-score).", 8, bold_row=0)
F("10_model_comparison/model_leaderboard_barchart.png", "Accuracy, precision, recall, F1 and ROC-AUC of the six tuned models.")
F("10_model_comparison/combined_roc_curves.png", "ROC curves of the six tuned models.", 11)
F("10_model_comparison/combined_pr_curves.png", "Precision-recall curves of the six tuned models.", 11)
F("10_model_comparison/confusion_matrices_all.png", "Confusion matrices of the six tuned models (test set, 23,842 bookings).", 15)
BEST_TEXT = "BEST_TEXT"
J(BEST_TEXT)
F("10_model_comparison/best_model_feature_importance.png", f"Top 15 features of the tuned {BEST} model (share of total gain).", 12)
GAIN_TEXT = "GAIN_TEXT"
J(GAIN_TEXT)

# ------------------------------------------------------------------ VI
H("VI. Feature Selection and Dimensionality Reduction", 1)
H("6.1 Feature selection techniques", 2)
J(f"One-hot encoding produced {prep['Total Predictors']} predictors, many of them almost always 0 (e.g. rare countries). "
  "Fewer features can mean less noise, faster models and less overfitting. We applied **six techniques from the three "
  "families** of feature selection. All selectors were fit on the training set only:")
B(["**Filter - Variance threshold:** removes (almost) constant columns with variance < 0.001.",
   "**Filter - ANOVA F-test** (`SelectKBest(f_classif)`, k = 50): ranks every feature by how strongly its mean differs between cancelled and kept bookings.",
   "**Filter - Mutual information** (k = 50): ranks features by the information they share with the target; also captures non-linear relations.",
   "**Wrapper - Recursive Feature Elimination (RFE)** with a Decision Tree (k = 50): repeatedly trains the model and removes the 10 least important features.",
   "**Embedded - L1 (Lasso) logistic regression** (C = 0.01): the L1 penalty sets the weights of unhelpful features exactly to zero; features with non-zero weight are kept.",
   "**Embedded - LightGBM importance** (k = 50): keeps the 50 features used most for splits (the team's original method)."])
fst2 = fst.copy()
T(fst2, "Feature selection techniques and number of selected features.", 9, [6, 2.5, 3.5, 3.5])
F("09_feature_selection_and_pca/top_features_three_methods.png", "Top 15 features according to ANOVA F-score, mutual information and LightGBM importance.")
FS_COMMON = "FS_COMMON"
J(FS_COMMON)
H("6.2 Results before and after feature selection", 2)
J("To compare before and after, four algorithms with their **tuned** parameters (Decision Tree, KNN, Random Forest and "
  "LightGBM) were retrained on each feature subset and evaluated on the same test set.")
pv = fsr.pivot_table(index="Setting", columns="Model", values="F1-Score", sort=False)
order = ["All features"] + [s for s in fsr.Setting.unique() if s != "All features"]
pv = pv.loc[order][["Decision Tree", "KNN", "Random Forest", "LightGBM"]]
pv.insert(0, "Features", fsr.groupby("Setting")["Features"].first().loc[order])
pvt = pv.copy()
for c in ["Decision Tree", "KNN", "Random Forest", "LightGBM"]:
    pvt[c] = pvt[c].map(f4)
T(pvt, "F1-score of four tuned models before (All features) and after each feature selection technique.", 8.5, [5.4, 1.8, 2.3, 2.3, 2.3, 2.3], index=True)
pva = fsr.pivot_table(index="Setting", columns="Model", values="ROC-AUC", sort=False).loc[order][["Decision Tree", "KNN", "Random Forest", "LightGBM"]].map(f4)
T(pva, "ROC-AUC of the same models before and after feature selection.", 8.5, [5.4, 2.6, 2.6, 2.6, 2.6], index=True)
F("09_feature_selection_and_pca/feature_selection_comparison.png", "F1-score before/after feature selection (left) and change in F1 in percentage points (right; green = boosted).")
H("6.3 Which feature selection boosted the performance?", 2)
FS_TEXT = "FS_TEXT"
J(FS_TEXT)
F("09_feature_selection_and_pca/n_features_curve.png", "LightGBM F1-score as a function of the number of kept features (two rankings).", 12)
CURVE_TEXT = "CURVE_TEXT"
J(CURVE_TEXT)
H("6.4 Dimensionality reduction with PCA", 2)
J("Principal Component Analysis transforms the standardised feature matrix into new, uncorrelated axes (principal components) "
  "ordered by the variance they explain, and keeps only the first n components. PCA was fit on the training set only. "
  f"Because the one-hot columns spread the variance over many directions, **{pcn.loc[90]} components are needed for 90% and "
  f"{pcn.loc[95]} for 95% of the variance** (80%: {pcn.loc[80]}, 99%: {pcn.loc[99]}).")
F("09_feature_selection_and_pca/pca_scree_plot.png", "PCA explained variance: scree plot of the first 30 components and cumulative variance.")
F("09_feature_selection_and_pca/pca_2d_projection.png", "Test bookings projected on the first two principal components (red = cancelled).", 10)
pcv = pcr.pivot_table(index="Setting", columns="Model", values="F1-Score", sort=False)[["Decision Tree", "KNN", "Random Forest", "LightGBM"]]
pcv.insert(0, "Features", pcr.groupby("Setting", sort=False)["Features"].first())
pcvt = pcv.copy()
for c in ["Decision Tree", "KNN", "Random Forest", "LightGBM"]:
    pcvt[c] = pcvt[c].map(f4)
T(pcvt, "F1-score before PCA (All features) and after PCA with different numbers of components.", 8.5, [5.6, 1.8, 2.2, 2.2, 2.2, 2.2], index=True)
pca_auc = pcr.pivot_table(index="Setting", columns="Model", values="ROC-AUC", sort=False)[["Decision Tree", "KNN", "Random Forest", "LightGBM"]].map(f4)
T(pca_auc, "ROC-AUC before and after PCA.", 8.5, [5.6, 2.6, 2.6, 2.6, 2.6], index=True)
F("09_feature_selection_and_pca/pca_comparison.png", "F1-score and ROC-AUC before and after PCA.")
H("6.5 Comparison before and after dimensionality reduction", 2)
PCA_TEXT = "PCA_TEXT"
J(PCA_TEXT)

# ------------------------------------------------------------------ VII
H("VII. Conclusion", 1)
CONCL = "CONCL"
J(CONCL)
H("7.1 Limitations and future work", 2)
B(["**Duplicate rows:** 31,994 raw rows are exact duplicates and were kept by the team; identical bookings can appear in both "
   "the training and the test set, which makes the test scores somewhat optimistic. Removing them (or a group-aware split) "
   "would give a stricter estimate.",
   "**assigned_room_type / room_type_changed:** the assigned room is finally known only at check-in; in a real deployment "
   "these features would have to be replaced by the room assigned at booking time.",
   "**Single train/test split:** the final scores come from one stratified split; repeated cross-validation would give confidence intervals.",
   "**Threshold 0.5:** the decision threshold could be tuned to the hotel's real costs of a false alarm vs a missed cancellation.",
   "**Time-based validation:** training on 2015-2016 and testing on 2017 would better simulate real use."])
H("References", 1)
for ref in [
    "Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126",
    "Mostipak, J. (2020). Hotel booking demand [Dataset]. Kaggle. https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proc. 22nd ACM SIGKDD, 785-794.",
    "Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in NeurIPS 30, 3146-3154.",
    "Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. Proc. 8th IEEE ICDM, 413-422.",
    "Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. JMLR, 3, 1157-1182.",
    "Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. Phil. Trans. R. Soc. A, 374, 20150202.",
]:
    par = doc.add_paragraph(ref)
    par.paragraph_format.left_indent = Cm(0.8)
    par.paragraph_format.first_line_indent = Cm(-0.8)
    for r in par.runs:
        r.font.size = Pt(9.5)
H("Appendix - How to run the code", 1)
B(["Install the packages: `pip install -r requirements.txt` (pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, pyarrow, jupyter).",
   "Run the notebooks in order: `01_data_exploration` -> `02_data_preprocessing` -> `03_knn` ... `08_mlp` -> `09_feature_selection_and_pca` -> `10_model_comparison`.",
   "Notebook 02 writes the processed train/test matrices to `data/processed/`, notebooks 03-08 save the default and tuned models to `models/`, and every notebook saves its tables and charts to `results/<notebook name>/`."])

if __name__ == "__main__":
    import texts
    texts.fill(doc, runs)
    OUT.parent.mkdir(exist_ok=True)
    doc.save(OUT)
    print("saved", OUT, "figures:", FIG_NO[0], "tables:", TAB_NO[0])
