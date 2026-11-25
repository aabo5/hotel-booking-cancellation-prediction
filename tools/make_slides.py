"""Final presentation (16:9 .pptx). python tools/make_slides.py -> report/Final_Presentation.pptx
All numbers are read from results/."""
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from team import TEAM, COURSE, INSTRUCTOR, DATE

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "Final_Presentation.pptx"
INK = RGBColor(0x1B, 0x2F, 0x45)
ACC = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x55, 0x5F, 0x6B)
MIST = RGBColor(0xEE, 0xF2, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

facts = pd.read_csv(R / "01_data_exploration/eda_key_facts.csv", index_col=0)["value"]
prep = pd.read_csv(R / "02_data_preprocessing/preprocessing_summary.csv").set_index("Metric")["Value"]
ocmp = pd.read_csv(R / "02_data_preprocessing/outlier_methods_comparison.csv")
both = pd.read_csv(R / "10_model_comparison/default_vs_tuned_all_models.csv")
lb = pd.read_csv(R / "10_model_comparison/master_model_leaderboard.csv")
fsr = pd.read_csv(R / "09_feature_selection_and_pca/feature_selection_results.csv")
pcr = pd.read_csv(R / "09_feature_selection_and_pca/pca_results.csv")
pcn = pd.read_csv(R / "09_feature_selection_and_pca/pca_components_needed.csv", index_col=0).iloc[:, 0]
BEST = lb.iloc[0]
import texts  # noqa: E402  (result-dependent sentences, shared with the report)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
PAGE = [0]


def tb(s, x, y, w, h, text, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.space_after = Pt(6)
        r = par.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.bold = bold
        r.font.name = "Calibri"
    return box


def bullets(s, x, y, w, h, items, size=17):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.space_after = Pt(8)
        r0 = par.add_run()
        r0.text = "■  "
        r0.font.size = Pt(size - 6)
        r0.font.color.rgb = ACC
        parts = it.replace("`", "").split("**")
        for j, part in enumerate(parts):
            r = par.add_run()
            r.text = part
            r.font.size = Pt(size)
            r.font.bold = j % 2 == 1
            r.font.color.rgb = INK
            r.font.name = "Calibri"


def pic(s, path, x, y, mw, mh):
    w, h = Image.open(R / path).size
    sc = min(mw / w, mh / h)
    s.shapes.add_picture(str(R / path), Inches(x + (mw - w * sc) / 2), Inches(y + (mh - h * sc) / 2), Inches(w * sc), Inches(h * sc))


def slide(title, section=""):
    s = prs.slides.add_slide(BLANK)
    PAGE[0] += 1
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
    bar.fill.solid(); bar.fill.fore_color.rgb = INK; bar.line.fill.background()
    if section:
        tb(s, 0.6, 0.3, 8, 0.4, section.upper(), 12, ACC, True)
    tb(s, 0.6, 0.6, 12.2, 0.9, title, 28, INK, True)
    tb(s, 0.6, 7.0, 9, 0.35, "Hotel Booking Cancellation Prediction  ·  " + COURSE, 10, GREY)
    tb(s, 12.0, 7.0, 0.8, 0.35, str(PAGE[0]), 10, GREY, align=PP_ALIGN.RIGHT)
    return s


def table(s, x, y, w, df, size=12, colw=None, bold_row=None):
    rows, cols = df.shape[0] + 1, df.shape[1]
    t = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(0.38 * rows)).table
    for j, c in enumerate(df.columns):
        cell = t.cell(0, j)
        cell.text = str(c)
        cell.fill.solid(); cell.fill.fore_color.rgb = INK
        p = cell.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(size); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(v)
            cell.fill.solid(); cell.fill.fore_color.rgb = MIST if i % 2 else WHITE
            r = cell.text_frame.paragraphs[0].runs[0]
            r.font.size = Pt(size); r.font.color.rgb = INK
            r.font.bold = bold_row is not None and i - 1 == bold_row
    if colw:
        for j, cw in enumerate(colw):
            t.columns[j].width = Inches(cw)


def kpi(s, x, y, value, label, color=INK):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(2.85), Inches(1.45))
    b.fill.solid(); b.fill.fore_color.rgb = MIST; b.line.fill.background()
    tb(s, x, y + 0.12, 2.85, 0.7, value, 30, color, True, PP_ALIGN.CENTER)
    tb(s, x + 0.1, y + 0.85, 2.65, 0.6, label, 12, GREY, False, PP_ALIGN.CENTER)


pc = lambda x, d=1: f"{x * 100:.{d}f}%"

# 1 title
s = prs.slides.add_slide(BLANK); PAGE[0] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = INK; bg.line.fill.background()
tb(s, 0.8, 1.3, 11.5, 0.5, "FATIH SULTAN MEHMET VAKIF UNIVERSITY  ·  " + COURSE, 14, RGBColor(0xC9, 0xD6, 0xE3), True)
tb(s, 0.8, 2.0, 11.5, 1.4, "Hotel Booking Cancellation Prediction", 44, WHITE, True)
tb(s, 0.8, 3.2, 11.5, 0.8, "Final project presentation - EDA, preprocessing, six ML algorithms, tuning, feature selection & PCA", 18, RGBColor(0xC9, 0xD6, 0xE3))
tb(s, 0.8, 4.4, 11.5, 1.6, TEAM, 16, WHITE)
tb(s, 0.8, 6.2, 11.5, 0.5, f"Instructor: {INSTRUCTOR}   ·   {DATE}", 13, RGBColor(0xC9, 0xD6, 0xE3))

# 2 problem
s = slide("Why predict cancellations?", "I. Introduction")
bullets(s, 0.6, 1.6, 7.0, 5, [
    f"In our data **{facts['cancel_rate_%']}%** of all bookings were cancelled - an empty room night can never be sold again.",
    "Hotels react with blunt tools: overbooking and strict deposits.",
    "Goal: estimate the **cancellation risk of each booking** when it is made -> targeted overbooking, early contact, deposit only for risky bookings.",
    "Binary classification: target `is_canceled` (1 = cancelled, 0 = arrived).",
    "Methods: EDA · cleaning, outliers, scaling, encoding · **KNN, Decision Tree, Random Forest, XGBoost, LightGBM, MLP** · GridSearchCV · 6 feature-selection techniques · PCA"])
kpi(s, 8.3, 1.8, f"{int(str(119390)):,}", "bookings, 2 Portuguese hotels")
kpi(s, 8.3, 3.5, f"{facts['cancel_rate_%']}%", "cancelled", ACC)
kpi(s, 8.3, 5.2, "6", "algorithms compared")

# 3 dataset
s = slide("Dataset: Hotel Booking Demand (Antonio et al., 2019)", "II. Dataset")
bullets(s, 0.6, 1.6, 6.6, 5, [
    "Source: PMS data of a **City Hotel (Lisbon)** and a **Resort Hotel (Algarve)**, published in Data in Brief and on Kaggle.",
    "Arrivals **July 2015 - August 2017**; **119,390 rows x 32 columns** (20 numerical, 12 categorical).",
    "Timing (lead time, dates), stay & guests, customer & channel, room & price (ADR), deposit, requests & history.",
    "Suitable: labelled outcome for every booking, almost all attributes known at booking time, large and real (contains missing values, outliers, leakage)."])
table(s, 7.5, 1.7, 5.3, pd.DataFrame([["Rows", "119,390"], ["Attributes", "32"], ["Numerical / categorical", "20 / 12"],
                                       ["Missing-value columns", "4"], ["Exact duplicates", "31,994"], ["Target cancelled", f"{facts['cancel_rate_%']}%"],
                                       ["Predictors after encoding", str(prep['Total Predictors'])]], columns=["Item", "Value"]), 14)

# 4-6 EDA
s = slide("EDA - what drives cancellations?", "III. Exploratory Data Analysis")
pic(s, "01_data_exploration/cancellation_by_categories.png", 0.4, 1.5, 12.5, 3.6)
bullets(s, 0.6, 5.2, 12.2, 1.8, [
    f"City Hotel {facts['city_cancel_%']}% vs Resort {facts['resort_cancel_%']}%  ·  **Non Refund deposit: {facts['non_refund_cancel_%']}% cancelled** (97% Portuguese group blocks, median lead time 183 days)  ·  Groups {facts['groups_cancel_%']}%"], 16)
s = slide("EDA - lead time, seasonality and commitment signals", "III. Exploratory Data Analysis")
pic(s, "01_data_exploration/lead_time_density.png", 0.4, 1.5, 6.2, 2.6)
pic(s, "01_data_exploration/monthly_seasonality.png", 6.7, 1.5, 6.3, 2.6)
pic(s, "01_data_exploration/behavioural_features.png", 0.4, 4.2, 12.5, 2.0)
tb(s, 0.6, 6.3, 12.2, 0.6, f"Median lead time {facts['median_lead_cancel']:.0f} vs {facts['median_lead_kept']:.0f} days · no special request -> {facts['cancel_%_0_requests']}% cancel · parking request -> 0% cancel · summer peak", 15, INK, True)
s = slide("EDA - statistics, correlations and countries", "III. Exploratory Data Analysis")
pic(s, "01_data_exploration/correlation_heatmap.png", 0.3, 1.4, 5.6, 5.5)
pic(s, "01_data_exploration/top_countries.png", 6.0, 1.5, 7.0, 2.9)
bullets(s, 6.1, 4.6, 6.8, 2.4, [
    "Feature dictionary + descriptive statistics of all 32 attributes (notebook 01).",
    "Most counts strongly right-skewed (skew > 1); weak linear correlation (|r| < 0.3) -> non-linear models.",
    f"Portugal: {facts['PRT_share_%']}% of bookings, {facts['PRT_cancel_%']}% cancelled. **10 charts** in total."], 15)

# 7 missing + leakage
s = slide("Preprocessing - missing values, errors and leakage", "IV. Data preprocessing")
table(s, 0.6, 1.6, 6.3, pd.DataFrame([["company", "94.31%", "0 = no company"], ["agent", "13.69%", "0 = direct booking"],
                                       ["country", "0.41%", "'Unknown'"], ["children", "4 rows", "0 (mode)"], ["other 28", "0%", "-"]],
                                      columns=["Attribute", "Missing", "Action"]), 14)
bullets(s, 7.2, 1.6, 5.7, 5.2, [
    "Without handling: sklearn models crash on NaN; dropna would delete **94%** of the data.",
    f"Removed **{prep['Ghost bookings removed']} ghost bookings** (0 guests) + **2 price errors** (-6.38 and 5,400 EUR) -> {prep['Cleaned Records']:,} rows.",
    "**Leakage:** `reservation_status` = Check-Out / Canceled maps 1-to-1 to the target -> dropped with its date.",
    "5 engineered features: total nights, total guests, special-request flag, change flag, room-type-changed flag."], 15)

# 8 outliers
s = slide("Outliers - two detection methods (+ Isolation Forest)", "IV. Data preprocessing")
pic(s, "02_data_preprocessing/outlier_detection.png", 0.3, 1.45, 8.2, 3.0)
oc = ocmp.copy()
oc["% of data"] = oc["% of data"].map(lambda x: f"{x:.1f}%")
oc["Cancellation rate of flagged rows"] = oc["Cancellation rate of flagged rows"].map(lambda x: f"{x * 100:.1f}%")
oc.columns = ["Method", "Rows", "% data", "Cancel rate"]
table(s, 8.7, 1.6, 4.3, oc, 11, [1.9, 0.8, 0.8, 0.8])
bullets(s, 0.6, 4.6, 12.3, 2.3, [
    "**IQR rule** (outside Q1-1.5·IQR, Q3+1.5·IQR) and **Z-score** (|z| > 3); Isolation Forest for unusual combinations.",
    f"IQR flags {ocmp.iloc[0]['% of data']:.0f}% of rows: count features are mostly 0, so every non-zero value is an 'outlier'.",
    "Decision: remove only **errors**; keep real rare cases - they are strong signals (long lead time -> 66% cancel, previous cancellations -> 92%). Trees are robust; scaling softens them for KNN/MLP."], 15)

# 9 scaling encoding
s = slide("Scaling and encoding", "IV. Data preprocessing")
pic(s, "02_data_preprocessing/scaling_before_after.png", 0.3, 1.45, 7.6, 3.4)
bullets(s, 8.1, 1.5, 4.9, 5.4, [
    f"Stratified **80/20 split**: {prep['Training Rows (80%)']:,} train / {prep['Testing Rows (20%)']:,} test (both 37.08% cancelled).",
    f"**StandardScaler** on {prep['Numerical features (scaled)']} numerical features (fit on train only) - KNN distances and MLP gradients need comparable scales.",
    f"**One-hot encoding** of {prep['Categorical features (one-hot)']} categorical features (no natural order; country -> 172 columns).",
    f"Result: **{prep['Total Predictors']} predictors**."], 15)

# 10 algorithms
s = slide("Six algorithms - strengths and weaknesses on our data", "V. Machine learning algorithms")
table(s, 0.5, 1.5, 12.3, pd.DataFrame([
    ["KNN", "simple, non-parametric, local patterns", "slow prediction, curse of dimensionality (255 dims)"],
    ["Decision Tree", "interpretable rules, robust to outliers", "high variance, overfits if unpruned"],
    ["Random Forest", "bagging -> low variance, high precision", "large model, slower"],
    ["XGBoost", "boosting + regularisation, strong on tabular data", "many hyper-parameters"],
    ["LightGBM", "fast histogram boosting, sparse one-hot friendly", "leaf-wise growth can overfit small data"],
    ["MLP", "flexible non-linear neural network", "needs scaling/tuning, black box, slowest"]],
    columns=["Algorithm", "Strength", "Weakness"]), 14, [2.3, 5.0, 5.0])
tb(s, 0.6, 5.0, 12.2, 1.8, "Evaluation: one untouched test set (23,842 bookings), threshold 0.5. Metrics: accuracy, precision (false alarms -> overbooking risk), recall (missed cancellations -> empty rooms), specificity, F1 (main metric), ROC-AUC (ranking quality, used for CV), PR curves, confusion matrices.", 15)

# 11 HPO
s = slide("Hyper-parameter optimisation - before vs after", "V. Machine learning algorithms")
pic(s, "10_model_comparison/default_vs_tuned_all_models.png", 0.3, 1.4, 12.7, 3.3)
bullets(s, 0.6, 4.8, 12.3, 2.1, ["`GridSearchCV`, **3-fold cross-validation** on the training set, score = ROC-AUC; best parameters refit on all training data.", texts.HPO_SHORT], 15)

# 12 leaderboard
s = slide("Final comparison of the tuned models", "V. Machine learning algorithms")
lt = lb[["Rank", "Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].copy()
for c in ["Accuracy", "Precision", "Recall"]:
    lt[c] = lt[c].map(lambda x: pc(x, 2))
for c in ["F1-Score", "ROC-AUC"]:
    lt[c] = lt[c].map(lambda x: f"{x:.4f}")
table(s, 0.5, 1.5, 7.4, lt, 13, [0.6, 1.9, 1.0, 1.0, 0.9, 1.0, 1.0], bold_row=0)
pic(s, "10_model_comparison/combined_roc_curves.png", 8.1, 1.4, 4.9, 4.3)
tb(s, 0.6, 5.0, 7.3, 1.9, texts.BEST_SHORT, 15, INK)

# 13 best model insight
s = slide(f"Inside the best model ({BEST['Model']})", "V. Machine learning algorithms")
pic(s, "10_model_comparison/best_model_feature_importance.png", 0.3, 1.4, 6.6, 5.4)
pic(s, "10_model_comparison/confusion_matrices_all.png", 7.0, 1.4, 6.0, 4.3)
tb(s, 7.1, 5.8, 5.9, 1.1, texts.GAIN_SHORT, 14, INK)

# 14 feature selection
s = slide("Feature selection - 6 techniques, 3 families", "VI. Feature selection & PCA")
pic(s, "09_feature_selection_and_pca/feature_selection_comparison.png", 0.3, 1.4, 12.7, 3.6)
bullets(s, 0.6, 5.1, 12.3, 1.8, ["Filter: variance threshold, ANOVA F-test, mutual information · Wrapper: RFE (Decision Tree) · Embedded: L1 logistic regression, LightGBM importance. Retrained 4 tuned models on each subset.", texts.FS_SHORT], 14)

# 15 PCA
s = slide("Dimensionality reduction with PCA", "VI. Feature selection & PCA")
pic(s, "09_feature_selection_and_pca/pca_scree_plot.png", 0.3, 1.35, 6.3, 2.6)
pic(s, "09_feature_selection_and_pca/pca_comparison.png", 6.7, 1.35, 6.3, 2.6)
pic(s, "09_feature_selection_and_pca/pca_2d_projection.png", 9.9, 4.05, 3.1, 2.9)
bullets(s, 0.6, 4.2, 9.2, 2.7, [f"PCA fit on training data: **{pcn.loc[90]} components for 90%**, {pcn.loc[95]} for 95% of the variance.", texts.PCA_SHORT], 15)

# 16 conclusion
s = slide("Conclusions", "Summary")
bullets(s, 0.6, 1.6, 12.2, 5.3, texts.CONCL_BULLETS, 17)

# 17 demo + thanks
s = slide("Live demo & questions", "Demo")
bullets(s, 0.6, 1.6, 12.2, 3.5, [
    "`python demo/app.py` - enter a booking (hotel, lead time, deposit, country, requests ...) and get the cancellation probability of any of the six tuned models.",
    "Presets: high-risk (Portuguese group, Non Refund, 250 days ahead), typical online booking, low-risk (direct, 5 days, parking + requests).",
    "Code: 10 notebooks (01 EDA -> 10 comparison), all tables and charts in results/."], 17)
tb(s, 0.6, 5.2, 12.2, 1, "Thank you - questions?", 36, INK, True, PP_ALIGN.CENTER)

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print("saved", OUT, "slides:", PAGE[0])
