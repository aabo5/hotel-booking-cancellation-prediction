"""Rebuilds notebooks 01, 02, 09 and 10 so that they cover every item of the course guideline.

Notebooks 03-08 (one per algorithm) are kept as written by the team; they are only re-executed.
Usage:  python tools/build_notebooks.py      then   python tools/run_notebooks.py 01 02 ... 10
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def save(name, cells):
    for c in cells:
        if c.cell_type == "markdown" and c.source in SUMMARIES:
            c.source = SUMMARIES[c.source]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, NB / name)
    print("wrote", name)


SUMMARIES = {}   # filled in after the notebooks were executed (final summary cells written from the real outputs)
try:
    from summaries import SUMMARIES
except ImportError:
    pass

md = new_markdown_cell
code = new_code_cell

COMMON = """import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
"""

# =====================================================================================  01
nb01 = [
    md("""# 01. Exploratory Data Analysis (EDA)
Guideline sections **II (Description of the dataset)** and **III (Exploratory Data Analysis)**.

In this notebook we:
1. describe the source and the dimensionality of the dataset,
2. explain every feature (meaning, data type, number of distinct values, example values),
3. give descriptive statistics for **every** attribute (numerical and categorical),
4. inspect the target variable and missing values,
5. build ten charts of the features that matter most for cancellations."""),
    code(COMMON + """
output_dir = '../results/01_data_exploration' if os.path.exists('../results') else 'results/01_data_exploration'
os.makedirs(output_dir, exist_ok=True)

data_path = '../data/raw/hotel_bookings.csv' if os.path.exists('../data/raw') else 'data/raw/hotel_bookings.csv'
df = pd.read_csv(data_path)
df.head()"""),
    md("""## 1. Source of the dataset
* **Name:** Hotel Booking Demand (file `hotel_bookings.csv`).
* **Origin:** Antonio, N., de Almeida, A. & Nunes, L. (2019). *Hotel booking demand datasets*. **Data in Brief, 22, 41-49.** The authors extracted the data from the Property Management Systems (PMS) of two real hotels in Portugal: a **Resort Hotel** (Algarve) and a **City Hotel** (Lisbon). Personal information was removed.
* **Distribution:** made public on Kaggle (*jessemostipak/hotel-booking-demand*, TidyTuesday 2020).
* **Period:** bookings with arrival dates between **1 July 2015 and 31 August 2017**.
* **Why it suits our problem:** every row is one real booking and the column `is_canceled` tells whether it was cancelled. It also stores what the hotel knows when the booking is made (lead time, deposit, channel, price, special requests ...), so we can learn the cancellation pattern from information that is available *before* the guest arrives - exactly what a supervised classification model needs. With ~119k labelled rows it is big enough to train and fairly compare six algorithms."""),
    md("## 2. Dimensionality of the dataset"),
    code("""n_rows, n_cols = df.shape
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(exclude='number').columns.tolist()

dim = pd.DataFrame({
    'Item': ['Rows (bookings)', 'Columns (attributes)', 'Numerical columns', 'Categorical / text columns',
             'Target column', 'Memory usage (MB)', 'Exact duplicate rows', 'First arrival', 'Last arrival'],
    'Value': [f'{n_rows:,}', n_cols, len(num_cols), len(cat_cols), 'is_canceled',
              round(df.memory_usage(deep=True).sum() / 1e6, 1), f'{df.duplicated().sum():,}',
              f"{df.arrival_date_year.min()}-{df.loc[df.arrival_date_year == df.arrival_date_year.min(), 'arrival_date_month'].iloc[0]}",
              f"{df.arrival_date_year.max()}-{df.loc[df.arrival_date_year == df.arrival_date_year.max(), 'arrival_date_month'].iloc[-1]}"]
})
dim.to_csv(f'{output_dir}/dataset_dimensionality.csv', index=False)
print(f'Dataset Dimensions: {n_rows:,} rows and {n_cols} columns')
dim"""),
    md("""## 3. Meaning and data type of every feature
The table below lists all 32 attributes: what they represent, their pandas data type, whether we treat them as numerical or categorical, the number of distinct values and some example values."""),
    code("""meaning = {
    'hotel': 'Hotel type: City Hotel or Resort Hotel',
    'is_canceled': 'TARGET - 1 if the booking was cancelled, 0 if the guest arrived',
    'lead_time': 'Days between the booking date and the arrival date',
    'arrival_date_year': 'Year of arrival (2015-2017)',
    'arrival_date_month': 'Month of arrival (January ... December)',
    'arrival_date_week_number': 'Week number of the year of arrival (1-53)',
    'arrival_date_day_of_month': 'Day of the month of arrival (1-31)',
    'stays_in_weekend_nights': 'Number of weekend nights (Saturday/Sunday) booked',
    'stays_in_week_nights': 'Number of week nights (Monday-Friday) booked',
    'adults': 'Number of adults',
    'children': 'Number of children',
    'babies': 'Number of babies',
    'meal': 'Meal package: BB (bed & breakfast), HB (half board), FB (full board), SC/Undefined (no meal)',
    'country': 'Country of origin of the guest (ISO 3166-1 alpha-3 code)',
    'market_segment': 'Market segment (Online TA, Offline TA/TO, Groups, Direct, Corporate, ...)',
    'distribution_channel': 'Booking distribution channel (TA/TO = travel agents/tour operators, Direct, Corporate, GDS)',
    'is_repeated_guest': '1 if the guest has stayed at the hotel before, else 0',
    'previous_cancellations': 'Number of earlier bookings of this guest that were cancelled',
    'previous_bookings_not_canceled': 'Number of earlier bookings of this guest that were not cancelled',
    'reserved_room_type': 'Code of the room type reserved (A-L, anonymised)',
    'assigned_room_type': 'Code of the room type actually assigned (can differ from the reserved one)',
    'booking_changes': 'Number of changes made to the booking until check-in or cancellation',
    'deposit_type': 'No Deposit, Non Refund (full prepayment) or Refundable deposit',
    'agent': 'ID of the travel agency that made the booking (anonymised, NaN = no agency)',
    'company': 'ID of the company that pays for the booking (anonymised, NaN = no company)',
    'days_in_waiting_list': 'Days the booking waited on the waiting list before it was confirmed',
    'customer_type': 'Transient, Transient-Party, Contract or Group',
    'adr': 'Average Daily Rate in EUR = total lodging price / number of nights',
    'required_car_parking_spaces': 'Number of car parking spaces requested',
    'total_of_special_requests': 'Number of special requests (e.g. twin bed, high floor)',
    'reservation_status': 'Final status: Check-Out, Canceled or No-Show (only known AFTER the outcome)',
    'reservation_status_date': 'Date the last status was set (only known AFTER the outcome)',
}
binary = {'is_canceled', 'is_repeated_guest'}
rows = []
for c in df.columns:
    s = df[c]
    kind = 'Target (binary)' if c == 'is_canceled' else (
        'Binary' if c in binary else ('Numerical' if pd.api.types.is_numeric_dtype(s) else 'Categorical'))
    if c in ('agent', 'company'):
        kind = 'Categorical ID (stored as number)'
    if c == 'reservation_status_date':
        kind = 'Date (text)'
    ex = ', '.join(map(str, s.dropna().unique()[:4]))
    rows.append({'Feature': c, 'Meaning': meaning[c], 'Data type': str(s.dtype), 'Kind': kind,
                 'Distinct values': s.nunique(), 'Example values': ex})
feature_dictionary = pd.DataFrame(rows)
feature_dictionary.to_csv(f'{output_dir}/feature_dictionary.csv', index=False)
with pd.option_context('display.max_colwidth', 70):
    display(feature_dictionary)"""),
    md("""**Note on leakage columns:** `reservation_status` and `reservation_status_date` are only filled in *after* the guest arrives or cancels (`Canceled` / `No-Show` = cancelled). They describe the answer, not the booking, so they are removed in preprocessing (notebook 02)."""),
    md("## 4. Descriptive statistics for every attribute\n### 4.1 Numerical attributes"),
    code("""num = df.select_dtypes(include='number')
desc_num = num.describe().T
desc_num['median'] = num.median()
desc_num['skewness'] = num.skew()
desc_num['kurtosis'] = num.kurt()
desc_num['missing'] = num.isnull().sum()
desc_num = desc_num[['count', 'missing', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'skewness', 'kurtosis']]
desc_num.round(3).to_csv(f'{output_dir}/descriptive_statistics_numerical.csv')
desc_num.round(2)"""),
    md("### 4.2 Categorical attributes"),
    code("""cat = df.select_dtypes(exclude='number')
desc_cat = cat.describe().T
desc_cat['top_share_%'] = (desc_cat['freq'] / len(df) * 100).astype(float).round(2)
desc_cat['missing'] = cat.isnull().sum()
desc_cat.to_csv(f'{output_dir}/descriptive_statistics_categorical.csv')
desc_cat"""),
    code("""# Frequency tables of the low-cardinality categorical attributes
for c in ['hotel', 'meal', 'market_segment', 'distribution_channel', 'deposit_type', 'customer_type', 'reservation_status']:
    vc = df[c].value_counts()
    print(f'--- {c} ---')
    print(pd.DataFrame({'count': vc, 'share_%': (vc / len(df) * 100).round(2),
                        'cancel_rate_%': (df.groupby(c)['is_canceled'].mean()[vc.index] * 100).round(1)}).to_string(), '\\n')"""),
    md("""**Reading the statistics:**
* Most numerical attributes are strongly **right-skewed** (skewness > 1): `lead_time`, `adr`, `stays_in_week_nights`, `days_in_waiting_list`, `previous_cancellations`, `babies`. Their maximum values are far above the 75th percentile - these are the candidate outliers we analyse in notebook 02.
* `adr` has a minimum below zero and a maximum of 5,400 EUR, which are data-entry errors.
* `adults` has a minimum of 0 - bookings without any guest are not real stays.
* `agent` and `company` are numbers but they are **IDs**, so their mean has no meaning."""),
    md("## 5. Target variable: booking cancellation rate"),
    code("""cancel_counts = df['is_canceled'].value_counts()
cancel_rate = df['is_canceled'].mean() * 100
print(f"Completed Stays (0): {cancel_counts[0]:,} ({100 - cancel_rate:.1f}%)")
print(f"Canceled Bookings (1): {cancel_counts[1]:,} ({cancel_rate:.1f}%)")

plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='is_canceled', hue='is_canceled', palette=['#1f77b4', '#d62728'], legend=False)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.title('Chart 1 - Booking Cancellation Distribution')
plt.xlabel('Booking Outcome')
plt.ylabel('Number of Bookings')
plt.xticks([0, 1], [f'Not Canceled ({100 - cancel_rate:.0f}%)', f'Canceled ({cancel_rate:.0f}%)'])
plt.tight_layout()
plt.savefig(f'{output_dir}/target_distribution.png', dpi=300)
plt.show()"""),
    md("The classes are **moderately imbalanced (63 / 37)**. Accuracy alone can therefore be misleading, so we also report precision, recall, F1-score and ROC-AUC, and we use a **stratified** train/test split."),
    md("## 6. Missing values (first look - handled in notebook 02)"),
    code("""missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_df = pd.DataFrame({'Missing_Rows': missing, 'Percentage': (missing / len(df) * 100).round(2)})
missing_df.to_csv(f'{output_dir}/missing_values_summary.csv')
print('Missing Values:')
missing_df"""),
    md("## 7. Charts of the most important features\n### Chart 2 - Cancellation rate by hotel type, deposit type and market segment"),
    code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col, pal, title in zip(axes, ['hotel', 'deposit_type', 'market_segment'], ['Blues_d', 'Reds_d', 'Greens_d'],
                               ['Hotel Type', 'Deposit Type', 'Market Segment']):
    sns.barplot(data=df, x=col, y='is_canceled', hue=col, ax=ax, palette=pal, legend=False, errorbar=None)
    ax.set_title(f'Cancellation Rate by {title}')
    ax.set_ylabel('Cancellation Rate')
    ax.set_xlabel('')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0%}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=9)
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/cancellation_by_categories.png', dpi=300)
plt.show()"""),
    md("### Chart 3 - Lead time and price (ADR) by cancellation status"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.boxplot(data=df, x='is_canceled', y='lead_time', hue='is_canceled', ax=axes[0], palette=['#1f77b4', '#d62728'], legend=False)
axes[0].set_title('Lead Time by Cancellation Status')
axes[0].set_xticks([0, 1], ['Not Canceled', 'Canceled'])
axes[0].set_ylabel('Lead Time (Days)')
axes[0].set_xlabel('')
sns.boxplot(data=df[(df['adr'] >= 0) & (df['adr'] < 500)], x='is_canceled', y='adr', hue='is_canceled', ax=axes[1],
            palette=['#1f77b4', '#d62728'], legend=False)
axes[1].set_title('Average Daily Rate (ADR) by Cancellation Status')
axes[1].set_xticks([0, 1], ['Not Canceled', 'Canceled'])
axes[1].set_ylabel('Price (EUR)')
axes[1].set_xlabel('')
plt.tight_layout()
plt.savefig(f'{output_dir}/lead_time_and_adr_boxplots.png', dpi=300)
plt.show()

summary_stats = df.groupby('is_canceled')[['lead_time', 'adr']].median()
summary_stats.to_csv(f'{output_dir}/lead_time_adr_medians.csv')
print('Median Values by Booking Status:')
print(summary_stats)"""),
    md("### Chart 4 - Distributions (histograms) of the main numerical features"),
    code("""cols = ['lead_time', 'adr', 'stays_in_week_nights', 'stays_in_weekend_nights', 'adults',
        'total_of_special_requests', 'booking_changes', 'days_in_waiting_list', 'previous_cancellations']
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, c in zip(axes.ravel(), cols):
    data = df[c] if c != 'adr' else df.loc[(df.adr >= 0) & (df.adr < 500), c]
    sns.histplot(data, bins=40, ax=ax, color='#4c72b0')
    ax.set_title(f'{c}  (skew = {df[c].skew():.2f})')
    ax.set_xlabel('')
plt.suptitle('Chart 4 - Distributions of numerical features', fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(f'{output_dir}/numerical_histograms.png', dpi=300)
plt.show()"""),
    md("### Chart 5 - Seasonality: bookings and cancellation rate per arrival month"),
    code("""months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthly = df.groupby(['arrival_date_month', 'hotel'])['is_canceled'].agg(['size', 'mean']).reset_index()
monthly['arrival_date_month'] = pd.Categorical(monthly['arrival_date_month'], months, ordered=True)
monthly = monthly.sort_values('arrival_date_month')

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
sns.lineplot(data=monthly, x='arrival_date_month', y='size', hue='hotel', marker='o', ax=axes[0])
axes[0].set_title('Number of bookings per arrival month')
axes[0].set_ylabel('Bookings')
sns.lineplot(data=monthly, x='arrival_date_month', y='mean', hue='hotel', marker='o', ax=axes[1])
axes[1].set_title('Cancellation rate per arrival month')
axes[1].set_ylabel('Cancellation rate')
for ax in axes:
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('')
plt.tight_layout()
plt.savefig(f'{output_dir}/monthly_seasonality.png', dpi=300)
plt.show()
monthly.to_csv(f'{output_dir}/monthly_bookings_cancel_rate.csv', index=False)"""),
    md("### Chart 6 - Correlation heatmap of numerical features"),
    code("""corr = df.select_dtypes(include='number').drop(columns=['agent', 'company']).corr()
plt.figure(figsize=(13, 10))
sns.heatmap(corr, cmap='RdBu_r', center=0, annot=True, fmt='.2f', annot_kws={'size': 7}, square=True, cbar_kws={'shrink': .7})
plt.title('Chart 6 - Pearson correlation of numerical features')
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300)
plt.show()

target_corr = corr['is_canceled'].drop('is_canceled').sort_values(key=abs, ascending=False)
target_corr.to_csv(f'{output_dir}/correlation_with_target.csv', header=['corr_with_is_canceled'])
print('Correlation with is_canceled:')
print(target_corr.round(3).to_string())"""),
    md("### Chart 7 - Top 10 countries: booking volume and cancellation rate"),
    code("""top = df['country'].value_counts().head(10).index
cty = df[df['country'].isin(top)].groupby('country')['is_canceled'].agg(['size', 'mean']).loc[top]
fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.bar(cty.index, cty['size'], color='#9ecae1', label='Bookings')
ax1.set_ylabel('Bookings')
ax2 = ax1.twinx()
ax2.plot(cty.index, cty['mean'], color='#d62728', marker='o', lw=2, label='Cancellation rate')
ax2.set_ylabel('Cancellation rate')
ax2.set_ylim(0, 0.8)
ax2.grid(False)
ax1.set_title('Chart 7 - Top 10 countries of origin')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88))
plt.tight_layout()
plt.savefig(f'{output_dir}/top_countries.png', dpi=300)
plt.show()
cty.rename(columns={'size': 'bookings', 'mean': 'cancel_rate'}).round(3)"""),
    md("### Chart 8 - Commitment signals: special requests, parking, previous cancellations, booking changes"),
    code("""fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, c in zip(axes, ['total_of_special_requests', 'required_car_parking_spaces', 'previous_cancellations', 'booking_changes']):
    tmp = df[[c, 'is_canceled']].copy()
    tmp[c] = tmp[c].clip(upper=3 if c != 'required_car_parking_spaces' else 1)
    sns.barplot(data=tmp, x=c, y='is_canceled', ax=ax, color='#e6550d', errorbar=None)
    ax.set_title(c)
    ax.set_ylabel('Cancellation rate')
    ax.set_xlabel('value (last bar = this value or more)')
plt.suptitle('Chart 8 - Cancellation rate by behavioural features', y=1.03)
plt.tight_layout()
plt.savefig(f'{output_dir}/behavioural_features.png', dpi=300)
plt.show()"""),
    md("### Chart 9 - Customer type and distribution channel"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, c in zip(axes, ['customer_type', 'distribution_channel']):
    ct = pd.crosstab(df[c], df['is_canceled'])
    ct.columns = ['Not Canceled', 'Canceled']
    ct.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#d62728'])
    ax.set_title(f'Bookings by {c} (stacked by outcome)')
    ax.set_ylabel('Bookings')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(f'{output_dir}/customer_type_channel.png', dpi=300)
plt.show()"""),
    md("### Chart 10 - Lead time density by outcome"),
    code("""plt.figure(figsize=(10, 4))
sns.kdeplot(data=df, x='lead_time', hue='is_canceled', fill=True, common_norm=False, palette=['#1f77b4', '#d62728'], clip=(0, 500))
plt.title('Chart 10 - Lead time density (canceled vs not canceled)')
plt.xlabel('Lead time (days)')
plt.tight_layout()
plt.savefig(f'{output_dir}/lead_time_density.png', dpi=300)
plt.show()

bins = [-1, 7, 30, 90, 180, 365, 800]
labels = ['0-7', '8-30', '31-90', '91-180', '181-365', '>365']
lt = df.groupby(pd.cut(df['lead_time'], bins, labels=labels), observed=True)['is_canceled'].agg(['size', 'mean'])
lt.columns = ['bookings', 'cancel_rate']
lt.to_csv(f'{output_dir}/cancel_rate_by_lead_time_band.csv')
lt.round(3)"""),
    md("## 8. Key numbers used in the summary"),
    code("""facts = {
    'cancel_rate_%': round(df.is_canceled.mean() * 100, 1),
    'city_cancel_%': round(df[df.hotel == 'City Hotel'].is_canceled.mean() * 100, 1),
    'resort_cancel_%': round(df[df.hotel == 'Resort Hotel'].is_canceled.mean() * 100, 1),
    'non_refund_cancel_%': round(df[df.deposit_type == 'Non Refund'].is_canceled.mean() * 100, 1),
    'no_deposit_cancel_%': round(df[df.deposit_type == 'No Deposit'].is_canceled.mean() * 100, 1),
    'median_lead_cancel': df[df.is_canceled == 1].lead_time.median(),
    'median_lead_kept': df[df.is_canceled == 0].lead_time.median(),
    'cancel_%_0_requests': round(df[df.total_of_special_requests == 0].is_canceled.mean() * 100, 1),
    'cancel_%_parking': round(df[df.required_car_parking_spaces > 0].is_canceled.mean() * 100, 1),
    'PRT_cancel_%': round(df[df.country == 'PRT'].is_canceled.mean() * 100, 1),
    'PRT_share_%': round((df.country == 'PRT').mean() * 100, 1),
    'groups_cancel_%': round(df[df.market_segment == 'Groups'].is_canceled.mean() * 100, 1),
}
pd.Series(facts).to_csv(f'{output_dir}/eda_key_facts.csv', header=['value'])
pd.Series(facts)"""),
    md("SUMMARY_01"),
]

# =====================================================================================  02
nb02 = [
    md("""# 02. Data Cleaning & Preprocessing
Guideline section **IV (Data preprocessing)**: missing values, outliers (two detection methods + handling), leakage removal, feature engineering, scaling and encoding.

The cleaning decisions are the ones the team made in the original version of this notebook; this version adds the missing-value audit for every attribute, the outlier analysis and the scaling/encoding documentation."""),
    code(COMMON + """import joblib
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

output_dir = '../results/02_data_preprocessing' if os.path.exists('../results') else 'results/02_data_preprocessing'
models_dir = '../models' if os.path.exists('../models') else 'models'
processed_dir = '../data/processed' if os.path.exists('../data') else 'data/processed'
for d in (output_dir, models_dir, processed_dir):
    os.makedirs(d, exist_ok=True)

data_path = '../data/raw/hotel_bookings.csv' if os.path.exists('../data/raw') else 'data/raw/hotel_bookings.csv'
df = pd.read_csv(data_path)
raw = df.copy()
print(f'Starting with {len(df):,} raw records.')"""),
    md("## 1. Missing values - percentage for every attribute"),
    code("""mv = pd.DataFrame({'Data type': df.dtypes.astype(str), 'Missing rows': df.isnull().sum(),
                   'Missing %': (df.isnull().mean() * 100).round(3)}).sort_values('Missing rows', ascending=False)
mv.to_csv(f'{output_dir}/missing_values_all_attributes.csv')
display(mv)

m = mv[mv['Missing rows'] > 0]
plt.figure(figsize=(8, 3.5))
ax = sns.barplot(x=m.index, y=m['Missing %'], hue=m.index, palette='Reds_r', legend=False)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.title('Missing values per attribute (only attributes with missing values; all others 0%)')
plt.ylabel('Missing %')
plt.tight_layout()
plt.savefig(f'{output_dir}/missing_values_bar.png', dpi=300)
plt.show()"""),
    md("""### How we handle the missing values and why
Only 4 of the 32 attributes have missing values; the other 28 are complete (0%).

| Attribute | Missing | Our action | Reason |
|---|---|---|---|
| `company` | 94.31% | fill with **0** | NaN means *no company paid* - the absence is information, not an error. Dropping the rows would delete 94% of the data. |
| `agent` | 13.69% | fill with **0** | NaN means *booked without a travel agency* (direct booking). |
| `country` | 0.41% | fill with **'Unknown'** | Origin not recorded; a separate category keeps the 488 rows. |
| `children` | 4 rows | fill with **0** | The mode; 4 rows cannot change the distribution. |

**What would happen if we did not handle them?**
* scikit-learn models (KNN, MLP, Random Forest, Decision Tree in older versions, StandardScaler, PCA) **raise an error** on NaN, so the pipeline would simply not run.
* Dropping every row with a NaN (`dropna`) would delete **~94% of the bookings** because of `company`, leaving a small and biased sample (only company bookings).
* Filling with the mean would be wrong for `agent`/`company` because they are IDs, and it would hide the useful fact that the booking was made directly / privately.
* `total_guests` would be NaN for the 4 rows with missing `children`, so the engineered features would also be broken."""),
    md("## 2. Remove impossible records (team decision)\n- 180 *ghost bookings* with 0 adults, 0 children and 0 babies.\n- ADR (price) errors: the negative price and the single 5,400 EUR typo."),
    code("""ghost_mask = (df['adults'] + df['children'].fillna(0) + df['babies']) == 0
print('Ghost bookings (0 guests):', int(ghost_mask.sum()))
print('Negative ADR rows:', int((df['adr'] < 0).sum()), '| ADR > 5000 rows:', int((df['adr'] > 5000).sum()))
df = df[~ghost_mask]
df = df[(df['adr'] >= 0) & (df['adr'] <= 5000)]
print(f'Clean records remaining: {len(df):,}')"""),
    md("""## 3. Outliers
An **outlier** is an observation that lies far away from most of the other observations of the same variable. It can be
(a) an **error** (typo, test record, impossible value such as a negative price), or
(b) a **real but rare** case (a booking made 600 days ahead, a group of 50 people, a guest with 26 earlier cancellations).
Errors must be removed or corrected; real rare cases carry information and should usually be kept, because the model must also handle them in production.

We detect outliers with **two univariate methods** and one **multivariate method**:
1. **IQR (Tukey) rule:** a value is an outlier if it is below Q1 - 1.5·IQR or above Q3 + 1.5·IQR.
2. **Z-score rule:** a value is an outlier if |z| = |x - mean| / std > 3.
3. **Isolation Forest** (multivariate): isolates unusual *combinations* of values; we mark the 1% most isolated bookings."""),
    code("""out_cols = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
            'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
            'adr', 'required_car_parking_spaces', 'total_of_special_requests']
X_out = df[out_cols].fillna(0)
rows = []
iqr_any = np.zeros(len(df), dtype=bool)
z_any = np.zeros(len(df), dtype=bool)
for c in out_cols:
    s = X_out[c]
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_mask = (s < lo) | (s > hi)
    z = np.abs((s - s.mean()) / s.std())
    z_mask = z > 3
    iqr_any |= iqr_mask.values
    z_any |= z_mask.values
    rows.append({'feature': c, 'Q1': q1, 'Q3': q3, 'IQR lower': lo, 'IQR upper': hi,
                 'IQR outliers': int(iqr_mask.sum()), 'IQR %': round(iqr_mask.mean() * 100, 2),
                 'Z>3 outliers': int(z_mask.sum()), 'Z %': round(z_mask.mean() * 100, 2),
                 'max': s.max(), 'cancel rate of IQR outliers': round(df.loc[iqr_mask.values, 'is_canceled'].mean(), 3) if iqr_mask.any() else np.nan})
outliers = pd.DataFrame(rows).set_index('feature')
outliers.to_csv(f'{output_dir}/outlier_detection_iqr_zscore.csv')
display(outliers)
print(f'Rows with at least one IQR outlier:     {iqr_any.sum():,} ({iqr_any.mean():.1%} of the data)')
print(f'Rows with at least one Z-score outlier: {z_any.sum():,} ({z_any.mean():.1%} of the data)')"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
o = outliers[['IQR outliers', 'Z>3 outliers']].sort_values('IQR outliers', ascending=True)
o.plot(kind='barh', ax=axes[0], color=['#3182bd', '#e6550d'])
axes[0].set_title('Outliers per feature: IQR rule vs Z-score rule')
axes[0].set_xlabel('Number of rows flagged')
sns.boxplot(data=(X_out[['lead_time', 'adr', 'stays_in_week_nights', 'days_in_waiting_list']]), orient='h', ax=axes[1], color='#9ecae1')
axes[1].set_title('Box plots (points beyond the whiskers = IQR outliers)')
plt.tight_layout()
plt.savefig(f'{output_dir}/outlier_detection.png', dpi=300)
plt.show()"""),
    code("""iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
iso_flag = iso.fit_predict(X_out) == -1
print(f'Isolation Forest flagged {iso_flag.sum():,} bookings (1%).')
cmp = pd.DataFrame({
    'Method': ['IQR rule (any feature)', 'Z-score > 3 (any feature)', 'Isolation Forest (1%)', 'All data'],
    'Rows flagged': [iqr_any.sum(), z_any.sum(), iso_flag.sum(), len(df)],
    '% of data': [iqr_any.mean() * 100, z_any.mean() * 100, iso_flag.mean() * 100, 100],
    'Cancellation rate of flagged rows': [df.is_canceled[iqr_any].mean(), df.is_canceled[z_any].mean(),
                                          df.is_canceled[iso_flag].mean(), df.is_canceled.mean()]}).round(3)
cmp.to_csv(f'{output_dir}/outlier_methods_comparison.csv', index=False)
display(cmp)
print('Agreement: rows flagged by BOTH IQR and Z-score:', int((iqr_any & z_any).sum()))
print('Examples of the most isolated bookings:')
display(df.loc[iso_flag, out_cols].assign(score=iso.score_samples(X_out)[iso_flag]).sort_values('score').head(5))"""),
    md("""### How we handled the outliers and why
* **Removed (errors):** the 180 ghost bookings and the 2 impossible prices (negative ADR and 5,400 EUR). These cannot be real stays.
* **Kept (real rare cases):** all other flagged values. Reasons:
  1. The IQR rule flags a very large part of the data (see the printed percentage above) because many features are counts with mostly zeros (`babies`, `previous_cancellations`, `days_in_waiting_list`, `booking_changes`): every non-zero value becomes an "outlier". Deleting them would throw away a big share of the bookings.
  2. The flagged rows are **not random**: their cancellation rate differs strongly from the average (last column above). E.g. long lead times and earlier cancellations are among the *strongest* cancellation signals - removing them would remove exactly the information the model needs and would bias the test set.
  3. Our best models are **tree-based** (Decision Tree, Random Forest, XGBoost, LightGBM). Trees split on thresholds and are therefore not sensitive to extreme values.
  4. For distance/gradient based models (KNN, MLP) extreme values are softened by **standardisation** (next section).
* The Z-score rule is less aggressive than IQR, but it assumes a roughly normal distribution, which our skewed features do not have; this is another reason we used both methods only for *detection* and decided per feature."""),
    md("## 4. Eliminate target leakage & engineer features (team decision)\n- **Drop `reservation_status` and `reservation_status_date`**: they record what happened after the booking finished (e.g. 'Check-Out' or 'Canceled'). Using them would cause 100% target leakage.\n- **New features**: total stay nights, total guests, special-requests flag, booking-changes flag, room-type-changed flag."),
    code("""print(pd.crosstab(raw['reservation_status'], raw['is_canceled']))   # proof of leakage: status determines the target
df = df.drop(columns=['reservation_status', 'reservation_status_date'])

df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies']
df['has_special_requests'] = (df['total_of_special_requests'] > 0).astype(int)
df['has_booking_changes'] = (df['booking_changes'] > 0).astype(int)
df['room_type_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)

# Missing value imputation (see section 1)
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)
print(f'Missing values remaining: {df.isnull().sum().sum()}')
print(df[['total_stay_nights', 'total_guests', 'has_special_requests', 'has_booking_changes', 'room_type_changed', 'is_canceled']]
      .groupby('has_special_requests')['is_canceled'].mean().rename('cancel rate by has_special_requests'))"""),
    md("## 5. Stratified train / test split\n80% training, 20% test, preserving the cancellation proportion (`stratify=y`, `random_state=42`)."),
    code("""X = df.drop(columns=['is_canceled'])
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f'Training instances: {X_train.shape[0]:,} (Cancel rate: {y_train.mean():.2%})')
print(f'Testing instances:  {X_test.shape[0]:,} (Cancel rate: {y_test.mean():.2%})')"""),
    md("""## 6. Scaling and encoding
* **Numerical features -> `StandardScaler`** (z = (x - mean) / std, mean and std learned on the training set only). The features have very different units and ranges (lead time up to 700 days, ADR in EUR, counts 0-10, flags 0/1). Without scaling, distance-based **KNN** would be dominated by `lead_time` and `adr`, and the gradient-based **MLP** would converge slowly. Trees do not need scaling, but it does not hurt them, so one common matrix is used for all models.
* **Categorical features -> `OneHotEncoder`** (one 0/1 column per category, `handle_unknown='ignore'` so an unseen category in the test set becomes all zeros). The categories have **no natural order** (`hotel`, `meal`, `country`, `market_segment` ...), so label/ordinal encoding would invent a false order (e.g. PRT < GBR). One-hot encoding avoids this.
* `agent` and `company` are IDs stored as numbers; they were kept numeric (scaled) as in the original team pipeline - tree models can still isolate single IDs by thresholds.
* The `ColumnTransformer` is **fit only on the training data** and then applied to the test data, so no information from the test set leaks into training."""),
    code("""cat_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()
print(f'Numerical (scaled) columns: {len(num_cols)}')
print(f'Categorical (one-hot) columns: {len(cat_cols)}')
print(f'Transformed feature matrix: {X_train_processed.shape[1]} features.')

enc = pd.DataFrame({'Categorical feature': cat_cols,
                    'Categories (= new 0/1 columns)': [len(c) for c in preprocessor.named_transformers_['cat'].categories_]})
enc.to_csv(f'{output_dir}/one_hot_encoding_summary.csv', index=False)
display(enc)"""),
    code("""scaler = preprocessor.named_transformers_['num']
show = ['lead_time', 'adr', 'total_stay_nights', 'total_of_special_requests']
Xtr_s = pd.DataFrame(X_train_processed[:, :len(num_cols)], columns=num_cols)
scale_tab = pd.DataFrame({'mean before': X_train[num_cols].mean(), 'std before': X_train[num_cols].std(),
                          'learned mean (scaler)': scaler.mean_, 'learned std (scaler)': scaler.scale_,
                          'mean after': Xtr_s.mean().values, 'std after': Xtr_s.std().values}).round(3)
scale_tab.to_csv(f'{output_dir}/scaling_before_after.csv')
display(scale_tab)

fig, axes = plt.subplots(2, 4, figsize=(17, 6))
for i, c in enumerate(show):
    sns.histplot(X_train[c], bins=40, ax=axes[0, i], color='#9ecae1')
    axes[0, i].set_title(f'{c} - original')
    sns.histplot(Xtr_s[c], bins=40, ax=axes[1, i], color='#31a354')
    axes[1, i].set_title(f'{c} - standardised')
plt.tight_layout()
plt.savefig(f'{output_dir}/scaling_before_after.png', dpi=300)
plt.show()"""),
    md("## 7. Save processed datasets & preprocessor"),
    code("""X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
X_train_df.to_parquet(f'{processed_dir}/X_train.parquet')
X_test_df.to_parquet(f'{processed_dir}/X_test.parquet')
pd.DataFrame(y_train).to_csv(f'{processed_dir}/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv(f'{processed_dir}/y_test.csv', index=False)
joblib.dump(preprocessor, f'{models_dir}/preprocessor.joblib')

summary_table = pd.DataFrame({
    'Metric': ['Initial Raw Rows', 'Ghost bookings removed', 'ADR errors removed', 'Cleaned Records',
               'Training Rows (80%)', 'Testing Rows (20%)', 'Numerical features (scaled)', 'Categorical features (one-hot)', 'Total Predictors'],
    'Value': [len(raw), int(ghost_mask.sum()), len(raw) - int(ghost_mask.sum()) - len(df), len(df),
              len(X_train), len(X_test), len(num_cols), len(cat_cols), len(feature_names)]
})
summary_table.to_csv(f'{output_dir}/preprocessing_summary.csv', index=False)
print('Data preprocessing complete and artifacts saved!')
summary_table"""),
]

# =====================================================================================  09
nb09 = [
    md("""# 09. Feature Selection and Dimensionality Reduction (PCA)
Guideline section **VI**. We
1. select features with **five different techniques** from the three families (filter, wrapper, embedded),
2. retrain four algorithms on each subset and **compare before and after feature selection**,
3. point out which selection **boosted** performance,
4. reduce dimensionality with **PCA** and compare before and after.

All feature selectors are fit on the **training set only**; the test set is used only for the final scores."""),
    code(COMMON + """import time
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

results_dir = '../results/09_feature_selection_and_pca' if os.path.exists('../results') else 'results/09_feature_selection_and_pca'
data_dir = '../data/processed' if os.path.exists('../data/processed') else 'data/processed'
os.makedirs(results_dir, exist_ok=True)

X_train = pd.read_parquet(f'{data_dir}/X_train.parquet')
X_test = pd.read_parquet(f'{data_dir}/X_test.parquet')
y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
N_ALL = X_train.shape[1]
print(f'Starting with all {N_ALL} features.')"""),
    md("""## 1. Models used for the comparison
We use the **tuned hyper-parameters found in notebooks 03-07** so that "before" means the best model we already have:
* Decision Tree (`max_depth=16, min_samples_leaf=8`)
* KNN (`n_neighbors=15, weights='distance'`)
* Random Forest (`n_estimators=150, max_depth=None, min_samples_leaf=1`)
* LightGBM (`num_leaves=127, learning_rate=0.2, n_estimators=150`) - the best model of the project."""),
    code("""import joblib
models_dir = '../models' if os.path.exists('../models') else 'models'
def tuned_params(name, fallback):
    try:
        return joblib.load(f'{models_dir}/{name}_tuned.joblib').get_params()
    except Exception:
        return fallback

MODELS = {
    'Decision Tree': lambda: DecisionTreeClassifier(**tuned_params('decision_tree', {'max_depth': 16, 'min_samples_leaf': 8, 'random_state': 42})),
    'KNN': lambda: KNeighborsClassifier(**{**tuned_params('knn', {'n_neighbors': 15, 'weights': 'distance'}), 'n_jobs': -1}),
    'Random Forest': lambda: RandomForestClassifier(**{**tuned_params('random_forest', {'n_estimators': 150, 'random_state': 42}), 'n_jobs': -1}),
    'LightGBM': lambda: LGBMClassifier(**{**tuned_params('lightgbm', {'num_leaves': 127, 'learning_rate': 0.2, 'n_estimators': 150, 'random_state': 42}), 'verbose': -1, 'n_jobs': -1}),
}
for k, f in MODELS.items():
    m = f(); print(k, {p: v for p, v in m.get_params().items() if p in ('max_depth', 'min_samples_leaf', 'n_neighbors', 'weights', 'n_estimators', 'num_leaves', 'learning_rate')})

def evaluate(Xtr, Xte, label, n_features):
    rows = []
    for name, make in MODELS.items():
        t = time.time()
        m = make().fit(Xtr, y_train)
        fit_time = time.time() - t
        prob = m.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({'Setting': label, 'Features': n_features, 'Model': name,
                     'Accuracy': accuracy_score(y_test, pred), 'Precision': precision_score(y_test, pred),
                     'Recall': recall_score(y_test, pred), 'F1-Score': f1_score(y_test, pred),
                     'ROC-AUC': roc_auc_score(y_test, prob), 'Train time (s)': round(fit_time, 2)})
    return rows"""),
    md("## 2. Before feature selection: all features"),
    code("""results = evaluate(X_train, X_test, 'All features', N_ALL)
pd.DataFrame(results).round(4)"""),
    md("""## 3. Feature selection techniques
| Family | Technique | Idea |
|---|---|---|
| Filter | **Variance threshold** | drop near-constant columns (variance < 0.001, e.g. one-hot columns of very rare countries) |
| Filter | **ANOVA F-test** (`SelectKBest(f_classif)`) | rank each feature by how much its mean differs between cancelled and kept bookings |
| Filter | **Mutual information** | rank each feature by the information it shares with the target (captures non-linear relations) |
| Wrapper | **RFE (Recursive Feature Elimination)** with a Decision Tree | repeatedly train a model and remove the 10 least important features until K remain |
| Embedded | **L1 (Lasso) Logistic Regression** | the L1 penalty pushes the weights of useless features to exactly zero; non-zero weights are kept |
| Embedded | **LightGBM feature importance** (team's original method) | keep the K features used most often for splits |

For the ranking methods we keep **K = 50** features (about 20% of the columns), the same size the team used originally."""),
    code("""K = 50
selected = {}
timing = {}

t = time.time(); vt = VarianceThreshold(threshold=1e-3).fit(X_train)
selected['Variance threshold'] = X_train.columns[vt.get_support()].tolist(); timing['Variance threshold'] = time.time() - t

t = time.time(); anova = SelectKBest(f_classif, k=K).fit(X_train, y_train)
selected['ANOVA F-test (k=50)'] = X_train.columns[anova.get_support()].tolist(); timing['ANOVA F-test (k=50)'] = time.time() - t

t = time.time()
sub = np.random.RandomState(42).choice(len(X_train), size=30000, replace=False)
discrete = X_train.columns.str.startswith('cat__')
mi = pd.Series(mutual_info_classif(X_train.iloc[sub], y_train[sub], discrete_features=discrete, random_state=42), index=X_train.columns).sort_values(ascending=False)
selected['Mutual information (k=50)'] = mi.head(K).index.tolist(); timing['Mutual information (k=50)'] = time.time() - t

t = time.time()
rfe = RFE(DecisionTreeClassifier(max_depth=16, min_samples_leaf=8, random_state=42), n_features_to_select=K, step=10).fit(X_train, y_train)
selected['RFE - Decision Tree (k=50)'] = X_train.columns[rfe.get_support()].tolist(); timing['RFE - Decision Tree (k=50)'] = time.time() - t

t = time.time()
l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, max_iter=500, random_state=42).fit(X_train, y_train)
coef = pd.Series(np.abs(l1.coef_[0]), index=X_train.columns)
selected['L1 Logistic Regression'] = coef[coef > 1e-6].index.tolist(); timing['L1 Logistic Regression'] = time.time() - t

t = time.time()
lgb_imp = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1).fit(X_train, y_train)
importances = pd.Series(lgb_imp.feature_importances_, index=X_train.columns).sort_values(ascending=False)
selected['LightGBM importance (k=50)'] = importances.head(K).index.tolist(); timing['LightGBM importance (k=50)'] = time.time() - t

summary = pd.DataFrame({'Technique': list(selected), 'Family': ['Filter', 'Filter', 'Filter', 'Wrapper', 'Embedded', 'Embedded'],
                        'Selected features': [len(v) for v in selected.values()],
                        'Selection time (s)': [round(timing[k], 1) for k in selected]})
summary.to_csv(f'{results_dir}/feature_selection_techniques.csv', index=False)
summary"""),
    code("""# Top 15 features according to three different techniques
anova_scores = pd.Series(anova.scores_, index=X_train.columns).sort_values(ascending=False)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (title, s, col) in zip(axes, [('ANOVA F-score', anova_scores, '#3182bd'), ('Mutual information', mi, '#31a354'),
                                      ('LightGBM split importance', importances, '#e6550d')]):
    s.head(15)[::-1].plot(kind='barh', ax=ax, color=col)
    ax.set_title(f'Top 15 - {title}')
plt.tight_layout()
plt.savefig(f'{results_dir}/top_features_three_methods.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
importances.head(15).plot(kind='barh', color='#1f77b4')
plt.title('Top 15 Most Important Features for Cancellation Prediction (LightGBM)')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{results_dir}/top_15_features.png', dpi=300)
plt.show()

ranking = pd.DataFrame({'ANOVA F': anova_scores, 'Mutual information': mi, 'LightGBM importance': importances,
                        'L1 |coef|': coef, 'RFE rank': pd.Series(rfe.ranking_, index=X_train.columns)})
ranking.sort_values('ANOVA F', ascending=False).round(4).to_csv(f'{results_dir}/feature_rankings_all_methods.csv')

common = set.intersection(*[set(selected[k]) for k in ['ANOVA F-test (k=50)', 'Mutual information (k=50)', 'RFE - Decision Tree (k=50)', 'LightGBM importance (k=50)']])
print(f'{len(common)} features are chosen by all four k=50 techniques:')
print(sorted(common))"""),
    md("## 4. After feature selection: retrain the four models on every subset"),
    code("""for name, cols in selected.items():
    results += evaluate(X_train[cols], X_test[cols], name, len(cols))
fs_df = pd.DataFrame(results)
fs_df.round(4).to_csv(f'{results_dir}/feature_selection_results.csv', index=False)

base = fs_df[fs_df.Setting == 'All features'].set_index('Model')
fs_df['Delta F1 vs all'] = fs_df.apply(lambda r: r['F1-Score'] - base.loc[r['Model'], 'F1-Score'], axis=1)
fs_df['Delta Acc vs all'] = fs_df.apply(lambda r: r['Accuracy'] - base.loc[r['Model'], 'Accuracy'], axis=1)
fs_df['Delta AUC vs all'] = fs_df.apply(lambda r: r['ROC-AUC'] - base.loc[r['Model'], 'ROC-AUC'], axis=1)
fs_df.round(4).to_csv(f'{results_dir}/feature_selection_results.csv', index=False)
pivot = fs_df.pivot_table(index='Setting', columns='Model', values='F1-Score').loc[['All features'] + list(selected)]
pivot['Features'] = fs_df.groupby('Setting')['Features'].first()
pivot.round(4)"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
pv = fs_df.pivot_table(index='Setting', columns='Model', values='F1-Score').loc[['All features'] + list(selected)]
pv.plot(kind='bar', ax=axes[0], width=0.8)
axes[0].set_ylim(0.70, 0.88)
axes[0].legend(loc='lower center', ncol=4, fontsize=8)
axes[0].set_title('F1-score before (All features) and after each feature selection')
axes[0].set_ylabel('F1-score (test set)')
axes[0].tick_params(axis='x', rotation=30)
for t in axes[0].get_xticklabels(): t.set_ha('right')
dl = fs_df[fs_df.Setting != 'All features'].pivot_table(index='Setting', columns='Model', values='Delta F1 vs all').loc[list(selected)]
sns.heatmap(dl * 100, annot=True, fmt='+.2f', cmap='RdYlGn', center=0, ax=axes[1], cbar_kws={'label': 'F1 change (percentage points)'})
axes[1].set_title('Change in F1 vs all features (green = boosted)')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(f'{results_dir}/feature_selection_comparison.png', dpi=300)
plt.show()

boosted = fs_df[(fs_df.Setting != 'All features') & (fs_df['Delta F1 vs all'] > 0)].sort_values('Delta F1 vs all', ascending=False)
boosted[['Setting', 'Model', 'Features', 'F1-Score', 'Delta F1 vs all', 'Delta Acc vs all', 'Delta AUC vs all']].round(4).to_csv(f'{results_dir}/feature_selection_boosted.csv', index=False)
print('Feature selections that BOOSTED the F1-score:')
boosted[['Setting', 'Model', 'Features', 'F1-Score', 'Delta F1 vs all', 'Delta Acc vs all', 'Delta AUC vs all']].round(4)"""),
    md("### How many features are really needed? (LightGBM, ANOVA vs LightGBM-importance ranking)"),
    code("""curve = []
for k in [5, 10, 20, 30, 50, 75, 100, 150, N_ALL]:
    for rank_name, rank in [('ANOVA F-test', anova_scores), ('LightGBM importance', importances)]:
        cols = rank.head(k).index
        m = MODELS['LightGBM']().fit(X_train[cols], y_train)
        prob = m.predict_proba(X_test[cols])[:, 1]
        curve.append({'k': k, 'Ranking': rank_name, 'F1-Score': f1_score(y_test, (prob >= .5).astype(int)), 'ROC-AUC': roc_auc_score(y_test, prob)})
curve = pd.DataFrame(curve)
curve.round(4).to_csv(f'{results_dir}/n_features_curve_lightgbm.csv', index=False)
plt.figure(figsize=(9, 4.5))
sns.lineplot(data=curve, x='k', y='F1-Score', hue='Ranking', marker='o')
plt.xscale('log')
plt.title('LightGBM F1-score vs number of selected features')
plt.xlabel('Number of features kept (log scale)')
plt.tight_layout()
plt.savefig(f'{results_dir}/n_features_curve.png', dpi=300)
plt.show()
curve.pivot(index='k', columns='Ranking', values='F1-Score').round(4)"""),
    md("""## 5. Dimensionality reduction with PCA
PCA finds new orthogonal axes (principal components) that capture the largest variance of the (standardised) data, and projects every booking onto the first *n* of them. We fit PCA on the **training set only**."""),
    code("""pca_full = PCA(random_state=42).fit(X_train)
cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
need = {p: int(np.argmax(cum_var >= p) + 1) for p in (80, 90, 95, 99)}
print('Components needed to keep 80/90/95/99% of the variance:', need)
pd.Series(need, name='components').to_csv(f'{results_dir}/pca_components_needed.csv', header=['components'])

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
axes[0].bar(range(1, 31), pca_full.explained_variance_ratio_[:30] * 100, color='#9ecae1', label='per component')
axes[0].plot(range(1, 31), cum_var[:30], marker='o', color='#2ca02c', label='cumulative')
axes[0].set_title('PCA scree plot (first 30 components)')
axes[0].set_xlabel('Principal component')
axes[0].set_ylabel('Explained variance (%)')
axes[0].legend()
axes[1].plot(range(1, len(cum_var) + 1), cum_var, color='#2ca02c')
for p, n in need.items():
    axes[1].axhline(p, ls='--', lw=.8, color='grey'); axes[1].annotate(f'{p}% -> {n} PCs', (n, p), xytext=(5, -12), textcoords='offset points', fontsize=8)
axes[1].set_title('Cumulative explained variance (all components)')
axes[1].set_xlabel('Number of components')
plt.tight_layout()
plt.savefig(f'{results_dir}/pca_scree_plot.png', dpi=300)
plt.show()

Z = pca_full.transform(X_test)[:, :2]
plt.figure(figsize=(7, 5.5))
idx = np.random.RandomState(0).choice(len(Z), 6000, replace=False)
plt.scatter(Z[idx, 0], Z[idx, 1], c=y_test[idx], cmap='coolwarm', s=6, alpha=.5)
plt.title('Test bookings on the first two principal components (red = cancelled)')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.tight_layout()
plt.savefig(f'{results_dir}/pca_2d_projection.png', dpi=300)
plt.show()"""),
    code("""pca_rows = []
for n in sorted({10, 30, need[90], need[95]}):
    pca = PCA(n_components=n, random_state=42).fit(X_train)
    Xtr_p, Xte_p = pca.transform(X_train), pca.transform(X_test)
    label = f'PCA {n} components ({pca.explained_variance_ratio_.sum() * 100:.1f}% variance)'
    pca_rows += evaluate(Xtr_p, Xte_p, label, n)
pca_df = pd.concat([fs_df[fs_df.Setting == 'All features'][pd.DataFrame(pca_rows).columns], pd.DataFrame(pca_rows)], ignore_index=True)
pca_df['Delta F1 vs all'] = pca_df.apply(lambda r: r['F1-Score'] - base.loc[r['Model'], 'F1-Score'], axis=1)
pca_df['Delta AUC vs all'] = pca_df.apply(lambda r: r['ROC-AUC'] - base.loc[r['Model'], 'ROC-AUC'], axis=1)
pca_df.round(4).to_csv(f'{results_dir}/pca_results.csv', index=False)
pca_df.pivot_table(index='Setting', columns='Model', values='F1-Score', sort=False).round(4)"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, metric in zip(axes, ['F1-Score', 'ROC-AUC']):
    pv = pca_df.pivot_table(index='Setting', columns='Model', values=metric, sort=False)
    pv.plot(kind='bar', ax=ax, width=.8)
    ax.set_ylim(0.6 if metric == 'F1-Score' else 0.8, 0.9 if metric == 'F1-Score' else 1.0)
    ax.legend(loc='lower center', ncol=4, fontsize=8)
    ax.set_title(f'{metric}: before PCA (All features) vs after PCA')
    ax.tick_params(axis='x', rotation=20)
    for t in ax.get_xticklabels(): t.set_ha('right')
plt.tight_layout()
plt.savefig(f'{results_dir}/pca_comparison.png', dpi=300)
plt.show()"""),
    md("## 6. Overall comparison (best model LightGBM): all features vs feature selection vs PCA"),
    code("""allres = pd.concat([fs_df, pca_df[pca_df.Setting != 'All features']], ignore_index=True)
allres.round(4).to_csv(f'{results_dir}/all_reduction_results.csv', index=False)
lgb = allres[allres.Model == 'LightGBM'][['Setting', 'Features', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Train time (s)']]
lgb.round(4).to_csv(f'{results_dir}/feature_selection_vs_pca.csv', index=False)
lgb.round(4)"""),
    md("SUMMARY_09"),
]

# =====================================================================================  10
nb10 = [
    md("""# 10. Final Model Comparison
Guideline section **V** (compare algorithms, before/after hyper-parameter optimisation, best performing algorithm).
We collect the default and tuned test results of all six algorithms from notebooks 03-08 and evaluate the saved models on the same 20% test set (23,842 bookings)."""),
    code(COMMON + """import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss

results_dir = '../results/10_model_comparison' if os.path.exists('../results') else 'results/10_model_comparison'
models_dir = '../models' if os.path.exists('../models') else 'models'
data_dir = '../data/processed' if os.path.exists('../data/processed') else 'data/processed'
base_results = '../results' if os.path.exists('../results') else 'results'
os.makedirs(results_dir, exist_ok=True)

X_test = pd.read_parquet(f'{data_dir}/X_test.parquet')
y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
model_folders = {'KNN': ('03_knn', 'knn'), 'Decision Tree': ('04_decision_tree', 'decision_tree'),
                 'Random Forest': ('05_random_forest', 'random_forest'), 'XGBoost': ('06_xgboost', 'xgboost'),
                 'LightGBM': ('07_lightgbm', 'lightgbm'), 'MLP Neural Net': ('08_mlp', 'mlp')}"""),
    md("## 1. Before vs after hyper-parameter optimisation (all six models)"),
    code("""rows = []
for model_name, (folder, _) in model_folders.items():
    m = pd.read_csv(f'{base_results}/{folder}/metrics_comparison.csv').set_index('Metric')
    for variant, col in [('Default', 'Default (No Tuning)'), ('Tuned', 'Tuned')]:
        rows.append({'Model': model_name, 'Variant': variant, **m[col].to_dict()})
both = pd.DataFrame(rows)
both.to_csv(f'{results_dir}/default_vs_tuned_all_models.csv', index=False)
display(both.pivot(index='Model', columns='Variant', values=['Accuracy', 'F1-Score', 'ROC-AUC']).round(4))

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, metric in zip(axes, ['Accuracy', 'F1-Score', 'ROC-AUC']):
    pv = both.pivot(index='Model', columns='Variant', values=metric).loc[list(model_folders)]
    pv.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71'], width=.75)
    ax.set_ylim(0.75 if metric != 'ROC-AUC' else 0.85, 0.98)
    ax.set_title(f'{metric}: default vs tuned')
    ax.tick_params(axis='x', rotation=30)
    ax.set_xlabel('')
plt.tight_layout()
plt.savefig(f'{results_dir}/default_vs_tuned_all_models.png', dpi=300)
plt.show()"""),
    md("## 2. Master leaderboard of the tuned models"),
    code("""leaderboard_df = both[both.Variant == 'Tuned'].drop(columns='Variant')
probs = {}
for model_name, (_, key) in model_folders.items():
    probs[model_name] = joblib.load(f'{models_dir}/{key}_tuned.joblib').predict_proba(X_test)[:, 1]
leaderboard_df['PR-AUC (AP)'] = leaderboard_df['Model'].map(lambda n: average_precision_score(y_test, probs[n]))
leaderboard_df['Brier score'] = leaderboard_df['Model'].map(lambda n: brier_score_loss(y_test, probs[n]))
leaderboard_df = leaderboard_df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
leaderboard_df.index = leaderboard_df.index + 1
leaderboard_df.to_csv(f'{results_dir}/master_model_leaderboard.csv', index_label='Rank')
print('=== MASTER MODEL LEADERBOARD (tuned models, test set) ===')
leaderboard_df.round(4)"""),
    code("""plt.figure(figsize=(12, 5))
lb = leaderboard_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
lb.plot(kind='bar', width=.8, ax=plt.gca(), colormap='tab10')
plt.ylim(0.7, 1.0)
plt.title('Comparison of All Six Tuned Classification Models (test set)')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='upper right', ncol=5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{results_dir}/model_leaderboard_barchart.png', dpi=300)
plt.show()"""),
    md("## 3. ROC and Precision-Recall curves of all six tuned models"),
    code("""colors = dict(zip(model_folders, ['#8c564b', '#9467bd', '#ff7f0e', '#d62728', '#2ca02c', '#1f77b4']))
plt.figure(figsize=(7, 6))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.3f})', lw=2, color=colors[name])
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.50)')
plt.title('ROC Curves - all six tuned models')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(loc='lower right'); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig(f'{results_dir}/combined_roc_curves.png', dpi=300); plt.show()

plt.figure(figsize=(7, 6))
for name, p in probs.items():
    pc, rc, _ = precision_recall_curve(y_test, p)
    plt.plot(rc, pc, label=f'{name} (AP = {average_precision_score(y_test, p):.3f})', lw=2, color=colors[name])
plt.axhline(y_test.mean(), color='black', linestyle='--', label=f'Baseline ({y_test.mean()*100:.1f}%)')
plt.title('Precision-Recall Curves - all six tuned models')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.05])
plt.legend(loc='lower left'); plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig(f'{results_dir}/combined_pr_curves.png', dpi=300); plt.show()"""),
    md("## 4. Confusion matrices of all six tuned models"),
    code("""fig, axes = plt.subplots(2, 3, figsize=(15, 9))
cm_rows = []
for ax, (name, p) in zip(axes.ravel(), probs.items()):
    pred = (p >= .5).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap='Blues', colorbar=False, values_format=',')
    ax.set_title(name); ax.grid(False)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cm_rows.append({'Model': name, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
plt.suptitle('Confusion matrices (0 = not cancelled, 1 = cancelled)', fontsize=14)
plt.tight_layout(); plt.savefig(f'{results_dir}/confusion_matrices_all.png', dpi=300); plt.show()
pd.DataFrame(cm_rows).to_csv(f'{results_dir}/confusion_matrices_all.csv', index=False)
pd.DataFrame(cm_rows)"""),
    md("## 5. What drives the best model? (LightGBM feature importance, gain)"),
    code("""best = joblib.load(f'{models_dir}/lightgbm_tuned.joblib')
gain = pd.Series(best.booster_.feature_importance(importance_type='gain'), index=X_test.columns)
gain = (gain / gain.sum() * 100).sort_values(ascending=False)
gain.round(3).to_csv(f'{results_dir}/best_model_feature_importance_gain.csv', header=['gain_%'])
plt.figure(figsize=(9, 6))
gain.head(15)[::-1].plot(kind='barh', color='#2ca02c')
plt.title('Tuned LightGBM - top 15 features by gain (% of total)')
plt.xlabel('Share of total gain (%)')
plt.tight_layout(); plt.savefig(f'{results_dir}/best_model_feature_importance.png', dpi=300); plt.show()
gain.head(15).round(2)"""),
    md("SUMMARY_10"),
]

if __name__ == "__main__":
    save("01_data_exploration.ipynb", nb01)
    save("02_data_preprocessing.ipynb", nb02)
    save("09_feature_selection_and_pca.ipynb", nb09)
    save("10_model_comparison.ipynb", nb10)
