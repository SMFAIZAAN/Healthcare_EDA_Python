"""
Healthcare Patient Analysis
============================
Runs full EDA on healthcare_dataset.csv.
Outputs: 8 charts to ./charts/, prints key stats.

Requirements: pandas, matplotlib
Usage: python3 analysis.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH  = 'healthcare_dataset.csv'
CHARTS_DIR = 'charts'
os.makedirs(CHARTS_DIR, exist_ok=True)

COLORS6 = ['#2563EB','#10B981','#F59E0B','#EF4444','#8B5CF6','#06B6D4']
COLORS3 = ['#2563EB','#F59E0B','#10B981']

# ── Load & clean ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date']    = pd.to_datetime(df['Discharge Date'])
df['LOS']              = (df['Discharge Date'] - df['Date of Admission']).dt.days
df_clean = df[df['Billing Amount'] > 0].copy()

neg_count = (df['Billing Amount'] < 0).sum()
print(f"Records         : {len(df):,}")
print(f"Negative billing: {neg_count} flagged and excluded from aggregations")
print(f"Avg billing     : ${df_clean['Billing Amount'].mean():,.2f}")
print(f"Avg LOS         : {df['LOS'].mean():.1f} days")
print(f"Avg age         : {df['Age'].mean():.1f} years")
print()

def savefig(name):
    plt.savefig(f'{CHARTS_DIR}/{name}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# Chart 1 — Billing by condition
billing_cond = df_clean.groupby('Medical Condition')['Billing Amount'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(billing_cond.index, billing_cond.values, color=COLORS6, edgecolor='white', height=0.6)
for bar, val in zip(bars, billing_cond.values):
    ax.text(val+150, bar.get_y()+bar.get_height()/2, f'${val:,.0f}', va='center', fontsize=10)
ax.set_xlabel('Average Billing Amount (USD)')
ax.set_title('Average Billing Amount by Medical Condition', fontweight='bold')
ax.set_xlim(0, billing_cond.max()*1.14)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); savefig('01_billing_by_condition.png')

# Chart 2 — Patient volume by condition
cond_counts = df['Medical Condition'].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(cond_counts.index, cond_counts.values, color=COLORS6[::-1], edgecolor='white', height=0.6)
for bar, val in zip(ax.patches, cond_counts.values):
    ax.text(val+40, bar.get_y()+bar.get_height()/2, f'{val:,}', va='center', fontsize=10)
ax.set_xlabel('Number of Patients')
ax.set_title('Patient Volume by Medical Condition', fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); savefig('02_patients_by_condition.png')

# Chart 3 — Admission type pie
adm_counts = df['Admission Type'].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
ax.pie(adm_counts.values, labels=adm_counts.index, colors=COLORS3,
       autopct='%1.1f%%', startangle=90,
       wedgeprops={'edgecolor':'white','linewidth':2})
ax.set_title(f'Admission Type Breakdown\n(n = {len(df):,})', fontweight='bold')
plt.tight_layout(); savefig('03_admission_type.png')

# Chart 4 — Test results by condition
pivot_pct = (df.groupby(['Medical Condition','Test Results']).size()
               .unstack(fill_value=0)
               .div(df.groupby('Medical Condition').size(), axis=0) * 100)
pivot_pct = pivot_pct[['Normal','Abnormal','Inconclusive']]
x = np.arange(len(pivot_pct)); w = 0.25
fig, ax = plt.subplots(figsize=(10, 5.5))
for i,(col,color) in enumerate(zip(['Normal','Abnormal','Inconclusive'],
                                    ['#10B981','#EF4444','#F59E0B'])):
    bars = ax.bar(x+i*w, pivot_pct[col], w, label=col, color=color, edgecolor='white')
ax.set_xticks(x+w); ax.set_xticklabels(pivot_pct.index)
ax.set_ylabel('Percentage (%)'); ax.set_ylim(0, 45)
ax.set_title('Test Result Distribution by Medical Condition', fontweight='bold')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); savefig('04_test_results_by_condition.png')

# Chart 5 — LOS histogram
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df['LOS'], bins=30, color='#2563EB', edgecolor='white', alpha=0.9)
ax.axvline(df['LOS'].mean(), color='#EF4444', linewidth=2, linestyle='--', label=f'Mean: {df["LOS"].mean():.1f} days')
ax.axvline(df['LOS'].median(), color='#F59E0B', linewidth=2, linestyle='--', label=f'Median: {int(df["LOS"].median())} days')
ax.set_xlabel('Length of Stay (Days)'); ax.set_ylabel('Number of Patients')
ax.set_title('Distribution of Length of Stay', fontweight='bold')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); savefig('05_los_distribution.png')

# Chart 6 — Billing by insurance
ins_billing = df_clean.groupby('Insurance Provider')['Billing Amount'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(ins_billing.index, ins_billing.values, color=COLORS6[:5], edgecolor='white', width=0.55)
for bar, val in zip(bars, ins_billing.values):
    ax.text(bar.get_x()+bar.get_width()/2, val+60, f'${val:,.0f}', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Average Billing Amount (USD)')
ax.set_title('Average Billing Amount by Insurance Provider', fontweight='bold')
ax.set_ylim(0, ins_billing.max()*1.1)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); savefig('06_billing_by_insurance.png')

# Chart 7 — Age distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df['Age'], bins=25, color='#8B5CF6', edgecolor='white', alpha=0.9)
ax.axvline(df['Age'].mean(), color='#EF4444', linewidth=2, linestyle='--', label=f'Mean: {df["Age"].mean():.1f} yrs')
ax.set_xlabel('Age (Years)'); ax.set_ylabel('Number of Patients')
ax.set_title('Patient Age Distribution', fontweight='bold')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); savefig('07_age_distribution.png')

# Chart 8 — LOS by admission (box)
fig, ax = plt.subplots(figsize=(8, 5))
data_by_type = [df[df['Admission Type']==t]['LOS'].values for t in ['Elective','Urgent','Emergency']]
bp = ax.boxplot(data_by_type, tick_labels=['Elective','Urgent','Emergency'],
                patch_artist=True, widths=0.5,
                medianprops={'color':'white','linewidth':2})
for patch, color in zip(bp['boxes'], COLORS3):
    patch.set_facecolor(color)
ax.set_ylabel('Length of Stay (Days)')
ax.set_title('Length of Stay by Admission Type', fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); savefig('08_los_by_admission.png')

print("All charts saved to ./charts/")
