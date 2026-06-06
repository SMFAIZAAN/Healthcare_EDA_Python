# 🏥 Healthcare Data Analysis — End-to-End EDA Project

_End-to-end patient analytics across 55,500 hospital records — cleaned in Python, explored across 10 business questions, and every visual finding statistically validated with T-tests, ANOVA, Chi-square, and linear regression before being reported as a conclusion._

---

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#business-problem">Business Problem</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#data-cleaning">Data Cleaning & Preparation</a>
- <a href="#eda">Exploratory Data Analysis (EDA)</a>
- <a href="#pipeline">Python Pipeline & Key Analyses</a>
- <a href="#sample-outputs">Sample Outputs</a>
- <a href="#how-to-run">How to Run This Project</a>
- <a href="#recommendations">Final Recommendations</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>

This project applies a six-stage analytical framework to 55,500 hospital patient records from a Kaggle synthetic healthcare dataset. The full pipeline runs from raw CSV through Python-based cleaning and feature engineering, seven-dimension EDA across 13 charts, IQR and Z-score outlier detection, three statistical hypothesis tests, and a linear regression model. Summer is the peak admission season. The overall abnormal test rate is **33.6%** — 18,627 patients. Three statistical findings stand out: billing is independent of gender (T-test p=0.295), condition differences in billing are borderline-insignificant (ANOVA p=0.052), and — unlike smaller synthetic datasets — admission type and medical condition are **statistically related** (Chi-square p=0.036). Regression R²≈0 confirms billing is not predictable from available clinical features. Each finding maps to a concrete operational recommendation.

---

<h2><a class="anchor" id="business-problem"></a>Business Problem</h2>

Healthcare organisations collect patient data at every touchpoint but rarely analyse it systematically. A patient admitted across multiple conditions gets the same operational treatment as any other. A system with 33.6% abnormal test results goes unaudited. This project answers ten business questions:

1. **What is the demographic profile of admitted patients?**
2. **Which conditions are most prevalent — and how do they vary by age group and gender?**
3. **Which conditions carry the highest abnormal test result rates?**
4. **What are the primary billing cost drivers — condition, admission type, or length of stay?**
5. **How does length of stay vary across conditions and admission types?**
6. **Which insurance providers are linked to higher-cost patient profiles?**
7. **Where is peak demand — and how stable are year-over-year admission volumes?**
8. **Are there billing or LOS outliers requiring special attention?**
9. **Do clinical factors statistically drive billing — or are visual differences just noise?**
10. **Which features can be used to predict billing amount?**

---

<h2><a class="anchor" id="dataset"></a>Dataset</h2>

| Detail | Value |
|---|---|
| **Dataset** | [Healthcare Dataset — Kaggle (prasad22)](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) |
| **Rows (raw)** | 55,500 patient records |
| **Rows (after cleaning)** | 55,392 valid billing records (108 negative excluded); 534 duplicate rows flagged |
| **Unique patients** | 55,500 |
| **Period** | 2019 – 2024 |
| **Region** | Synthetic — modelled on real-world hospital data |
| **Format** | `.csv` loaded directly via `pd.read_csv()` |

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

| Tool | Purpose |
|---|---|
| Python (Pandas, NumPy) | Data cleaning and feature engineering |
| Matplotlib | Static EDA charts (13 visualisations) |
| SciPy | Statistical hypothesis testing (T-test, ANOVA, Chi-square) |
| Scikit-learn | Linear regression, train/test split, R², MAE |
| Jupyter Notebook | Analysis environment |
| MySQL 8.0 | SQL query design and validation |

---

<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
healthcare-analysis/
│
├── healthcare_analysis.ipynb       # Full notebook: cleaning → EDA → stats → regression
├── healthcare_dataset.csv          # Kaggle dataset (55,500 rows, 15 columns)
├── analysis.py                     # Standalone chart generation script
├── Healthcare_Analysis_Report.pdf  # Full project report
│
└── charts/                         # Exported visualisations
    ├── 01_billing_by_condition.png
    ├── 02_patients_by_condition.png
    ├── 03_admission_type.png
    ├── 04_test_results_by_condition.png
    ├── 05_los_distribution.png
    ├── 06_billing_by_insurance.png
    ├── 07_age_distribution.png
    ├── 08_los_by_admission.png
    ├── 09_billing_overview.png
    ├── 10_outlier_detection.png
    ├── 11_statistical_tests.png
    ├── 12_regression_results.png
    └── 13_time_trends.png
```

---

<h2><a class="anchor" id="data-cleaning"></a>Data Cleaning & Preparation</h2>

All cleaning and feature engineering done in Python (Pandas) before analysis:

- Parsed date columns to datetime — `Date of Admission` and `Discharge Date`
- Verified zero null values across all 15 columns
- Found **534 duplicate rows** (0.96%) — flagged as synthetic data artefacts; retained since removing them shifts avg billing by $5 and abnormal rate by 0.1 pp (no material impact on any finding)
- Flagged **108 negative billing records** (0.19% of dataset) — excluded from all billing aggregations; retained for count-based analyses
- Created `LOS = Discharge Date − Date of Admission` as the primary length-of-stay metric
- Binned `Age` into 5 groups: `13-29`, `30-44`, `45-59`, `60-74`, `75-89`
- Derived `Season` from admission month — Winter, Spring, Summer, Fall
- Created `Billing Tier` — Low / Medium / High / Very High based on quartile distribution

```python
df['LOS'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

df['Age Group'] = pd.cut(
    df['Age'],
    bins=[12, 29, 44, 59, 74, 89],
    labels=['13-29', '30-44', '45-59', '60-74', '75-89']
)

df['Season'] = df['Admission Month'].apply(
    lambda m: 'Winter' if m in [12,1,2]
         else 'Spring' if m in [3,4,5]
         else 'Summer' if m in [6,7,8]
         else 'Fall'
)

df['Billing Tier'] = pd.qcut(
    df['Billing Amount'], q=4,
    labels=['Low', 'Medium', 'High', 'Very High']
)
```

---

<h2><a class="anchor" id="eda"></a>Exploratory Data Analysis (EDA)</h2>

**Patient & Condition Distribution:**

After cleaning, 55,500 patients distributed across 6 medical conditions. All 6 conditions (Arthritis, Diabetes, Hypertension, Obesity, Cancer, Asthma) sit at ~16.5–16.8% prevalence each — no single condition dominates. The 45–59 age group is the largest cohort at 22.4%. Male and Female patients are exactly balanced at 50.0% each. Abnormal test results affect **18,627 patients — 33.6% of the dataset**.

**Key EDA stats:**
- Total patients: **55,500** | Total billing: **$1.42B** | Avg per patient: **$25,590** | Median: **$25,574**
- Arthritis: **9,308 (16.8%)** | Diabetes: **9,304 (16.8%)** | Hypertension: **9,245 (16.7%)**
- Obesity: **9,231 (16.6%)** | Cancer: **9,227 (16.6%)** | Asthma: **9,185 (16.5%)**
- Abnormal test rate: **33.6%** | Avg LOS: **15.5 days** | Billing outliers: **0**

---

<h2><a class="anchor" id="pipeline"></a>Python Pipeline & Key Analyses</h2>

Six stages process the raw dataset through to statistically validated findings.

```
healthcare_dataset.csv   ← Kaggle source (55,500 rows, 15 columns)
      │
      ▼
Stage 1: Data Cleaning         ← null check, negative billing flag, type casting
      │
      ▼
Stage 2: Feature Engineering   ← LOS, Age Group, Season, Billing Tier
      │
      ▼
Stage 3: EDA — 7 Dimensions    ← demographics → conditions → admissions → billing
      │                              → insurance → time trends → correlation
      ▼
Stage 4: Outlier Detection     ← IQR method + Z-score on Billing Amount and LOS
      │
      ▼
Stage 5: Statistical Testing   ← Independent T-test, One-Way ANOVA, Chi-square
      │
      ▼
Stage 6: Regression Analysis   ← Linear regression, R², MAE, feature coefficients
```

---

### Stage 4 — Outlier Detection (IQR + Z-Score)

```python
from scipy import stats

# IQR fences — Billing Amount
Q1  = df_clean['Billing Amount'].quantile(0.25)   # $13,297
Q3  = df_clean['Billing Amount'].quantile(0.75)   # $37,849
IQR = Q3 - Q1                                     # $24,552
lower_fence = Q1 - 1.5 * IQR   # = -$23,530
upper_fence = Q3 + 1.5 * IQR   # = $74,677

outliers_iqr = df_clean[
    (df_clean['Billing Amount'] < lower_fence) |
    (df_clean['Billing Amount'] > upper_fence)
]  # → 0 records

# Z-Score cross-check
z_scores   = np.abs(stats.zscore(df_clean['Billing Amount']))
outliers_z = df_clean[z_scores > 3]  # → 0 records
```

**Finding:** 0 outliers detected by either method. IQR fences ($-23,530 to $74,677) sit well outside the actual data range ($9–$52,764). Billing is uniformly distributed — no extreme-cost patients exist in this dataset.

---

### Q1 — Abnormal Test Rate by Condition (BQ3)

```python
abnormal_rate = (
    df[df['Test Results'] == 'Abnormal']
    .groupby('Medical Condition').size()
    / df.groupby('Medical Condition').size() * 100
).sort_values(ascending=False).round(1)

print(f"Overall abnormal rate: {(df['Test Results']=='Abnormal').mean()*100:.1f}%")
```

**Finding:** **33.6% overall abnormal rate** — 18,627 patients. Arthritis is highest at 34.3%; Hypertension lowest at 32.6%. The 1.7 percentage point spread is minimal — this is a system-wide quality issue, not condition-specific.

---

### Q2 — Do Clinical Factors Statistically Drive Billing? (BQ9)

```python
from scipy import stats

# T-test: Does gender affect billing?
t_stat, p_ttest = stats.ttest_ind(
    df_clean[df_clean['Gender'] == 'Male']['Billing Amount'],
    df_clean[df_clean['Gender'] == 'Female']['Billing Amount']
)
# t = 1.048, p = 0.295

# ANOVA: Does medical condition affect billing?
condition_groups = [
    df_clean[df_clean['Medical Condition'] == c]['Billing Amount']
    for c in df_clean['Medical Condition'].unique()
]
f_stat, p_anova = stats.f_oneway(*condition_groups)
# F = 2.191, p = 0.052

# Chi-square: Is admission type related to medical condition?
contingency = pd.crosstab(df['Medical Condition'], df['Admission Type'])
chi2_stat, p_chi2, dof, expected = stats.chi2_contingency(contingency)
# chi2 = 19.338, dof = 10, p = 0.036
```

**Finding:** Gender does not significantly affect billing (p=0.295). Condition differences are borderline — ANOVA F=2.191, p=0.052 just fails to reject H0 at α=0.05. The Chi-square result is the standout: p=0.036 confirms that **admission type and medical condition are not independent** in this dataset — knowing a patient's condition carries weak but statistically detectable information about how they will be admitted.

---

### Q3 — Which Features Best Predict Billing? (Regression)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
reg_df['Gender_enc']    = le.fit_transform(reg_df['Gender'])
reg_df['Condition_enc'] = le.fit_transform(reg_df['Medical Condition'])
reg_df['Admission_enc'] = le.fit_transform(reg_df['Admission Type'])

X = reg_df[['Age', 'LOS', 'Gender_enc', 'Condition_enc', 'Admission_enc']]
y = reg_df['Billing Amount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

r2  = r2_score(y_test, model.predict(X_test))           # ≈ 0.0002
mae = mean_absolute_error(y_test, model.predict(X_test)) # $12,361
```

**Finding:** R²≈0. The model explains essentially none of the billing variation. A Mean Absolute Error of $12,361 on a $25,590 average billing is a 48% error rate — no better than always predicting the mean. All feature coefficients are negligible relative to the $52,764 billing range (largest coefficient: Condition at $52). The variables that actually drive billing — procedures performed, lab tests, room type — are not in this dataset.

---

<h2><a class="anchor" id="sample-outputs"></a>Sample Outputs</h2>

**1 — Average Billing by Medical Condition**

<img width="1334" height="733" alt="01_billing_by_condition" src="https://github.com/user-attachments/assets/4b905cd8-e34a-4883-84d3-e11506858cf4" />

Obesity has the highest average billing at $25,860; Cancer is lowest at $25,215 — a $645 spread. ANOVA (p=0.052) confirms this is borderline: the differences are near-significant but fail the 5% threshold with 55,500 records.

Age distribution — mean 51.5 years, largest cohort 45–59 (22.4%).

---

**2 — Patient Volume by Condition**

<img width="1334" height="733" alt="02_patients_by_condition" src="https://github.com/user-attachments/assets/3e1bd16b-1ca7-433e-9326-37e4bfef5814" />

All 6 conditions range from 9,185 (Asthma) to 9,308 (Arthritis).

---

**3 — Admission Type Distribution**

<img width="834" height="731" alt="03_admission_type" src="https://github.com/user-attachments/assets/96a0f5f5-f031-4c5d-95c2-cb25a1faa497" />

<img width="1185" height="733" alt="08_los_by_admission" src="https://github.com/user-attachments/assets/39af3013-cf86-4b1b-858f-41fc827542c4" />

Elective (33.6%), Urgent (33.5%), Emergency (32.9%) are near-equally split. Average LOS is 15.5 days with no meaningful variation across types.

---

**4 — Test Result Distribution by Condition**

<img width="1485" height="808" alt="04_test_results_by_condition" src="https://github.com/user-attachments/assets/76daf1a9-7700-4b1e-b4d8-2833198e031f" />

33.6% overall abnormal rate. Arthritis highest at 34.3%; Hypertension lowest at 32.6%. The flat distribution across all three result types confirms a system-wide rather than condition-specific issue.

---

**5 — Statistical Testing**

<img width="2084" height="763" alt="11_statistical_tests" src="https://github.com/user-attachments/assets/22c4be90-85d2-4db3-a8a3-146807be5627" />

T-test (p=0.295) and ANOVA (p=0.052) both fail to reject H0. Chi-square (p=0.036) rejects H0 — admission type and condition are statistically related, though the effect size is small (chi²=19.34, dof=10).

---

<h2><a class="anchor" id="how-to-run"></a>How to Run This Project</h2>

1. Clone the repository:
```bash
git clone https://github.com/yourusername/healthcare-analysis.git
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib scipy scikit-learn jupyter
```

3. Download the dataset from Kaggle and place in the project root:
```
https://www.kaggle.com/datasets/prasad22/healthcare-dataset
→ Save as: healthcare_dataset.csv
```

4. Open and run the notebook:
```bash
jupyter notebook healthcare_analysis.ipynb
```

5. Run all cells top-to-bottom. The notebook is sequentially dependent:
```
Cell 1:  Imports (pandas, numpy, matplotlib, scipy, sklearn)
Cell 2:  Load dataset + date parsing + LOS computation
Cell 3:  Feature engineering (Age Group, Season, Billing Tier)
Cell 4:  Data quality check + df_clean creation
Cell 5+: EDA → Outlier detection → Statistical testing → Regression
```

---

<h2><a class="anchor" id="recommendations"></a>Final Recommendations</h2>

| Finding | Priority | Recommended Action | Channel |
|---|---|---|---|
| 33.6% abnormal test rate (18,627 patients) | 🔴 Critical | System-wide clinical quality audit — consistent rate across all 6 conditions confirms no single department is responsible | Internal clinical review |
| Avg LOS = 15.5 days | 🔴 Critical | Introduce structured discharge planning from Day 3 — a 2-day reduction saves $1,500–$2,000 per patient across 55,500 records | Clinical protocol |
| Chi-square p=0.036 — condition and admission type related | 🟡 High | Investigate the admission pathway per condition — statistically detectable relationship warrants operational review | Clinical + admin |
| Billing R²≈0 (not predictable) | 🟡 High | Expand the dataset — add procedure codes, lab tests, and room type to produce a usable billing model | Data engineering |
| Summer peak demand (14,343 admissions) | 🟡 High | Build a pre-season capacity plan each May ahead of the June–August surge | Operations + staffing |
| ANOVA p=0.052 — borderline condition effect | 🟡 Medium | Monitor with a larger time window — the borderline result in a 55,500-record dataset is worth tracking as data grows | Analytics team |
| 108 negative billing records | 🟢 Low | Investigate source — likely credit adjustments; implement a billing validation rule to prevent future negatives | Finance team |

**The single most urgent action:** Investigate the 33.6% abnormal test rate. 18,627 patients returned abnormal results. The rate is consistent across all 6 conditions — this is not a condition-specific problem. A system-wide clinical audit identifying whether abnormal results extend stays, increase readmissions, or indicate departmental process failures has a higher ROI than any other intervention available from this data.

**For the next iteration:**
- Add procedure and treatment codes — R²≈0 proves the current feature set explains nothing; procedure-level data is the most likely source of genuine billing signal
- Add patient ID to enable readmission analysis — a patient admitted multiple times with worsening results is the highest-risk profile
- Test ensemble models (Random Forest, Ridge) — non-linear billing relationships may exist that linear regression cannot detect
- Run the Chi-square finding through a Cramér's V calculation — confirm whether the statistically significant condition–admission relationship has any practical effect size
