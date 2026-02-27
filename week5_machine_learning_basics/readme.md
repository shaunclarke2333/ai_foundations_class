# The Diagnostic Prediction Engine

**Course:** CSC6313 AI Foundations  
**Week:** 05 Supervised Machine Learning Pipeline  
**Author:** Shaun Clarke

---

## Project Overview

This project implements a full **supervised machine learning pipeline** using Scikit-Learn to predict patient health outcomes from synthetic patient data. The pipeline covers every stage of an ML workflow: synthetic data generation, exploratory analysis, preprocessing, model training, bias-variance comparison, and interactive real-time inference from the terminal.

The project demonstrates three regression models an underfit model, an overfit model, and an optimal model — to illustrate the **bias-variance tradeoff**, then uses the optimal model alongside a Logistic Regression classifier to power a terminal-based diagnostic tool.

---

## How the ML Pipeline Works

```
[Data Generation]
      ↓
Generate 500 synthetic patient records (age, bmi, blood_sugar_level, health_risk_score)
Intentionally punch holes in 10% of the data (simulate real-world missing values)
Save to patient_data.csv
      ↓
[Exploration]
      ↓
Load CSV → inspect shape, dtypes, missing value counts, statistical summary
      ↓
[Preprocessing]
      ↓
Split: 80% training / 20% test (stratified by random_state=42)
Fit SimpleImputer (median) on training data → transform train + test
Fit StandardScaler (Z-score) on imputed training data → transform train + test
      ↓
[Model Training — Bias-Variance Tradeoff]
      ↓
Model 1 (Underfit): LinearRegression on age only (1 feature)
Model 2 (Overfit):  LinearRegression on PolynomialFeatures(degree=10) → 285 features
Model 3 (Optimal):  LinearRegression on all 3 scaled features
      ↓
[Evaluation]
      ↓
Print MAE and R² for train and test sets for all 3 models
Plot side-by-side bar charts (MSE + R²) saved as bias_variance_tradeoff.png
      ↓
[Inference Engine]
      ↓
Train LogisticRegression on binary risk labels (health_risk_score ≥ 60 → AT RISK)
Prompt user for age, bmi, blood_sugar_level
Apply same imputer + scaler to user input (transform only never refit)
Predict: health risk score (Model 3) + probability of risk (LogisticRegression)
Output diagnosis: HEALTHY or AT RISK
```

---

## The Health Risk Score Formula

The synthetic target variable is generated using a weighted linear formula with added noise:

```
health_risk_score = (age × 0.5)
                  + ((bmi − 25) × 2)
                  + ((blood_sugar_level − 90) × 0.3)
                  + noise ~ N(0, 10)
```

**Feature weight rationale:**

| Feature | Weight | Baseline | Effect |
|---|---|---|---|
| `age` | 0.5 | — | Every year adds 0.5 risk points; 60-year-old contributes 30 points |
| `bmi` | 2.0 | 25 (upper boundary of healthy BMI) | BMI above 25 increases risk; BMI of 30 adds 10 points |
| `blood_sugar_level` | 0.3 | 90 mg/dL | Mild impact; blood sugar of 110 adds 6 points |
| Noise | — | N(0, 10) | Adds realistic variation so patients with similar vitals differ |

The score is continuous, not clamped to 0–100 — the dataset's actual range is approximately **−30.6 to +61.5**.

---

## Data Generation

The `get_mean_std()` helper calculates the mean and standard deviation for each feature from its range:

```python
mean  = (min + max) / 2
std   = (max − mean) / 2
```

Features are drawn from normal distributions using these parameters:

| Feature | Range | Mean | Std Dev |
|---|---|---|---|
| `age` | 20–80 | 50 | 15 |
| `bmi` | 18–30 | 24 | 3 |
| `blood_sugar_level` | 70–100 | 85 | 7.5 |

> **Note:** Normal distributions do not enforce hard boundaries the generated data includes values outside the intended ranges (e.g., age of 1.4 or 107.8). This is intentional: it reflects the realistic behavior of normally distributed data and is an acknowledged tradeoff in the design.

**Intentional missing values:** After generating features, 10% of indices (50 rows) are randomly selected and replaced with `NaN` in `age`, `bmi`, and `blood_sugar_level`. The `health_risk_score` column has no missing values (it was calculated before the holes were punched).

The generated dataset is saved to `patient_data.csv` in the working directory.

---

## Preprocessing

### Train/Test Split
- **80% training** (400 samples), **20% test** (100 samples)
- `random_state=42` ensures reproducible splits

### Median Imputation (`SimpleImputer`)
Machine learning models cannot process `NaN` values. Median imputation replaces each missing value with the **median of that column in the training set**. The median is preferred over the mean because it is robust to skewed distributions and outliers.

- `fit_transform()` is called on training data this calculates the medians and fills the gaps in one step
- `transform()` is called on test data the same training medians are applied without recalculating

**Results after imputation:**

| Set | Missing Before | Missing After |
|---|---|---|
| Training | 117 | 0 |
| Test | 33 | 0 |

### Z-Score Standardization (`StandardScaler`)
Features have different scales (age: ~20–80, blood sugar: ~70–100, BMI: ~18–30). Without standardization, the model might weight a feature more heavily simply because its values are numerically larger.

Standardization transforms every feature to have **mean = 0** and **std = 1** using:

```
z = (x − mean) / std
```

- `fit_transform()` is called on the imputed training data
- `transform()` is called on the imputed test data (same learned mean/std applied)

After standardization, `df.describe()` shows mean ≈ 0 (floating point rounding near zero) and std ≈ 1.001 (pandas uses sample std, not population std intentional and expected).

### Zero Warning Policy
The **UserWarning "X does not have valid feature names"** is triggered when a model trained on a DataFrame is passed a NumPy array at inference time. This is prevented by:
1. Converting all inputs to NumPy arrays before preprocessing (imputer and scaler work on arrays)
2. Passing `np.array([[age, bmi, blood_sugar]])` directly to the inference pipeline

All warnings are globally suppressed with `warnings.filterwarnings("ignore")` as a belt-and-suspenders measure.

---

## The Bias-Variance Challenge

Three models are trained to demonstrate the bias-variance tradeoff the fundamental tension between underfitting (too simple) and overfitting (too complex).

### Model 1 — Underfit (High Bias)
**Configuration:** LinearRegression on `age` only (1 feature, extracted as `X_train_scaled[:, 0:1]`)

```
Training MAE: 8.99  |  Test MAE: 9.94
Training R²:  0.28  |  Test R²:  0.21
```

**Interpretation:** The model only has access to one feature and cannot capture the full pattern in the data. High training and test error, with low R² (the model explains only ~21–28% of variance). Train and test scores are similar the model is consistently bad, not specifically bad on new data. This is the signature of **high bias (underfitting)**: the model is too simple to learn the true relationship.

---

### Model 2 — Overfit (High Variance)
**Configuration:** LinearRegression on `PolynomialFeatures(degree=10)` — 285 features from the original 3

```
Training MAE:  3.07    |  Test MAE:   208,436.87
Training R²:   0.83    |  Test R²:   −16,780,845,294
```

**Interpretation:** With 285 polynomial features, the model memorizes the training data almost perfectly (low training error, high R²). But it completely fails on new data the test MAE is catastrophic and the test R² is a massive negative number (worse than just predicting the mean). This is the signature of **high variance (overfitting)**: the model learned the noise in the training data, not the underlying pattern.

---

### Model 3 — Optimal
**Configuration:** LinearRegression on all 3 scaled features (`age`, `bmi`, `blood_sugar_level`)

```
Training MAE: 7.83  |  Test MAE: 8.41
Training R²:  0.46  |  Test R²:  0.46
```

**Interpretation:** The training and test scores are nearly identical, indicating the model generalizes well to new data. R² of 0.46 means the model explains approximately 46% of the variance in health risk score reasonable given the intentional noise added during data generation. This is the **Goldilocks model**: not too simple, not too complex.

---

### Visual Comparison
A bar chart comparing MAE and R² across all three models for both training and test sets is saved as `bias_variance_tradeoff.png`.

---

## Interactive Inference Engine

After training, a `LogisticRegression` model is trained on **binary risk labels** derived from the health risk score:

```python
y_binary = (health_risk_score >= 60).astype(int)
# 1 = AT RISK, 0 = HEALTHY
```

The inference loop prompts the user for patient vitals and outputs a full diagnostic report.

**Example session:**

```
============================================================
DIAGNOSTIC PREDICTION ENGINE - PATIENT ASSESSMENT
============================================================
Enter patient vitals to receive a health risk assessment.
Type 'quit' at any prompt to exit.

------------------------------------------------------------
Enter patient age (years): 20
Enter patient bmi: 25
Enter patient blood sugar level: 80

============================================================
DIAGNOSTIC RESULTS
============================================================
Predicted Health Risk Score: 6.1 / 100
Probability of Risk:          0.0%
Final Diagnosis:              Healthy
```

**Inference pipeline for user input:**
1. User inputs are read as strings and converted to `float`
2. Packed into `np.array([[age, bmi, blood_sugar]])` (2D array models expect rows × columns)
3. `imputer.transform()` apply training medians (does not refit)
4. `scaler.transform()` apply training mean/std (does not refit)
5. `model_optimal.predict()` → continuous health risk score
6. `logistic_model.predict_proba()` → `[[prob_class_0, prob_class_1]]` → index `[0][1]` × 100 = risk %
7. Threshold: risk score ≥ 60 → `AT RISK`, otherwise `HEALTHY`

> **Why transform-only at inference?** Refitting the imputer or scaler on a single patient's data would recalculate statistics from that one row and apply different transformations than what the model was trained on. The model was trained on data scaled using the training set's statistics — inference data must be scaled the same way.

---

## Prerequisites

- Python 3.10+
- Jupyter Notebook or Google Colab

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

### Option A Google Colab
1. Upload the `.ipynb` file to Google Colab
2. Run all cells in order (`Runtime → Run all`)
3. The inference engine will prompt for input in the last cell's output area

### Option B Jupyter Notebook
```bash
jupyter notebook 1772156764168_shaun_clarke_csc6313.ipynb
```
Run all cells in order. The inference loop runs in the final cell.

### Option C Convert to Python script
```bash
jupyter nbconvert --to script 1772156764168_shaun_clarke_csc6313.ipynb
python 1772156764168_shaun_clarke_csc6313.py
```

---

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Synthetic data generation, array operations, random seeding |
| `pandas` | DataFrame creation, CSV I/O, exploratory analysis |
| `matplotlib` | Bias-variance comparison bar charts |
| `seaborn` | Plot styling (`whitegrid`) |
| `sklearn.model_selection` | `train_test_split` 80/20 stratified split |
| `sklearn.preprocessing` | `StandardScaler` (Z-score), `PolynomialFeatures` (degree 10) |
| `sklearn.impute` | `SimpleImputer` (median strategy) |
| `sklearn.linear_model` | `LinearRegression`, `LogisticRegression` |
| `sklearn.metrics` | `r2_score`, `mean_absolute_error` |
| `warnings` | Suppress `UserWarning` for clean terminal output |

---

## File Structure

```
week05/
├── 1772156764168_shaun_clarke_csc6313.ipynb   # Full ML pipeline notebook
├── patient_data.csv                            # Auto-generated on first run
├── bias_variance_tradeoff.png                  # Auto-generated model comparison chart
└── README.md                                   # This file
```

---

## Design Notes

**Why use `get_mean_std()` instead of hardcoding?** Centralizing the mean/std calculation in a helper function makes the data generation logic self-documenting you can read the feature ranges directly and know exactly how the normal distribution parameters were derived.

**Why punch holes before splitting the data?** The missing values are introduced at the raw data stage to simulate a realistic dataset. The imputer is then fit only on the training split if we had already removed missing values before splitting, we'd have fewer samples and potentially lost the opportunity to demonstrate imputation properly.

**Why `max_iter=1000` for LogisticRegression?** Logistic regression uses an iterative optimization algorithm (similar to gradient descent). The default of 100 iterations is sometimes insufficient for the algorithm to converge, which triggers a `ConvergenceWarning`. Setting `max_iter=1000` gives the solver enough iterations to reach a stable solution.

**Why `include_bias=False` in PolynomialFeatures?** `LinearRegression` already adds its own intercept term. If `PolynomialFeatures` also adds a bias (constant 1) column, the model would have a redundant intercept, which is unnecessary and can cause numerical issues.

**Why `predict_proba()[0][1]`?** `predict_proba()` returns an array of arrays: `[[P(class_0), P(class_1)]]`. Index `[0]` gets the first (and only) patient's probabilities. Index `[1]` gets the probability of being in class 1 (AT RISK). Multiplying by 100 converts to a percentage.