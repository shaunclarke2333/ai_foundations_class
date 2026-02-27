# The Classifier Showdown (Non-Linear Methods)

**Course:** CSC6313 AI Foundations  
**Week:** 06 Non-Linear Classification  
**Author:** Shaun Clarke

---

## Project Overview

This project moves beyond linear models to implement and compare three **non-linear classification algorithms**: Decision Trees, Random Forests, and k-Nearest Neighbors (k-NN). The program generates synthetic patient data, trains all three classifiers, visualizes the Decision Tree's internal logic in a Matplotlib popup, and runs an interactive diagnostic tool where all three models cast independent "votes" on whether a new patient is healthy or at risk — with majority vote determining the final diagnosis.

The design is object-oriented, split across two classes (`PatientDataGenerator` and `ClassifierShowdown`) and a `main()` orchestrator function.

---

## Expected Terminal Workflow

```
Dataset saved to: classifier_patient_data.csv

==================================================
Model Accuracy Comparison
==================================================
Decision Tree   0.9300 (93.00%)
Random Forest   0.9500 (95.00%)
KNN             0.9200 (92.00%)
==================================================

[Matplotlib popup opens user reviews decision tree visualization, then closes it]

============================================================
DIAGNOSTIC PREDICTION ENGINE PATIENT ASSESSMENT
The Classifier Showdown (Non-Linear Methods)
============================================================
Enter patient vitals to receive a health risk assessment.
Type 'quit' at any prompt to exit.

------------------------------------------------------------
Enter patient age (years): 65
Enter patient bmi: 32
Enter patient blood sugar level: 180

==================================================
Model Voting Results
==================================================
Decision Tree   At Risk
Random Forest   At Risk
KNN             Healthy
==================================================
Votes > At Risk: 2, Healthy: 1
FINAL DIAGNOSIS: AT RISK

Assess another patient? (yes/no): no
```

---

## How the Pipeline Works

```
[PatientDataGenerator]
      ↓
Generate 500 synthetic patient records (age, bmi, blood_sugar)
Compute health_risk_score with weighted formula + noise
Convert score > 60 to binary diagnosis (1 = AT RISK, 0 = HEALTHY)
Save to classifier_patient_data.csv
      ↓
[ClassifierShowdown.preprocess_data()]
      ↓
X = [age, bmi, blood_sugar]   Y = diagnosis
80/20 train-test split (random_state=42)
Fit StandardScaler on X_train → transform X_train and X_test
      ↓
[ClassifierShowdown.train_models()]
      ↓
Model 1: DecisionTreeClassifier(max_depth=3)
Model 2: RandomForestClassifier()  ← 100 trees, default settings
Model 3: KNeighborsClassifier(n_neighbors=5)
      ↓
[ClassifierShowdown.evaluate_models()]
      ↓
Generate predictions on X_test_scaled for all 3 models
Calculate accuracy_score against Y_test
Print formatted accuracy table
      ↓
[ClassifierShowdown.visualize_decision_tree()]
      ↓
Matplotlib popup: Feature Importance (horizontal bar) + Tree Structure (plot_tree)
plt.show() blocks execution — script waits until user closes the window
      ↓
[ClassifierShowdown.run_inference()]
      ↓
Prompt user for age, bmi, blood_sugar
Apply scaler.transform() — never refit
All 3 models predict independently → tally votes → final diagnosis
```

---

## Data Generation (`PatientDataGenerator`)

### Feature Ranges

The `get_mean_std()` helper computes mean and std from a min/max range:
```
mean    = (min + max) / 2
std_dev = (max − mean) / 2
```

Features are drawn from normal distributions:

| Feature | Min | Max | Mean | Std Dev |
|---|---|---|---|---|
| `age` | 18 | 90 | 54 | 18 |
| `bmi` | 18.5 | 45 | 31.75 | 6.625 |
| `blood_sugar` | 70 | 200 | 135 | 32.5 |

> Normal distributions don't enforce hard boundaries. The generated data may include values outside the intended ranges this is expected behavior and mirrors Week 5's approach.

### Diagnosis Formula

The binary `diagnosis` target is derived from a weighted health risk score:

```python
health_risk_score = (age × 0.5)
                  + ((bmi − 25) × 2)
                  + ((blood_sugar − 90) × 0.3)
                  + noise ~ N(0, 10)

diagnosis = 1  if health_risk_score > 60   →  AT RISK
diagnosis = 0  otherwise                   →  HEALTHY
```

The expanded ranges (blood sugar 70–200 vs. 70–100 in Week 5; BMI 18.5–45 vs. 18–30) create more variation in the dataset, which helps produce a more balanced class distribution between AT RISK and HEALTHY patients giving the classifiers more to learn from.

The dataset is saved to `classifier_patient_data.csv`. No missing values are introduced the focus this week is on classifier comparison rather than imputation.

---

## Preprocessing (`preprocess_data`)

### Feature Matrix and Target
```python
X = data[["age", "bmi", "blood_sugar"]]   # input features
Y = data["diagnosis"]                       # binary classification target
```

### Train/Test Split
- 80% training (400 samples), 20% test (100 samples)
- `random_state=42` ensures reproducible splits

### StandardScaler
k-NN is a distance-based algorithm. Without scaling, features with larger numerical values (blood sugar ranging 70–200) would dominate distance calculations over features with smaller values (BMI ranging 18–45). StandardScaler transforms all features to **mean = 0, std = 1**.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # calculates stats, then scales
X_test_scaled  = scaler.transform(X_test)        # applies same stats — never refits
```

Decision Trees and Random Forests don't need scaling they make decisions based on threshold comparisons, not distances. But the same preprocessed data feeds all three models for consistency.

---

## The Three Classifiers (`train_models`)

### Model 1 — Decision Tree (`max_depth=3`)

A Decision Tree learns a sequence of if-then rules by finding the best feature split thresholds at each node, measured by **Gini impurity** the lower the impurity after a split, the better.

**What `.fit()` does:**
1. For every feature and every possible threshold, calculates how much Gini impurity would decrease if the data were split there
2. Picks the split that gives the greatest impurity reduction stores it as a decision node
3. Recurses down the tree, repeating at each branch, until `max_depth=3` is reached
4. Final stored structure: a tree of split rules, e.g. _"if blood\_sugar > 1.2 AND age > 0.8 → AT RISK"_

`max_depth=3` limits the tree to 3 levels (8 possible leaf nodes maximum), preventing overfitting and keeping the visualization readable.

**Strengths:** Fully interpretable you can read every rule the model learned. Fastest prediction time.  
**Weakness:** A single tree may underfit or be unstable with small data changes.

---

### Model 2 Random Forest (100 trees, default settings)

A Random Forest builds an ensemble of 100 Decision Trees, each trained on a slightly different version of the data, then takes a **majority vote** across all trees.

**What `.fit()` does:**
1. Creates 100 bootstrapped datasets by randomly sampling rows from `X_train` with replacement each tree sees a slightly different subset
2. Trains one Decision Tree per bootstrapped dataset. At each split, only a random subset of features is considered (not all 3), so trees diverge from each other
3. Stores all 100 trained trees in memory, each with its own decision rules

**What `.predict()` does:**  
Each of the 100 trees independently predicts 0 or 1. The forest tallies votes if 70 trees say "Healthy" and 30 say "At Risk", the final prediction is "Healthy".

**Strengths:** Most accurate of the three. Averaging over many diverse trees reduces variance and handles non-linear relationships well. Robust to noise.  
**Weakness:** Less interpretable than a single tree you can't easily read 100 trees at once.

---

### Model 3 k-Nearest Neighbors (`k=5`)

k-NN is the **"lazy learner"** it does no learning during `.fit()` at all. All computation happens at prediction time.

**What `.fit()` does:**
- Stores `X_train_scaled` and `Y_train` in memory
- No pattern learning, no weight optimization, no rule building

**What `.predict()` does for a new patient:**
1. Calculates Euclidean distance from the new patient to every patient in the training set
2. Finds the 5 closest neighbors (`n_neighbors=5`)
3. Takes a majority vote on their `diagnosis` labels: if `[1, 1, 1, 0, 0]`, the prediction is `1` (AT RISK)

**Why scaling is critical for k-NN:** Without StandardScaler, a blood sugar difference of 50 mg/dL would numerically outweigh a BMI difference of 5 points just because blood sugar values are larger numbers. Scaling ensures all three features contribute equally to distance.

**Strengths:** Adapts to local patterns well-suited for unusual or rare feature combinations not well-represented by global rules.  
**Weakness:** Slow at prediction time on large datasets (must compute distances to every training point). Produces no interpretable rules.

---

## Model Comparison

| Aspect | Decision Tree | Random Forest | k-NN |
|---|---|---|---|
| Training | Builds if-then rules | Builds 100 trees | Memorizes training data |
| Prediction speed | Fastest | Moderate | Slowest (large datasets) |
| Typical accuracy | Good | Best | Good |
| Interpretability | High visualizable | Low | None |
| Sensitive to feature scale | No | No | Yes requires scaling |
| Best use case | Explainable decisions | Maximum accuracy | Rare / edge-case patterns |

---

## Visualization (`visualize_decision_tree`)

Before the inference loop starts, a Matplotlib popup displays two side-by-side plots:

**Graph A — Feature Importance (horizontal bar chart)**  
Built from `tree_model.feature_importances_` the Gini importance scores assigned to each feature during training. A feature that appears higher in the tree and splits more samples cleanly receives a higher importance score.

```python
feature_names = dataframe[["age", "bmi", "blood_sugar"]].copy().columns.to_list()
importance_values = self.tree_model.feature_importances_
ax1.barh(feature_names, importance_values)
```

**Graph B — Tree Structure**  
`plot_tree()` renders the full if-then decision logic. Each node shows:
- The split condition (e.g., `blood_sugar <= 1.23`)
- Gini impurity value
- Sample count at that node
- Predicted class (color-filled: blue = Healthy, orange = At Risk)

```python
plot_tree(self.tree_model, feature_names=feature_names,
          class_names=["Healthy", "At Risk"], filled=True, ax=ax2)
```

`plt.show()` **blocks execution** — the inference loop does not start until the user closes this window.

---

## Interactive Inference Engine (`run_inference`)

Each patient assessment runs through all three models independently and combines their predictions into a **majority vote**.

**Inference pipeline for user input:**
1. User inputs read as strings → converted to `float` (invalid input caught by `try/except ValueError`)
2. Packed into `np.array([[age, bmi, blood_sugar]])` 2D array, models expect rows × columns
3. `scaler.transform()` applies training-set statistics (does not refit critical for consistency)
4. All three models predict independently
5. Voting table printed via `_print_voting_results_table()`
6. Majority vote determines final diagnosis

```python
patient_data   = np.array([[age, bmi, blood_sugar]])
patient_scaled = self.scaler.transform(patient_data)

self.patient_tree_pred   = self.tree_model.predict(patient_scaled)
self.patient_forest_pred = self.forest_model.predict(patient_scaled)
self.patient_knn_pred    = self.knn_model.predict(patient_scaled)
```

> Each model's `.predict()` always returns an array even for a single patient. `pred[0]` extracts the scalar prediction value.

**Voting logic:**
- `risk_count > healthy_count` → `FINAL DIAGNOSIS: AT RISK`
- Otherwise → `FINAL DIAGNOSIS: HEALTHY`

Type `quit` at any input prompt to exit cleanly.

---

## Prerequisites

- Python 3.10+

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## How to Run

```bash
python shaun_clarke_csc6313_week06.py
```

**Sequence of events:**
1. CSV generated and saved confirmation printed to terminal
2. All three models train (silent)
3. Accuracy table printed to terminal
4. Matplotlib popup opens review the decision tree visualization, then close the window
5. Inference loop begins enter patient vitals at the prompts
6. Voting table and final diagnosis printed after each patient

---

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Synthetic data generation, array operations, random seed |
| `pandas` | DataFrame creation and CSV I/O |
| `matplotlib.pyplot` | Feature importance chart and decision tree visualization |
| `sklearn.model_selection` | `train_test_split` 80/20 split |
| `sklearn.preprocessing` | `StandardScaler` Z-score normalization |
| `sklearn.tree` | `DecisionTreeClassifier`, `plot_tree` |
| `sklearn.ensemble` | `RandomForestClassifier` |
| `sklearn.neighbors` | `KNeighborsClassifier` |
| `sklearn.metrics` | `accuracy_score` |
| `warnings` | Suppress convergence warnings |

---

## File Structure

```
week06/
├── shaun_clarke_csc6313_week06.py    # Full implementation
├── classifier_patient_data.csv       # Auto-generated on first run
└── README.md                         # This file
```

---

## Design Notes

**Why two classes?** `PatientDataGenerator` owns everything about synthetic data production — features, formula, and CSV output. `ClassifierShowdown` owns everything about model training and inference. Either class can be swapped out independently: replace `PatientDataGenerator` with a real CSV reader without touching a single line of classifier logic.

**Why are `_print_accuracy_table()` and `_print_voting_results_table()` private methods?** Both are formatting helpers that should only be called internally `_print_accuracy_table()` from `evaluate_models()`, and `_print_voting_results_table()` from `run_inference()`. The leading underscore signals they are implementation details, not part of the public interface.

**Why `max_depth=3` for the Decision Tree?** Without a depth limit, a Decision Tree grows until it perfectly classifies every training sample memorizing noise and overfitting badly. `max_depth=3` means at most 8 leaf nodes, enough to capture the main patterns in the data while keeping the visualization readable.

**Why `n_neighbors=5` for k-NN?** 5 is the scikit-learn default and a well-established starting point. It's odd avoiding tie votes on binary classification and balances between underfitting (too large a `k` smooths out real patterns) and overfitting (too small a `k` adapts to individual noise points).

**Why `accuracy_score` instead of `.score()`?** Both return the same number, but using `accuracy_score` explicitly separates prediction from evaluation making it easy to swap in `f1_score`, `precision_score`, or `recall_score` later by changing only one line.

**Why `scaler.transform()` at inference and never `fit_transform()`?** Calling `fit_transform()` on a single new patient would recalculate mean and std from that one row alone, producing completely different scaling than what the model was trained on. The scaler must apply the exact same statistics it learned from the training set to every new input.