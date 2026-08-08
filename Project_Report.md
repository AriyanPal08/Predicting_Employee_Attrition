# Predicting Employee Attrition Using Supervised Learning

**Student:** Ariyan Pal | **Student ID:** 2025EB1100270
**Course:** [Course Name] | **Date:** [Submission Date]

---

## 1. Introduction

Employee attrition -- the voluntary departure of employees -- imposes significant costs through recruitment, onboarding, and the loss of institutional knowledge. Identifying employees likely to leave enables HR to intervene proactively with targeted retention strategies.

This project applies supervised classification to the IBM HR Analytics Employee Attrition dataset. The objective is to train and evaluate multiple models that predict whether an employee will leave, based on demographic, compensation, satisfaction, and tenure-related attributes. The binary target variable is `Attrition` (Yes = 1, No = 0). A fixed `random_state = 42` is used throughout for reproducibility.

---

## 2. Dataset Overview and Exploratory Analysis

### 2.1 Dataset Dimensions

The dataset contains **1,470 employees** and **35 columns** (1 target + 34 predictor features): 26 numerical and 8 categorical. The dataset has **zero missing values** and **zero duplicated rows**.

### 2.2 Target Class Distribution

| Class | Count | Percentage |
|-------|-------|-----------|
| No (0) | 1,233 | 83.88% |
| Yes (1) | 237 | 16.12% |

The target is **imbalanced**: only 16.12% of employees experienced attrition. A naive classifier predicting "No" for every employee would achieve ~84% accuracy while failing to identify any departing employee. Consequently, accuracy alone is insufficient; Precision, Recall, F1-score, and ROC-AUC are essential.

![Target Class Distribution](charts/target_distribution.png)

### 2.3 Key Exploratory Findings

**Constant-value columns:** `EmployeeCount` (1), `Over18` ('Y'), `StandardHours` (80) carry no predictive information.

**Numerical features by attrition class:**

![Numerical Feature Distributions by Attrition Class](charts/numerical_boxplots.png)

Employees who left tend to have lower MonthlyIncome, fewer TotalWorkingYears, and fewer YearsAtCompany. These differences suggest that compensation and tenure may provide predictive signal for classification.

**Categorical attrition rates:**

![Categorical Attrition Rates](charts/categorical_attrition_rates.png)

OverTime shows a strong association with attrition: employees who work overtime have an observed attrition rate approximately three times higher than those who do not. No single numerical feature has a high individual correlation with attrition, reinforcing the need for multivariate classification models.

---

## 3. Data Preprocessing

All preprocessing transformations are fitted using the training data only and then applied to the test data, preventing information leakage.

| Step | Action |
|------|--------|
| Missing values | None found; no imputation |
| Dropped columns | `EmployeeCount`, `Over18`, `StandardHours` (constant), `EmployeeNumber` (identifier) |
| Target encoding | No = 0, Yes = 1 |
| Train-test split | 80/20 stratified, `random_state=42` |
| Categorical encoding | One-hot encoding (7 features to 28 binary columns) |
| Numerical scaling | Standard scaling (23 features) |
| Final feature count | **51** processed features |
| Leakage prevention | Preprocessor fitted on training data only; `Pipeline` used for CV |

**Partitions:**

| Partition | Shape | Class 1 Count | Class 1 % |
|-----------|-------|---------------|-----------|
| Training | 1,176 x 30 | 190 | 16.16% |
| Test | 294 x 30 | 47 | 15.99% |

For cross-validation, a `sklearn.pipeline.Pipeline` wraps the `ColumnTransformer` and each classifier, ensuring that encoding and scaling are fitted within each CV fold independently.

---

## 4. Model Development

Four classification models were trained. All use `random_state=42`.

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | `solver='lbfgs'`, `max_iter=1000`, `C=1.0` |
| Decision Tree | `criterion='gini'`, `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5` |
| Random Forest | `n_estimators=300`, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=4`, `class_weight='balanced'` |
| Bagging Classifier | `n_estimators=100`, base tree: `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5` |

---

## 5. Cross-Validation Results

Generalisation was assessed using **5-fold stratified cross-validation** (`StratifiedKFold`, `shuffle=True`, `random_state=42`). Preprocessing was performed inside each fold via Pipeline to prevent leakage.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **0.8903 +/- 0.0191** | **0.7763 +/- 0.0744** | **0.4421 +/- 0.0962** | **0.5610 +/- 0.0977** | **0.8392 +/- 0.0289** |
| Random Forest | 0.8648 +/- 0.0158 | 0.6444 +/- 0.1030 | 0.3842 +/- 0.1099 | 0.4710 +/- 0.0929 | 0.8092 +/- 0.0317 |
| Bagging Classifier | 0.8588 +/- 0.0151 | 0.7269 +/- 0.1312 | 0.2158 +/- 0.0805 | 0.3239 +/- 0.1001 | 0.7854 +/- 0.0356 |
| Decision Tree | 0.8393 +/- 0.0092 | 0.5008 +/- 0.0583 | 0.2684 +/- 0.0714 | 0.3465 +/- 0.0679 | 0.6695 +/- 0.0805 |

![CV Performance Comparison](charts/cv_performance.png)

Logistic Regression achieves the highest CV F1-score (0.5610), CV ROC-AUC (0.8392), and CV Recall (0.4421), confirming it as the strongest model across the primary evaluation metrics. Random Forest ranks second with CV F1 of 0.4710 and ROC-AUC of 0.8092.

---

## 6. Test-Set Evaluation

After model selection via CV, all models were evaluated once on the held-out test set (294 employees, 47 attrition cases).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **0.8605** | **0.6154** | 0.3404 | **0.4384** | **0.8115** |
| Random Forest | 0.8367 | 0.4857 | **0.3617** | 0.4146 | 0.7977 |
| Decision Tree | 0.8367 | 0.4737 | 0.1915 | 0.2727 | 0.6824 |
| Bagging Classifier | 0.8367 | 0.4667 | 0.1489 | 0.2258 | 0.7553 |

### Confusion Matrices

![Confusion Matrices](charts/confusion_matrices.png)

| | Logistic Regression | Decision Tree | Random Forest | Bagging |
|---|---|---|---|---|
| **TN** | 237 | 237 | 229 | 239 |
| **FP** | 10 | 10 | 18 | 8 |
| **FN** | 31 | 38 | 30 | 40 |
| **TP** | 16 | 9 | 17 | 7 |

---

## 7. Threshold Analysis

At the default threshold of 0.50, all models show moderate-to-low recall. Using out-of-fold predictions from 5-fold CV (the test set was NOT used), the precision-recall trade-off was analysed for the two strongest models:

**Logistic Regression:**

| Threshold | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| 0.20 | 0.4281 | 0.7211 | 0.5373 |
| 0.25 | 0.4741 | 0.6737 | 0.5565 |
| 0.30 | 0.5374 | 0.6421 | 0.5851 |
| **0.35** | **0.6203** | **0.6105** | **0.6154** |
| 0.40 | 0.6753 | 0.5474 | 0.6047 |
| 0.50 | 0.7850 | 0.4421 | 0.5657 |
| 0.60 | 0.8395 | 0.3579 | 0.5018 |

**Best F1 threshold: 0.35** (F1=0.6154, Precision=0.6203, Recall=0.6105)

![Threshold Analysis](charts/threshold_analysis.png)

Lowering the threshold from 0.50 to 0.35 increases Logistic Regression's validation recall from 0.44 to 0.61, while maintaining precision above 0.62. The optimal threshold depends on the relative business cost of missing a departure (false negative) versus an unnecessary intervention (false positive).

---

## 8. Model Selection and Justification

### Selected Model: Logistic Regression

Based on the multi-criteria analysis, Logistic Regression is selected as the preferred candidate:

| Criterion | Logistic Regression | Random Forest | Decision Tree | Bagging |
|-----------|-------------------|---------------|---------------|---------|
| CV F1 (mean) | **0.5610** | 0.4710 | 0.3465 | 0.3239 |
| CV ROC-AUC (mean) | **0.8392** | 0.8092 | 0.6695 | 0.7854 |
| CV Recall (mean) | **0.4421** | 0.3842 | 0.2684 | 0.2158 |
| CV F1 (SD) | 0.0977 | 0.0929 | **0.0679** | 0.1001 |
| Test F1 | **0.4384** | 0.4146 | 0.2727 | 0.2258 |
| Test ROC-AUC | **0.8115** | 0.7977 | 0.6824 | 0.7553 |
| Interpretability | High | Medium | High | Medium |
| Deployment | Simple | Complex | Simple | Complex |

### Generalisation Analysis

| Model | CV F1 | Test F1 | Diff | CV AUC | Test AUC | Diff |
|-------|-------|---------|------|--------|----------|------|
| Logistic Regression | 0.5610 | 0.4384 | -0.1226 | 0.8392 | 0.8115 | -0.0277 |
| Random Forest | 0.4710 | 0.4146 | -0.0564 | 0.8092 | 0.7977 | -0.0115 |
| Decision Tree | 0.3465 | 0.2727 | -0.0738 | 0.6695 | 0.6824 | +0.0129 |
| Bagging Classifier | 0.3239 | 0.2258 | -0.0981 | 0.7854 | 0.7553 | -0.0301 |

All models show a modest drop between CV and test F1 performance, which is expected given the small test set (47 positive cases). ROC-AUC differences are small, suggesting consistent probability-ranking ability. None of the models exhibit severe overfitting.

### Recall Limitation

All models show moderate-to-low recall at the default 0.5 threshold. The threshold analysis demonstrates that recall can be improved significantly (from 0.44 to 0.61 for Logistic Regression) by lowering the threshold to 0.35, at the cost of reduced precision. The appropriate threshold should be determined by the organisation based on the relative cost of missing a departure versus an unnecessary intervention.

No explicit business cost matrix was provided, so this analysis uses standard classification metrics rather than assigning arbitrary monetary costs.

---

## 9. Limitations and Future Improvements

**Limitations:**
1. The target class is imbalanced (16% positive), constraining minority-class recall.
2. No explicit business cost matrix was available to optimise the classification threshold.
3. The dataset contains 1,470 employees -- a moderate size that limits CV statistical power.
4. Feature engineering was not explored; models use original features only.
5. The selected model is a preferred candidate among evaluated alternatives, not a validated production-ready system.

**Suggested improvements:**
1. **Threshold optimisation with business input:** Define the relative cost of false negatives versus false positives and select the threshold that minimises expected cost.
2. **Feature engineering:** Construct derived features such as income-to-job-level ratios or satisfaction composite scores. Additional HR data could improve prediction if available at inference time.

---

## 10. Conclusion

This project followed a structured supervised-learning workflow to predict employee attrition using the IBM HR Analytics dataset (1,470 employees, 51 processed features). Four classification models were trained and evaluated using both 5-fold stratified cross-validation and a held-out test set.

**Logistic Regression** was selected as the preferred model based on its leading performance across all primary CV metrics: F1-score (0.5610), ROC-AUC (0.8392), Recall (0.4421), and Precision (0.7763), combined with its interpretability and deployment simplicity.

The primary limitation is moderate recall at the default 0.5 threshold. The threshold analysis demonstrates that lowering the threshold to 0.35 substantially improves recall (to 0.6105) with an F1-score of 0.6154. Future work should focus on threshold optimisation with business input and feature engineering.

---

## References

1. IBM HR Analytics Employee Attrition Dataset. Kaggle. Available at: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, pp. 2825-2830.
