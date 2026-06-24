# Predictive Maintenance — Logistic Regression Analysis (AI4I 2020)

A multivariate logistic regression study on the **AI4I 2020 Predictive Maintenance Dataset**, framing machine failure as a binary classification problem and extracting actionable quality-control insight through odds-ratio interpretation.

> Dataset: [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

---

## Problem Definition

The target variable `Machine failure` is **categorical** (0 = normal, 1 = failure), not continuous — making this a classification rather than a regression-of-a-number problem. The goal is to build a multivariate logistic regression model that predicts whether a unit is defective from new process inputs `X`, and to identify *which process parameters actually drive failure*.

A failure (`Machine failure = 1`) is triggered if **any** of five underlying failure modes occurs:

| Mode | Trigger condition |
|------|-------------------|
| **TWF** (Tool Wear Failure) | Tool fails at a tool-wear time between 200–240 min |
| **HDF** (Heat Dissipation Failure) | Air–process temp difference < 8.6 K **and** rotational speed < 1380 rpm |
| **PWF** (Power Failure) | Torque × rotational speed (power) below 3500 W or above 9000 W |
| **OSF** (Overstrain Failure) | Tool wear × torque exceeds 11,000 (L) / 12,000 (M) / 13,000 (H) min·Nm |
| **RNF** (Random Failure) | 0.1% chance regardless of parameters |

Because the binary label hides *which* mode fired, the model has to infer the true physical drivers from the raw process variables alone.

---

## Feature Overview

| Variable | Description |
|----------|-------------|
| Type (L/M/H) | Product quality variant (50% / 30% / 20%) |
| Air temperature [K] | Random walk, ~2 K SD around 300 K |
| Process temperature [K] | Air temperature + 10 K, ~1 K SD |
| Rotational speed [rpm] | Derived from 2860 W power + noise |
| Torque [Nm] | ~Normal around 40 Nm, SD 10, non-negative |
| Tool wear [min] | Accumulated wear, +5/3/2 min for H/M/L |

---

## 1. Exploratory Data Analysis

- **No missing values.**
- **Severe class imbalance** — Normal 96.61% vs. Failure 3.39%. Handled later at the modeling stage rather than by resampling.
- **Failure mode counts:** TWF 46, HDF 115, PWF 95, OSF 98, RNF 19.
- **Failure by quality type:** L 235, M 83, H 21. The naive expectation that almost all failures come from the L grade does **not** hold — failures appear across all three grades.
- **Multicollinearity:** the correlation heatmap shows `Air temperature` and `Process temperature` are strongly correlated (≈ 0.88). Since air temperature is the more *direct* process-side variable, **process temperature is dropped**.
- The weak raw correlation between individual process variables (air temp, tool wear) and `Machine failure` is an artifact of the **class imbalance**, not evidence of irrelevance.

---

## 2. Preprocessing

1. Drop identifiers (`UDI`, `Product ID`) and the five failure-mode columns to avoid **target leakage** — the model must learn from process inputs, not from the labels that define the target.
2. Drop `Process temperature` (multicollinearity, from EDA).
3. **Encode `Type` as an ordinal feature**: L → 1, M → 2, H → 3 (quality ordering preserved).
4. **Standardize** the continuous variables (air temperature, rotational speed, torque, tool wear) so coefficient magnitudes are comparable. `Type`, being ordinal, is excluded from scaling.

---

## 3. Model Construction & Evaluation

- **Logistic regression** with an 80/20 stratified train/test split (imbalance ratio preserved in both splits).
- **Class imbalance is handled via `class_weight='balanced'`** — no resampling, no data loss or interpolation. The minority (failure) class is penalized more heavily in the cross-entropy loss, which also keeps the odds ratios cleanly interpretable.

### Results

Because the data is imbalanced, **precision / recall / F1 matter more than accuracy.**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| 0 (Normal) | 0.99 | 0.81 | 0.89 | 1932 |
| 1 (Failure) | 0.13 | 0.81 | 0.23 | 68 |

**ROC-AUC: 0.8918**

The low precision on class 1 reflects a high false-alarm rate. For a manufacturing line this is an acceptable, deliberately **conservative** posture — catching defects is worth tolerating false alarms. The model recalls **0.81 of all 68 true defects**, which is solid for this purpose.

### Odds Ratio Interpretation

Odds ratios (exp of the standardized coefficients) reveal the *true drivers* of failure. Each is read **per 1 standard deviation increase**:

| Feature | Coefficient | Odds Ratio | Reading |
|---------|-------------|------------|---------|
| Torque [Nm] | 2.304 | **10.01** | +1 SD (9.97 Nm) → failure odds ×10 |
| Rotational speed [rpm] | 1.612 | **5.02** | +1 SD (179.28 rpm) → failure odds ×5 |
| Air temperature [K] | 0.769 | **2.16** | +1 SD (2.00 K) → failure odds ×2.1 |
| Tool wear [min] | 0.765 | **2.15** | +1 SD (63.65 min) → failure odds ×2.1 |
| Type | −0.263 | **0.77** | +1 grade → failure odds ×0.77 (higher grade = safer) |

### Quality-Control Insight

1. **Torque & rotational speed dominate.** Avoid processes that push throughput by ramping these up. Set conservative UCLs with an interlock that trips when exceeded, and revisit any routine that aggressively scales output.
2. **Temperature sensitivity.** Failure odds roughly double per 2 K. Inspect coolant/lubricant delivery and maintain HVAC for stable ambient temperature.
3. **Tool wear** is gradual rather than spiky (unlike temp/torque shocks), so it's the ideal candidate for scheduled **predictive maintenance** rather than reactive interlocks.
4. **Type:** higher quality grades fail less, so **L-grade production warrants stricter monitoring.**

---

## 4. Cut-off Threshold Tuning

The default 0.5 threshold is swept from 0.10 upward in 0.05 steps, tracking accuracy, precision, recall, and F1.

Two decision rules:

1. **F1-maximizing cut-off → 0.80** (best precision/recall balance for imbalanced data).
2. **Recall-driven (QC-conservative) cut-off → 0.35**, the point where recall stays above ~0.90. From a quality-control standpoint — where missing a defect is the costly error — this is the optimal operating point.

The recall/precision trade-off curves cross around the high-threshold region, making the choice of operating point an explicit business decision rather than a fixed default.

---

## 5. Multinomial Logistic Regression (≥ 3 classes)

Extension of the binary case to recover the probability of any one class via the **softmax** formulation. Starting from the log-odds of each class relative to a reference class `K`:

$$\frac{P(Y=j)}{P(Y=K)} = e^{x^{\top}\theta_j}$$

summing over all classes and solving gives the softmax:

$$P(Y=j) = \frac{e^{x^{\top}\theta_j}}{\sum_{i=1}^{K} e^{x^{\top}\theta_i}}$$

which generalizes the binary sigmoid to the multi-class setting.

---

## 6. Why Maximizing Log-Likelihood = Minimizing Cross-Entropy

**MLE** maximizes the likelihood of the observed data. For the binary case with prediction $\hat{y}_i$:

$$P(y_i \mid x_i) = \hat{y}_i^{\,y_i}\,(1-\hat{y}_i)^{1-y_i}$$

The joint likelihood over `N` samples, after a log transform, becomes:

$$\log L = \sum_{i=1}^{N}\big[\, y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\,\big]$$

**Cross-entropy** measures the divergence between the true distribution `P` (the labels) and the predicted distribution `Q`:

$$H = -\sum_{i=1}^{N}\big[\, y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\,\big]$$

The two expressions are identical up to a sign. **Maximizing the log-likelihood is therefore exactly equivalent to minimizing the cross-entropy loss** — the same optimization viewed from a statistical (MLE) versus an information-theoretic (loss-function) standpoint.

---

