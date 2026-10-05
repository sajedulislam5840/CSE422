# 🤖 Career Switch Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Academic-BRAC_University-red.svg)](https://www.bracu.ac.bd/)

An end-to-end Machine Learning classification system designed to predict whether a data professional intends to switch careers based on demographic, education, and corporate attributes from ~19,000 candidate records.

---

## 📌 Project Overview
Identifying candidate job transition intent is critical for HR departments to optimize talent retention and reduce recruitment overhead. This project benchmarks multiple supervised machine learning architectures, tackles severe class imbalance, and validates model generalizability through Stratified K-Fold Cross-Validation.

---

## 📊 Dataset & Pipeline Highlights

* **Dataset Size:** ~19,000 candidate records across 13 input features and 1 binary target (`will_change_career`).
* **Exploratory Data Analysis (EDA):**
  * Discovered strong negative correlation between `city_development_index` and career switch probability.
  * Identified skewed distribution in `training_hours` and preserved valid edge cases.
* **Data Preprocessing & Feature Engineering:**
  * **Missing Values:** Handled via median imputation (numerical) and mode imputation (categorical).
  * **Categorical Encoding:** Applied **Label Encoding** for ordinal features and **One-Hot Encoding** for nominal features.
  * **Feature Scaling:** Normalized continuous variables using `StandardScaler`.
* **Imbalanced Data Strategy:**
  * Implemented `class_weight='balanced'` in classification models.
  * Evaluated models primarily on **Precision, Recall, F1-Score, and ROC-AUC** to penalize False Negatives.

---

## 🧠 Models Benchmarked

| Model | Purpose / Rationale | Key Techniques |
| :--- | :--- | :--- |
| **Logistic Regression** | Baseline linear classifier | Class-weight balancing, threshold tuning |
| **K-Nearest Neighbors (KNN)** | Non-linear distance-based classification | Optimal $K$ selection via **Elbow Method** |
| **Neural Network (MLP)** | Non-linear feature interaction modeling | Multilayer Perceptron architecture |
| **K-Means Clustering** | Unsupervised exploratory segmentation | Within-Cluster Sum of Squares (WCSS) validation |

---

## 📈 Validation & Results
* **Stratified 5-Fold Cross-Validation:** Validated model stability across multiple data splits to eliminate overfitting risk and ensure high generalizability on unseen data.
* **Findings:** The Multilayer Perceptron (Neural Network) and KNN captured complex overlapping demographic patterns most effectively.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `Scikit-Learn`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
* **Environment:** Jupyter Notebook / Google Colab

---

## 👥 Contributors
* **Mir Mohammad Sajedul Islam** ([@sajedulislam5840](https://github.com/sajedulislam5840))
