# 🤖 Career Switch Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Course](https://img.shields.io/badge/Course-CSE422-red)

A machine learning pipeline that predicts whether a data professional is likely to **switch careers**, using demographic, education and employment attributes. It compares three supervised models (Logistic Regression, KNN, Neural Network), adds an unsupervised K-Means analysis, and checks stability with Stratified 5-Fold Cross-Validation.

> 📓 **[Open the notebook](CSE422_Project.ipynb)** · 📄 **[Read the full lab report](CSE422%20Lab%20report.pdf)**

---

## 📌 Problem

Knowing which candidates or employees are likely to leave helps HR teams plan hiring and retention. This is a **binary classification** task:

| Target: `will_change_career` | Meaning |
| --- | --- |
| `0` | Will **not** change career |
| `1` | Will change career |

---

## 📊 Dataset

| Item | Value |
| --- | --- |
| File | `Career_Switch_Prediction_Dataset.csv` |
| Rows | 5,000 |
| Input features used | 11 (after dropping `enrollee_id` and `city`) |
| Numerical features | `city_development_index`, `training_hours` |
| Categorical features | `gender`, `relevent_experience`, `enrolled_university`, `education_level`, `major_discipline`, `experience`, `company_size`, `company_type`, `last_new_job` |
| Class balance | 3,738 stay (74.8%) vs 1,262 switch (25.2%) |

**Missing values:** `company_type` (1,621), `company_size` (1,571), `gender` (1,113) and `major_discipline` (724) have the most gaps.

<p align="center">
  <img src="assets/class_distribution.png" width="420" alt="Class distribution">
</p>

---

## 🔧 Pipeline

1. **Cleaning:** dropped ID-like columns (`enrollee_id`, `city`) to reduce overfitting risk.
2. **EDA:** histograms, boxplots, categorical bar charts, feature-vs-target plots, correlation heatmap.
3. **Missing values:** mode imputation (categorical), median imputation (numerical, because `training_hours` is right-skewed).
4. **Encoding:** Label Encoding for `education_level`, `experience`, `company_size`, `last_new_job`; One-Hot Encoding (`drop_first=True`) for the rest.
5. **Scaling:** `StandardScaler`.
6. **Split:** 80/20 stratified train-test split (`random_state=42`).
7. **Imbalance handling:** `class_weight='balanced'` for Logistic Regression, plus Precision / Recall / F1 / AUC instead of accuracy alone.

<p align="center">
  <img src="assets/correlation_heatmap.png" width="600" alt="Correlation heatmap">
</p>

---

## 🧠 Models

| Model | Role | Notes |
| --- | --- | --- |
| Logistic Regression | Baseline | `class_weight='balanced'`, `max_iter=1000` |
| K-Nearest Neighbors | Distance-based | K chosen with an elbow plot over K = 1 to 20 |
| Neural Network (MLP) | Non-linear model | Hidden layers `(64, 32)` |
| K-Means | Unsupervised exploration | Elbow method on WCSS, K = 1 to 10 |

<p align="center">
  <img src="assets/knn_elbow.png" width="380" alt="KNN elbow">
  <img src="assets/kmeans_elbow.png" width="380" alt="K-Means elbow">
</p>

---

## 📈 Results (20% hold-out test set)

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.707 | 0.443 | **0.627** | **0.519** | **0.716** |
| KNN | **0.761** | **0.551** | 0.278 | 0.369 | 0.694 |
| Neural Network | 0.728 | 0.453 | 0.381 | 0.414 | 0.687 |

**How to read this**

- **KNN** has the best accuracy and precision, but it misses most people who actually switch (recall 0.28). Because ~75% of candidates stay, accuracy alone flatters it.
- **Logistic Regression** has the best recall, F1 and AUC. If the goal is to *catch* likely switchers, it is the strongest of the three here.
- **Neural Network** sits in the middle on every metric.
- All three have modest AUC (about 0.69 to 0.72). The features carry useful but limited signal.

<p align="center">
  <img src="assets/roc_curves.png" width="480" alt="ROC curves">
</p>

<details>
<summary>Confusion matrices</summary>

<p align="center">
  <img src="assets/cm_logreg.png" width="280" alt="Logistic Regression">
  <img src="assets/cm_knn.png" width="280" alt="KNN">
  <img src="assets/cm_nn.png" width="280" alt="Neural Network">
</p>

</details>

### ✅ Stratified 5-Fold Cross-Validation

Accuracy stays within a narrow band across folds, so results do not depend on one lucky split:

| Model | Approx. accuracy range across folds |
| --- | --- |
| Logistic Regression | 0.69 to 0.72 |
| KNN | 0.75 to 0.77 |
| Neural Network | 0.75 to 0.78 |

<p align="center">
  <img src="assets/cv_accuracy.png" width="480" alt="Cross-validation accuracy">
</p>

---

## ⚠️️ Limitations and next steps

- Imputation, encoding and scaling were fit on the full dataset before splitting, which can leak information. A scikit-learn `Pipeline` fixed inside each CV fold would be cleaner.
- K for KNN was chosen using test-set error. Choosing it with cross-validation would be more reliable.
- The Neural Network did not use class weighting, which likely hurts its recall.
- Accuracy-based model selection favors the majority class; tuning the decision threshold for recall is a natural improvement.
- Not yet tried: Random Forest, XGBoost, SMOTE, hyperparameter search, feature importance / SHAP.

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter / Google Colab`

---

## 🚀 Run it yourself

```bash
git clone [https://github.com/sajedulislam5840/CSE422.git](https://github.com/sajedulislam5840/CSE422.git)
cd CSE422
pip install -r requirements.txt
jupyter notebook CSE422_Project.ipynb
