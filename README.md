# 💳 Credit Card Fraud Detection using Machine Learning

### 📘 Project Overview

This project focuses on **detecting fraudulent credit card transactions** using multiple machine learning algorithms.
It leverages the popular **Credit Card Fraud Detection Dataset** (European transactions, September 2013) and compares the performance of various models on a highly **imbalanced classification problem**.

---

## 📂 Dataset Summary

* **Total Transactions:** 284,807
* **Features:** 31 (including anonymized PCA features `V1–V28`)
* **Target Variable:** `Class` → 0 = Non-fraud, 1 = Fraud
* **Fraud Cases:** 492 (≈0.17%)
* **Non-fraud Cases:** 284,315

📊 The dataset is **highly imbalanced**, meaning accuracy alone is not a reliable performance metric — hence, metrics like **AUC, Recall, and F1-score** are emphasized.

---

## 🧾 **Section 1 – Load Dataset**

**Code Summary:**

```python
df = pd.read_csv('creditcard.csv')
print(df.shape)
print(df.head())
print(df['Class'].value_counts())
print(df['Amount'].describe())
```

**Output Overview:**

(284807, 31)
   Time        V1        V2        V3        V4        V5        V6        V7  \
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9  ...       V21       V22       V23       V24       V25  \
0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   
1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   
2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   
3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   
4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   

        V26       V27       V28  Amount  Class  
0 -0.189115  0.133558 -0.021053  149.62      0  
1  0.125895 -0.008983  0.014724    2.69      0  
2 -0.139097 -0.055353 -0.059752  378.66      0  
3 -0.221929  0.062723  0.061458  123.50      0  
4  0.502292  0.219422  0.215153   69.99      0  

[5 rows x 31 columns]
Class
0    284315
1       492
Name: count, dtype: int64
count    284807.000000
mean         88.349619
std         250.120109
min           0.000000
25%           5.600000
50%          22.000000
75%          77.165000
max       25691.160000
Name: Amount, dtype: float64

* Shape: `(284807, 31)`
* Legitimate transactions: `284,315`
* Fraudulent transactions: `492`
* Average amount: `€88.35` (max ≈ €25,691)

**Key Insights:**

* Severe **class imbalance (~0.17% fraud)**.
* Transaction amounts are **right-skewed** (most transactions are small).
* PCA-transformed features (`V1–V28`) ensure privacy.

---

## 🔍 **Section 2 – Data Exploration**

### **Fraud vs Non-Fraud Visualization**

![Fraud vs Non-Fraud (Log Scale)](https://github.com/Charithanl/credit-card-fraud-detection/blob/main/images%2Fsection1b.png)

**Interpretation:**

* 99.827% of transactions are **legitimate**, only 0.173% are **fraudulent**.
* Logarithmic scale helps visualize the imbalance effectively.

---

### **Transaction Amount Distribution**

![Transaction Amount Distribution](https://github.com/Charithanl/credit-card-fraud-detection/blob/main/images/histogram-plot.png)

**Insights:**

* Most transactions are **small (< €100)**.
* A few large transactions create a **long tail**.
* This feature requires **scaling** for modeling.

---

## ⚙️ **Section 4 – Data Preprocessing / Manipulation**

**Steps:**

* Standardized `Amount` using `StandardScaler()`.
* Dropped unnecessary columns: `Time`, `Amount`.
* Created:

  * `X` → feature set (29 columns)
  * `y` → target (`Class`)

✅ Result: Dataset cleaned, normalized, and ready for training.

---

## ✂️ **Section 5 – Train-Test Split**

**Code:**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=123, stratify=y)
```

**Output:**

```
X_train: (227,845, 29)
X_test: (56,962, 29)
```

**Notes:**

* 80% training / 20% testing split.
* `stratify=y` maintains the same class ratio.
* Ensures model evaluation consistency.

---

## 🧮 **Section 6 – Logistic Regression Model**

![ROC – Logistic Regression](https://github.com/Charithanl/credit-card-fraud-detection/blob/main/images/Lr-PLOT.png)

**Metrics:**

| Metric                | Value |
| --------------------- | ----- |
| **AUC**               | 0.961 |
| **Precision (Fraud)** | 0.86  |
| **Recall (Fraud)**    | 0.57  |
| **F1-score (Fraud)**  | 0.69  |

**Insights:**

* Excellent **AUC (0.96)** — strong ranking performance.
* Moderate recall (misses some frauds).
* Strong precision (few false positives).
   **Solid baseline model.**

---

##  **Section 7 – Decision Tree Classifier**

![ROC – Decision Tree](https://github.com/Charithanl/credit-card-fraud-detection/blob/main/images/DT-plot.png)

**Metrics:**

| Metric                | Value |
| --------------------- | ----- |
| **AUC**               | 0.857 |
| **Precision (Fraud)** | 0.71  |
| **Recall (Fraud)**    | 0.71  |
| **F1-score (Fraud)**  | 0.71  |

**Insights:**

* Recall improved from 0.57 → 0.71.
* Slightly lower AUC — possible overfitting.
* More balanced fraud detection performance.

---

## 🤖 **Section 8 – Artificial Neural Network (MLP Classifier)**

![ROC – MLP Classifier](https://github.com/Charithanl/credit-card-fraud-detection/blob/main/images/MLP-plot.png)

**Metrics:**

| Metric                | Value |
| --------------------- | ----- |
| **AUC**               | 0.936 |
| **Precision (Fraud)** | 0.86  |
| **Recall (Fraud)**    | 0.72  |
| **F1-score (Fraud)**  | 0.78  |

**Insights:**

* Best recall and F1 performance among all models.
* Slightly lower AUC than Logistic Regression but higher recall.
  ✅ **Best overall fraud detection model.**

---

## 🚀 **Section 9 – Gradient Boosting (HistGradientBoosting)**

![HistGradientBoosting](https://github.com/Charithanl/credit-card-fraud-detection/blob/main/images/hgb-plot.png)

**Metrics:**

| Metric                | Value |
| --------------------- | ----- |
| **AUC**               | 0.832 |
| **Precision (Fraud)** | 0.42  |
| **Recall (Fraud)**    | 0.59  |
| **F1-score (Fraud)**  | 0.49  |

**Insights:**

* Moderate AUC (0.83) — decent start but needs tuning.
* Higher recall than Logistic Regression, lower precision.
* Benefits from **hyperparameter tuning** and **class balancing**.

---

## 📊 **Model Performance Comparison**

| Model                    | AUC   | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | Remarks                        |
| ------------------------ | ----- | ----------------- | -------------- | ---------- | ------------------------------ |
| **Logistic Regression**  | 0.961 | 0.86              | 0.57           | 0.69       | Strong baseline, excellent AUC |
| **Decision Tree**        | 0.857 | 0.71              | 0.71           | 0.71       | Balanced, interpretable        |
| **MLP Classifier**       | 0.936 | 0.86              | 0.72           | 0.78       | Best overall fraud detection   |
| **HistGradientBoosting** | 0.832 | 0.42              | 0.59           | 0.49       | Decent recall, needs tuning    |

---

##  **Key Takeaways**

* **MLP Classifier** achieved the best fraud detection performance.
* **Logistic Regression** remains a strong, interpretable baseline.
* **Ensemble methods** can improve with tuning and resampling.

### Future Improvements

* Apply **SMOTE** for synthetic oversampling.
* Use **cross-validation** for stable results.
* Explore **cost-sensitive learning** to penalize false negatives.

---

 **Tech Stack**

* **Language:** Python 🐍
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
* **Models:** Logistic Regression, Decision Tree, MLPClassifier, HistGradientBoostingClassifier
* **Metrics:** ROC-AUC, Precision, Recall, F1-score, Confusion Matrix

---

**Conclusion**

Fraud detection is a **highly imbalanced classification problem** where accuracy alone can be misleading.
Through experimentation with multiple models, the **MLP Classifier** offered the **best trade-off** between recall and precision, effectively identifying fraudulent activities.

This project emphasizes:

* Data preprocessing and scaling
* Stratified sampling
* Using AUC, recall, and F1 as performance metrics
* Handling class imbalance stratergically

Author
Project by: Charitha NL
Dataset Source: Kaggle – Credit Card Fraud Detection (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
License: MIT