# Task 4: Classification with Logistic Regression

**Author:** Purab Roy  
**Email:** roy.purab.28@gmaill.com  

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- Loaded directly from `sklearn.datasets.load_breast_cancer()` (same dataset as the UCI/Kaggle "Breast Cancer Wisconsin" CSV linked in the task).  
- A CSV copy is saved locally as `breast_cancer_data.csv` so the raw data file is present in the repo, as required by the submission rules.

**Label encoding (as in sklearn):**  
- `target = 0` → malignant  
- `target = 1` → benign  

In this project, the **positive class is 1 (benign)**. All precision/recall/F1 metrics are computed with respect to this positive class unless stated otherwise.

---

## Steps Performed

1. Loaded a binary classification dataset (malignant = 0, benign = 1).
2. Train/test split + feature standardization.
3. Fit a Logistic Regression model.
4. Evaluated with confusion matrix, precision, recall, ROC-AUC.
5. Tuned the decision threshold and visualized/explained the sigmoid function.

---

## Tools Used

- Python 3
- scikit-learn
- pandas
- matplotlib

---

## Files in This Repository

- `logistic_regression_classifier.py` – main script that:
  - loads the dataset,
  - performs preprocessing,
  - trains the model,
  - evaluates it,
  - saves all plots and result files.
- `breast_cancer_data.csv` – CSV copy of the dataset.
- `threshold_tuning_results.csv` – metrics at multiple thresholds.
- `README.md` – this file.
- `images/` – folder containing generated plots:
  - `confusion_matrix.png` – confusion matrix at threshold = 0.5.
  - `roc_curve.png` – ROC curve and AUC.
  - `sigmoid_function.png` – sigmoid function visualization.
  - `threshold_tuning.png` – precision & recall vs decision threshold.

---

## How to Run

```bash
pip install scikit-learn pandas matplotlib
python logistic_regression_classifier.py
```

This will regenerate all plots and result files in the current directory.

---

## Key Results (default threshold = 0.5)

At the default threshold (0.5), the model achieves very high performance on the test set:

- Accuracy ≈ 0.98  
- Precision ≈ 0.99  
- Recall ≈ 0.99  
- F1-score ≈ 0.99  
- ROC-AUC ≈ 0.995  

Confusion matrix shows only 2 misclassifications out of 114 test samples.

Exact numbers will depend on the random split, but the script prints them in the console and saves:

- `images/confusion_matrix.png`
- `images/roc_curve.png`
- `images/threshold_tuning.png`
- `threshold_tuning_results.csv`

---

## Threshold Tuning

The script evaluates multiple decision thresholds: `[0.3, 0.4, 0.5, 0.6, 0.7]`.

Because the positive class is **benign (1)**:

- Lowering the threshold makes the model more likely to predict benign.
- This increases recall for benign cases (more true benign detected).
- It can reduce precision (more malignant incorrectly labeled as benign).

In a real medical setting, we usually care more about not missing malignant cases. To optimize for that, we could:

- Treat malignant as the positive class (by recoding the target), or  
- Explicitly inspect metrics for class 0 (malignant) from the classification report and choose a threshold accordingly.

The saved file `threshold_tuning_results.csv` contains accuracy, precision, recall, and F1 for each tested threshold.

---

## Interview Questions & Answers

### 1. How does logistic regression differ from linear regression?

- **Linear regression** predicts a continuous numeric output and fits a straight line by minimizing squared error.  
- **Logistic regression** predicts the probability of a binary class (between 0 and 1) by passing the linear combination of inputs through a **sigmoid** function.  
- It is trained by minimizing **log loss (cross-entropy)**, not squared error.  
- The output is a probability, which is then converted to a class using a decision threshold (commonly 0.5).

### 2. What is the sigmoid function?

The sigmoid (logistic) function maps any real number to a value between 0 and 1:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

where \( z = w \cdot x + b \).

- It converts the model’s raw linear score into a probability.
- \(\sigma(0) = 0.5\) acts as the natural midpoint.
- In the plot `images/sigmoid_function.png`, you can see how probabilities approach 0 for large negative \(z\) and 1 for large positive \(z\).

### 3. What is precision vs recall?

- **Precision** = TP / (TP + FP)  
  Of everything predicted positive, how many were actually positive. High precision means few false alarms.

- **Recall** = TP / (TP + FN)  
  Of everything actually positive, how many did the model catch. High recall means few missed positives.

There is usually a trade-off between the two, controlled by the decision threshold.

In this project, “positive” = benign (1), so:

- Precision = of all tumors predicted benign, how many are truly benign.
- Recall = of all truly benign tumors, how many were correctly predicted as benign.

### 4. What is the ROC-AUC curve?

- The **ROC (Receiver Operating Characteristic)** curve plots the **True Positive Rate** vs **False Positive Rate** at every possible threshold.
- **AUC (Area Under the Curve)** summarizes this in one number from 0 to 1:
  - 0.5 ≈ random guessing
  - 1.0 ≈ perfect classifier
- AUC measures how well the model ranks positives above negatives, independent of any single threshold choice.

In this task, ROC-AUC ≈ 0.995 indicates excellent separation between malignant and benign tumors.

### 5. What is the confusion matrix?

A table comparing actual vs predicted classes:

|                  | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

It is the basis for computing accuracy, precision, recall, and F1-score.

In the saved `images/confusion_matrix.png`, the labels correspond to:

- Negative = malignant (0)
- Positive = benign (1)

### 6. What happens if classes are imbalanced?

- Accuracy becomes misleading—a model that always predicts the majority class can still score high accuracy while being useless for the minority class.
- Precision/recall/F1 (especially for the minority class), ROC-AUC, and Precision-Recall AUC are more informative.
- Techniques to handle imbalance include:
  - Class weighting (`class_weight='balanced'`)
  - Resampling (oversampling minority / undersampling majority, e.g. SMOTE)
  - Choosing an appropriate decision threshold instead of the default 0.5
  - Using stratified train/test splits (as done in this project) to preserve class proportions.

### 7. How do you choose the threshold?

- The default is 0.5, but the optimal threshold depends on the cost of false positives vs false negatives for the specific problem.
- You can inspect the precision-recall vs threshold curve (or ROC curve) and pick the threshold that meets a business/domain requirement.
- In medical diagnosis, we often prefer higher recall for the dangerous class (e.g., malignant) even if it reduces precision, because missing a disease case is worse than a false alarm.
- In this code, the positive class is benign, so lowering the threshold increases recall for benign. To focus on malignant, we would either recode the target or explicitly analyze metrics for class 0.

### 8. Can logistic regression be used for multi-class problems?

Yes. Common approaches:

- **One-vs-Rest (OvR):** train one binary classifier per class.
- **Multinomial (softmax) logistic regression:** directly generalizes the sigmoid to multiple classes using the softmax function.

`sklearn.linear_model.LogisticRegression` supports both via the `multi_class` parameter. This project uses binary classification (malignant vs benign), so the default binary setting is sufficient.

---

## Notes

- All metrics and plots are generated by `logistic_regression_classifier.py`.
- The script uses a fixed `RANDOM_STATE = 42` so results are reproducible.
- No paid tools or external downloads are required beyond standard Python libraries.
