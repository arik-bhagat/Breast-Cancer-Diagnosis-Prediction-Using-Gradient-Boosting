# Breast Cancer Diagnosis Prediction Using Gradient Boosting

Python-based tumour diagnosis prediction using exploratory data analysis, mutual information, and a
Gradient Boosting classifier to classify breast masses as **Malignant** or **Benign**.

> Educational project only. Not a medical device and not for clinical use.

## Dataset
Wisconsin Diagnostic Breast Cancer (WDBC): 569 samples, 30 numeric features computed from digitised
images of fine-needle aspirates (mean, standard error and "worst" value of 10 nucleus measurements such
as radius, texture, perimeter, area, concavity). Target: `diagnosis` (M = malignant, B = benign).
`data.csv` was exported from `sklearn.datasets.load_breast_cancer`; the original is available from the
UCI Machine Learning Repository. Class balance: 357 benign (62.7%), 212 malignant (37.3%).

## Project structure
```
.
├── README_Breast_Cancer_Diagnosis.md
├── Breast_Cancer_Diagnosis_Gradient_Boosting_Using_Python.ipynb
├── data.csv
└── requirements.txt
```

## Workflow
1. Load and inspect data (types, missing values, duplicates)
2. EDA: class balance, feature distributions by diagnosis, correlation heatmap
3. Stratified 80/20 train/test split
4. Mutual information feature ranking (training split only)
5. Baseline Gradient Boosting with 5-fold stratified CV
6. Hyperparameter tuning with `GridSearchCV` (ROC-AUC)
7. Test-set evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve
8. Decision-threshold analysis (recall vs. precision)
9. Feature importance (impurity-based and permutation)
10. Single-sample prediction

## Results (random_state = 42)
| Metric | Value |
|---|---|
| Baseline 5-fold CV ROC-AUC | 0.991 |
| Tuned 5-fold CV ROC-AUC | 0.994 |
| Test accuracy | 0.965 |
| Test precision (malignant) | 1.000 |
| Test recall (malignant) | 0.905 |
| Test F1 | 0.950 |
| Test ROC-AUC | 0.993 |

Best parameters: `learning_rate=0.1, max_depth=2, n_estimators=300, subsample=1.0`.

Notes: the test set has only 114 rows, so a single misclassification moves the metrics noticeably. At the
default 0.5 threshold 4 malignant cases were missed; the threshold table in the notebook shows how lowering
it raises recall at some cost to precision, which is usually the preferred trade-off for screening.

## How to run
```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
jupyter notebook Breast_Cancer_Diagnosis_Gradient_Boosting_Using_Python.ipynb
```

## Possible extensions
- Compare against Random Forest, XGBoost/LightGBM and Logistic Regression
- Probability calibration and SHAP explanations
- Repeated cross-validation for tighter performance estimates
- External validation on a different dataset
