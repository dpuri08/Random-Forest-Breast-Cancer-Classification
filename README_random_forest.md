# Random Forest Classification — Breast Cancer Wisconsin

A machine learning project exploring **Random Forest classification** on the Breast Cancer Wisconsin dataset. The notebook builds a baseline Random Forest, evaluates out-of-bag (OOB) performance, tunes hyperparameters with `GridSearchCV`, and compares the results with a Decision Tree.

## Objective

Classify tumors as **malignant** or **benign** and investigate how ensemble learning, feature importance, OOB evaluation, and hyperparameter tuning affect model performance.

## Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569
- **Features:** 30 numerical features
- **Classes:** malignant and benign
- The dataset is loaded directly from scikit-learn; no separate download is required.

## Workflow

1. Load and inspect the dataset.
2. Split the data into training and test sets (80/20), using stratification.
3. Train a baseline `RandomForestClassifier`.
4. Evaluate the model with accuracy, a classification report, and a confusion matrix.
5. Inspect and visualize feature importances.
6. Evaluate the model using the out-of-bag (OOB) score.
7. Tune `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` using 5-fold `GridSearchCV`.
8. Evaluate the tuned model on the held-out test set.
9. Compare the Random Forest models with a Decision Tree.

## Results

Results below reflect the saved notebook outputs. Re-run all cells before publishing to confirm they match the current code.

| Evaluation | Result |
|---|---:|
| Baseline Random Forest test accuracy | 96.49% |
| Baseline Random Forest OOB score | 95.82% |
| Best GridSearchCV mean CV macro F1 | 95.97% |
| Tuned Random Forest test accuracy | 96.49% |
| Decision Tree test accuracy (`max_depth=10`) | 96.49% |

**Selected GridSearchCV parameters**

- `n_estimators`: 200
- `max_depth`: `None`
- `min_samples_split`: 5
- `min_samples_leaf`: 1
- `max_features`: `sqrt`

The baseline and tuned Random Forest have the same test accuracy on this particular split. The tuned model should therefore be discussed using the cross-validation macro F1 score and class-level metrics as well—not accuracy alone.

## Feature Importance

The highest-ranked features in the saved baseline model include:

- `worst concave points`
- `worst perimeter`
- `worst area`
- `mean concave points`
- `worst radius`

Feature importance describes how the fitted forest uses features to reduce impurity. It should not be interpreted as proof of causation or clinical importance.

## Technologies

- Python
- Jupyter Notebook
- pandas
- Matplotlib
- scikit-learn

## How to Run

Install the dependencies:

```bash
pip install pandas matplotlib scikit-learn jupyter
```

Open `random_forest_classification.ipynb` in Jupyter Notebook or VS Code and run all cells from top to bottom.

## Suggested Repository Structure

```text
Random-Forest-Breast-Cancer-Classification/
├── random_forest_classification.ipynb
└── README.md
```

## Notes and Limitations

- The test set is held out from model fitting and hyperparameter selection; cross-validation is performed on the training set.
- The notebook uses one train/test split and one random seed. Results may vary with a different split.
- This is an educational machine-learning project, not a medical diagnostic tool.
- Impurity-based feature importance can favor certain feature types and can be misleading when predictors are correlated.

## Author

Deepanshu Puri
