
# ==============================================================
# hate_dataset_model_training.py
# Logistic regression using only spaCy-derived features
# ==============================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


# ==============================================================
# Step 1: Define paths
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "hatexplain_with_features.csv")


# ==============================================================
# Step 2: Load dataset
# ==============================================================

df = pd.read_csv(INPUT_PATH)

print("Dataset preview:")
print(df.head())

print("\n" + "-"*60 + "\n")


# ==============================================================
# Step 3: Define target and excluded columns
# ==============================================================

target_col = "disagreement_category"

exclude_cols = [
    "id",
    "sentence",
    "disagreement_score",
    "disagreement_category",
]


# ==============================================================
# Step 4: Prepare features and labels
# ==============================================================

feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols].copy()

y = df[target_col].map({"Low": 0, "High": 1})


# ==============================================================
# Step 5: Drop rows where target label is NaN
#
# disagreement_category returns np.nan for missing entropy values
# (see hate_dataset_disagreement.py: disagreement_category()).
# Rows with NaN targets must be removed before splitting,
# otherwise stratify=y in train_test_split raises a ValueError.
# ==============================================================

nan_count = y.isna().sum()

if nan_count > 0:
    print(f"Dropping {nan_count} rows with NaN disagreement_category.")
    valid_idx = y.dropna().index
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

print(f"Feature count: {len(feature_cols)}")
print(f"Class distribution:\n{y.value_counts()}")

print("\n" + "-"*60 + "\n")


# ==============================================================
# Step 6: 5-Fold Cross-Validation
#
# Instead of a single 80/20 split, we split the data 5 times.
# Each fold uses a different 20% as the test set so every row
# gets tested exactly once. We report mean and std across folds
# so the paper can cite a stable, reliable performance estimate
# rather than a single potentially lucky or unlucky split.
#
# StratifiedKFold preserves the Low/High class ratio in every
# fold, which matters because the classes are imbalanced.
#
# We wrap scaler + model in a Pipeline so the scaler is fit
# only on training folds and never sees test fold data —
# this prevents data leakage across folds.
# ==============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        max_iter=4000,
        class_weight="balanced",
        random_state=42,
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "f1_weighted", "roc_auc"],
    return_train_score=False,
)

print("5-Fold Cross-Validation Results:")
print("-"*60)

for metric, key in [
    ("Accuracy",     "test_accuracy"),
    ("F1 Weighted",  "test_f1_weighted"),
    ("ROC-AUC",      "test_roc_auc"),
]:
    scores = cv_results[key]
    print(f"{metric}:")
    print(f"  Per fold : {[round(s, 4) for s in scores]}")
    print(f"  Mean     : {round(scores.mean(), 4)}")
    print(f"  Std      : {round(scores.std(), 4)}")
    print()

print("\n" + "-"*60 + "\n")


# ==============================================================
# Step 7: Final model — train on full data
#
# After cross-validation confirms performance is stable,
# we train one final model on the entire dataset to get
# the most reliable coefficient estimates for interpretation.
# This model is used only for feature analysis, not evaluation
# (evaluation was done via CV above).
# ==============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    max_iter=4000,
    class_weight="balanced",
    random_state=42,
)

final_model.fit(X_scaled, y)

print("Final model trained on full dataset.")

print("\n" + "-"*60 + "\n")


# ==============================================================
# Step 8: Held-out evaluation (single split for report)
#
# We also run a single 80/20 split so we can produce a full
# classification report (per-class precision, recall, F1)
# which cross_validate does not return directly.
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler_held = StandardScaler()
X_train_scaled = scaler_held.fit_transform(X_train)
X_test_scaled  = scaler_held.transform(X_test)

held_model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    max_iter=4000,
    class_weight="balanced",
    random_state=42,
)

held_model.fit(X_train_scaled, y_train)
y_pred = held_model.predict(X_test_scaled)

print("Held-out Evaluation (80/20 split):")
print("-"*60)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "High"]))

print("\n" + "-"*60 + "\n")


# ==============================================================
# Step 9: Inspect feature coefficients
#
# We use the final model (trained on all data) for coefficient
# analysis because it has seen the most data and gives the
# most stable estimates of feature importance.
# ==============================================================

coef_df = pd.DataFrame({
    "feature":     feature_cols,
    "coefficient": final_model.coef_[0],
})

# L1 penalty zeroes out uninformative features — keep only active ones
coef_df = coef_df[coef_df["coefficient"] != 0].copy()

print(f"Active features after L1 regularisation: {len(coef_df)} / {len(feature_cols)}")

print("\n" + "-"*60 + "\n")

print("Top features increasing disagreement (High):")
print(
    coef_df
    .sort_values("coefficient", ascending=False)
    [["feature", "coefficient"]]
    .head(10)
    .to_string(index=False)
)

print("\n" + "-"*60 + "\n")

print("Top features decreasing disagreement (Low):")
print(
    coef_df
    .sort_values("coefficient", ascending=True)
    [["feature", "coefficient"]]
    .head(10)
    .to_string(index=False)
)

print("\n" + "-"*60 + "\n")






















'''# ==============================================================
# hate_dataset_model_training.py
# Logistic regression using only spaCy-derived features
# ==============================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "hatexplain_with_features.csv")

# Load dataset
df = pd.read_csv(INPUT_PATH)

# Target and excluded columns
target_col = "disagreement_category"
exclude_cols = [
    "id",
    "sentence",
    "disagreement_score",
    "disagreement_category"

]

# Features and labels
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].copy()
y = df[target_col].map({"Low": 0, "High": 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression
model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    max_iter=4000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluation
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "High"]))

# Coefficients
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_[0]
})

coef_df = coef_df[coef_df["coefficient"] != 0].copy()

print("\nTop features increasing disagreement (High):")
print(
    coef_df.sort_values("coefficient", ascending=False)[["feature", "coefficient"]].head(10)
)

print("\nTop features decreasing disagreement (Low):")
print(
    coef_df.sort_values("coefficient", ascending=True)[["feature", "coefficient"]].head(10)
)


'''

