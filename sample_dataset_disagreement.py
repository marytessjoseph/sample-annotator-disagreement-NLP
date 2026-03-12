import pandas as pd
import numpy as np
from math import log2

#%% Function to calculate entropy
def calculate_entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    return entropy

#%% Function to convert score into category
def disagreement_category(score, threshold=0.5):
    if score >= threshold:
        return "High"
    else:
        return "Low"

#%% Step 1: Load dataset
df = pd.read_csv("dataset.csv")

print("Original dataset:")
print(df)
print("\n" + "-" * 50 + "\n")

#%% Step 2: Compute disagreement score
annotator_cols = ["ann1", "ann2", "ann3", "ann4", "ann5"]

df["disagreement_score"] = df[annotator_cols].apply(
    lambda row: calculate_entropy(row.values),
    axis=1
)

# #%% Step 3: Convert score into High/Low
df["disagreement_category"] = df["disagreement_score"].apply(disagreement_category)

print("Final output:")
print(df[["sentence", "disagreement_score", "disagreement_category"]].round(3))

# #%% Save result
df.to_csv("dataset_with_disagreement.csv", index=False)

print("\nSaved as dataset_with_disagreement.csv")