# ==============================================================
# hate_dataset_disagreement.py
# Compute annotator disagreement for HateXplain dataset
# ==============================================================

import os
import re
import pandas as pd
import numpy as np
from math import log2


# ==============================================================
# Function: Calculate Shannon Entropy
# ==============================================================

def calculate_entropy(labels):
    """
    Calculates Shannon entropy for annotator labels.

    Entropy measures disagreement:
        0       -> complete agreement
        0.918   -> 2 vs 1 disagreement
        1.585   -> all annotators disagree
    """

    labels = pd.Series(labels).dropna().values

    if len(labels) == 0:
        return np.nan

    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()

    entropy = -sum(p * log2(p) for p in probabilities if p > 0)

    return max(entropy, 0.0)


# ==============================================================
# Function: Convert score to category
# ==============================================================

def disagreement_category(score):
    """
    Binary disagreement classification.

    0   -> Low disagreement
    >0  -> High disagreement
    """

    if pd.isna(score):
        return np.nan

    if score == 0:
        return "Low"
    else:
        return "High"


# ==============================================================
# Function: Extract annotator labels
# ==============================================================

def extract_labels_from_annotators(annotator_text):
    """
    Extract labels from HateXplain annotator column.

    Example string:
    {'label': array([0, 2, 2], dtype=int64), ...}

    Returns:
        [0,2,2]
    """

    if pd.isna(annotator_text):
        return []

    annotator_text = str(annotator_text)

    match = re.search(r"'label'\s*:\s*array\(\[([0-9,\s]+)\]", annotator_text)

    if not match:
        return []

    label_text = match.group(1)

    labels = [int(x.strip()) for x in label_text.split(",") if x.strip()]

    return labels


# ==============================================================
# Function: Rebuild sentence from tokens
# ==============================================================

def extract_sentence_from_post_tokens(token_text):
    """
    Convert HateXplain token list to sentence.
    """

    if pd.isna(token_text):
        return ""

    token_text = str(token_text)

    tokens = re.findall(r"'([^']*)'", token_text)

    sentence = " ".join(tokens)

    sentence = re.sub(r"\s+([.,!?;:])", r"\1", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()

    return sentence


# ==============================================================
# Function: Identify label pattern
# ==============================================================

def label_pattern_type(labels):
    """
    Classify annotator label pattern.
    """

    if not labels or len(labels) != 3:
        return "Other"

    unique_count = len(set(labels))

    if unique_count == 1:
        return "Unanimous"
    elif unique_count == 2:
        return "Two-vs-One"
    elif unique_count == 3:
        return "All-Different"
    else:
        return "Other"


# ==============================================================
# Step 1: Define paths
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "hatexplain_train.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "hatexplain_with_disagreement.csv")


# ==============================================================
# Step 2: Load dataset
# ==============================================================

df = pd.read_csv(INPUT_PATH)

print("Original dataset preview:")
print(df.head())

print("\n" + "-"*60 + "\n")


# ==============================================================
# Step 3: Verify required columns
# ==============================================================

required_cols = ["id", "annotators", "post_tokens"]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")


# ==============================================================
# Step 4: Extract labels and sentences
# ==============================================================

df["labels"] = df["annotators"].apply(extract_labels_from_annotators)

df["sentence"] = df["post_tokens"].apply(extract_sentence_from_post_tokens)

print("Sample extracted labels and sentences:")

print(df[["id","labels","sentence"]].head())

print("\n" + "-"*60 + "\n")


# ==============================================================
# Step 5: Compute disagreement scores
# ==============================================================

df["disagreement_score"] = df["labels"].apply(calculate_entropy)

df["disagreement_score_rounded"] = df["disagreement_score"].round(3)


# ==============================================================
# Step 6: Create disagreement category
# ==============================================================

df["disagreement_category"] = df["disagreement_score"].apply(disagreement_category)


# ==============================================================
# Step 7: Validation analysis
# ==============================================================

df["label_pattern"] = df["labels"].apply(label_pattern_type)

print("Label pattern distribution:")

print(df["label_pattern"].value_counts())

print("\n" + "-"*60 + "\n")

print("Disagreement score distribution:")

print(df["disagreement_score_rounded"].value_counts().sort_index())

print("\n" + "-"*60 + "\n")

print("Pattern vs Entropy score:")

print(pd.crosstab(df["label_pattern"], df["disagreement_score_rounded"]))

print("\n" + "-"*60 + "\n")

# ==============================================================
# Step 8: Create minimal output dataset
# ==============================================================

# Only keep the columns needed for the next pipeline step
output_df = df[
    [
        "id",
        "sentence",
        "disagreement_score",
        "disagreement_category"
    ]
].copy()

# Round score for readability
output_df["disagreement_score"] = output_df["disagreement_score"].round(3)

# Fix floating point artifact (-0.000 → 0.000)
output_df["disagreement_score"] = output_df["disagreement_score"].apply(
    lambda x: 0.0 if abs(x) < 1e-10 else x
)

print("Final output preview:")
print(output_df.head())

print("\nDataset shape:", output_df.shape)


# ==============================================================
# Step 9: Save dataset
# ==============================================================

# Ensure folder exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Remove existing file if it exists (avoids Windows permission issue)
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

# Save processed dataset
output_df.to_csv(OUTPUT_PATH, index=False)

print("\nProcessed dataset saved to:")
print(OUTPUT_PATH)