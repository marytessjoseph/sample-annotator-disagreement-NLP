import pandas as pd
import numpy as np
import re
from math import log2
from pathlib import Path


# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "hatexplain_train.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hatexplain_with_disagreement.csv"


# =========================
# Function to calculate entropy
# =========================
def calculate_entropy(labels):
    if not labels:
        return 0.0

    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -sum(p * log2(p) for p in probabilities if p > 0)


# =========================
# Function to classify disagreement
# thershold is hardcoded according to the input, 
#needs to explore how to find it according to input
# =========================
def disagreement_category(score, threshold=0.5):
    return "High" if score >= threshold else "Low"


# =========================
# Extract label list
# =========================
def extract_labels(annotator_string):
    if pd.isna(annotator_string):
        return []

    match = re.search(r"'label': array\(\[([0-9,\s]+)\]", str(annotator_string))
    if match:
        return [int(x.strip()) for x in match.group(1).split(",")]

    return []


# =========================
# Reconstruct sentence
# =========================
def extract_sentence(post_tokens_string):
    if pd.isna(post_tokens_string):
        return ""

    tokens = re.findall(r"'([^']+)'", str(post_tokens_string))
    return " ".join(tokens)


# =========================
# Main processing
# =========================
def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = ["id", "annotators", "post_tokens"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Dataset missing columns: {missing_cols}"
        )

    df["label_list"] = df["annotators"].apply(extract_labels)
    df["disagreement_score"] = df["label_list"].apply(calculate_entropy)
    df["disagreement_category"] = df["disagreement_score"].apply(disagreement_category)
    df["sentence"] = df["post_tokens"].apply(extract_sentence)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    return df


# =========================
# Run script
# =========================
if __name__ == "__main__":
    df = main()