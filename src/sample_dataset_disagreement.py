# ==============================================================
# Import required libraries
# ==============================================================

import os                      # Used for working with file paths
import pandas as pd            # Used for data manipulation (DataFrames)
import numpy as np             # Used for numerical operations
from math import log2          # Used to calculate log base 2 for entropy


# ==============================================================
# Function to calculate entropy (disagreement score)
# ==============================================================

def calculate_entropy(labels):
    """
    Calculates Shannon entropy for a list of annotator labels.
    
    Entropy measures how much annotators disagree.
    
    Low entropy  → annotators mostly agree
    High entropy → annotators disagree more
    
    Formula:
        H = - Σ (p * log2(p))
    where p is the probability of each label.
    """

    # Remove missing labels if any exist
    labels = pd.Series(labels).dropna().values

    # If no labels remain after cleaning, return NaN
    if len(labels) == 0:
        return np.nan

    # Find unique label values and their counts
    values, counts = np.unique(labels, return_counts=True)

    # Convert counts into probabilities
    probabilities = counts / counts.sum()

    # Apply Shannon entropy formula
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)

    # Prevent tiny floating point negatives like -0.000
    return max(entropy, 0.0)


# ==============================================================
# Function to convert disagreement score into a category
# ==============================================================

def disagreement_category(score, threshold):
    """
    Converts entropy score into a disagreement category.

    If score >= threshold → High disagreement
    Otherwise → Low disagreement
    """

    if score >= threshold:
        return "High"
    else:
        return "Low"


# ==============================================================
# Step 1: Define file paths based on project structure
# ==============================================================

# Get the base directory of the project
# Example:
# sample-annotator-disagreement-NLP/
# ├── data/
# ├── src/
# └── notebooks/

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to input dataset
INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "dataset.csv")

# Path to processed output dataset
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset_with_disagreement.csv")


# ==============================================================
# Step 2: Load the dataset
# ==============================================================

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(INPUT_PATH)

# Display the original dataset
print("Original dataset:")
print(df)

# Print separator line for readability
print("\n" + "-" * 50 + "\n")


# ==============================================================
# Step 3: Define annotator columns
# ==============================================================

# These columns contain labels assigned by annotators
annotator_cols = ["ann1", "ann2", "ann3", "ann4", "ann5"]

# Verify that these columns exist in the dataset
missing_cols = [col for col in annotator_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing annotator columns: {missing_cols}")


# ==============================================================
# Step 4: Compute disagreement score using entropy
# ==============================================================

# Apply entropy calculation row-wise
# Each row contains the labels given by annotators

df["disagreement_score"] = df[annotator_cols].apply(
    lambda row: calculate_entropy(row.values),
    axis=1
)


# ==============================================================
# Step 5: Convert entropy score into High/Low category
# ==============================================================

# Instead of using a fixed threshold, use the median score
# This creates a balanced split of the dataset

threshold = df["disagreement_score"].median()

# Apply the category function using the computed threshold
df["disagreement_category"] = df["disagreement_score"].apply(
    lambda score: disagreement_category(score, threshold)
)

# Display threshold used
print(f"Median threshold used: {threshold:.3f}")

# Show how many examples fall into each class
print("\nClass distribution:")
print(df["disagreement_category"].value_counts())

print("\n" + "-" * 50 + "\n")


# ==============================================================
# Step 6: Create a clean output dataframe for viewing
# ==============================================================

# Select the columns we want to display
output_df = df[["sentence", "disagreement_score", "disagreement_category"]].copy()

# Round score to 3 decimals for readability
output_df["disagreement_score"] = output_df["disagreement_score"].round(3)


# ==============================================================
# Step 7: Print the final processed output
# ==============================================================

print("Final output:")
print(output_df)


# ==============================================================
# Step 8: Save the processed dataset
# ==============================================================

# Save the updated dataset with disagreement scores
# into the processed data folder

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nProcessed dataset saved to: {OUTPUT_PATH}")