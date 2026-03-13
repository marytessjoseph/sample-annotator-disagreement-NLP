# Import required libraries
import os                      # Used for working with file paths
import pandas as pd            # Used for data manipulation (DataFrames)
import numpy as np             # Used for numerical operations
from math import log2          # Used to calculate log base 2 for entropy


#%% Function to calculate entropy (disagreement score)
def calculate_entropy(labels):
    """
    Calculates Shannon entropy for a list of annotator labels.
    Higher entropy = more disagreement between annotators.
    """
    
    # Find unique values and how many times each appears
    values, counts = np.unique(labels, return_counts=True)
    
    # Convert counts into probabilities
    probabilities = counts / counts.sum()
    
    # Calculate entropy using the formula: H = -Σ p log2(p)
    entropy = -sum(p * log2(p) for p in probabilities if p > 0)
    
    # Return entropy value
    return entropy


#%% Function to convert disagreement score into a category
def disagreement_category(score, threshold=0.5):
    """
    Converts the entropy score into a category.
    If score >= threshold → High disagreement
    Otherwise → Low disagreement
    """
    
    if score >= threshold:
        return "High"
    else:
        return "Low"


#%% Step 1: Define file paths based on project structure

# Get the base directory of the project (sample-annotator-disagreement-NLP)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to input dataset
INPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "dataset.csv")

# Path to processed output dataset
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset_with_disagreement.csv")


#%% Step 2: Load the dataset

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(INPUT_PATH)

# Print the original dataset to console
print("Original dataset:")
print(df)

# Print separator line for readability
print("\n" + "-" * 50 + "\n")


#%% Step 3: Define annotator columns

# These columns contain the labels given by annotators
annotator_cols = ["ann1", "ann2", "ann3", "ann4", "ann5"]


#%% Step 4: Compute disagreement score using entropy

# Apply the entropy function to each row of annotator labels
df["disagreement_score"] = df[annotator_cols].apply(
    lambda row: calculate_entropy(row.values),  # Calculate entropy for that row
    axis=1                                      # axis=1 means row-wise operation
)


#%% Step 5: Convert entropy score into High/Low category

# Apply categorization function to disagreement_score column
df["disagreement_category"] = df["disagreement_score"].apply(disagreement_category)


#%% Step 6: Create a clean output dataframe for viewing in Spyder

# Select only useful columns for viewing
output_df = df[["sentence", "disagreement_score", "disagreement_category"]].copy()

# Round the score to 3 decimal places for readability
output_df["disagreement_score"] = output_df["disagreement_score"].round(3)


#%% Step 7: Print the final output

print("Final output:")
print(output_df)


#%% Step 8: Save the processed dataset

# Save the updated dataframe to processed folder
# If the file already exists, it will be overwritten with new content
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nProcessed dataset saved to: {OUTPUT_PATH}")