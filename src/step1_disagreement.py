# import pandas as pd
# import numpy as np
# from math import log2

# #%% Function to calculate entropy
# def calculate_entropy(labels):
#     values, counts = np.unique(labels, return_counts=True)
#     probabilities = counts / counts.sum()
#     entropy = -sum(p * log2(p) for p in probabilities if p > 0)
#     return entropy

# #%% Function to convert score into category
# def disagreement_category(score, threshold=0.5):
#     if score >= threshold:
#         return "High"
#     else:
#         return "Low"

# #%% Step 1: Load dataset
# df = pd.read_csv("hatexplain_train.csv")

# print("Original dataset:")
# print(df)
# print("\n" + "-" * 50 + "\n")

# #%% Step 2: Compute disagreement score
# annotator_cols = ["ann1", "ann2", "ann3", "ann4", "ann5"]

# df["disagreement_score"] = df[annotator_cols].apply(
#     lambda row: calculate_entropy(row.values),
#     axis=1
# )

# #%% Step 3: Convert score into High/Low
# df["disagreement_category"] = df["disagreement_score"].apply(disagreement_category)

# print("Final output:")
# print(df[["sentence", "disagreement_score", "disagreement_category"]].round(3))

# #%% Save result
# df.to_csv("dataset_with_disagreement.csv", index=False)

# print("\nSaved as dataset_with_disagreement.csv")


import pandas as pd
import numpy as np
import re
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


#%% Function to extract label list from the annotators column
def extract_labels(annotator_string):
    """
    Extract labels from strings like:
    {'label': array([0, 2, 2], dtype=int64), ...}
    """
    match = re.search(r"'label': array\(\[([0-9,\s]+)\]", annotator_string)
    if match:
        label_text = match.group(1)
        labels = [int(x.strip()) for x in label_text.split(",")]
        return labels
    return []


#%% Function to convert post_tokens string into readable sentence
def extract_sentence(post_tokens_string):
    """
    Convert strings like:
    ['u' 'really' 'think' 'i' 'would']
    into:
    u really think i would
    """
    tokens = re.findall(r"'([^']+)'", post_tokens_string)
    return " ".join(tokens)


#%% Step 1: Load dataset
df = pd.read_csv("hatexplain_train.csv")

print("Original dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\n" + "-" * 50 + "\n")


#%% Step 2: Extract annotator labels
df["label_list"] = df["annotators"].apply(extract_labels)

print("Example extracted labels:")
print(df["label_list"].head())
print("\n" + "-" * 50 + "\n")


#%% Step 3: Compute disagreement score using entropy
df["disagreement_score"] = df["label_list"].apply(calculate_entropy)


#%% Step 4: Convert score into High/Low
df["disagreement_category"] = df["disagreement_score"].apply(disagreement_category)


#%% Step 5: Reconstruct sentence text from post_tokens
df["sentence"] = df["post_tokens"].apply(extract_sentence)


#%% Show final useful columns
print("Final output sample:")
print(df[["id", "sentence", "label_list", "disagreement_score", "disagreement_category"]].head())


#%% Save processed dataset
df.to_csv("hatexplain_with_disagreement.csv", index=False)

print("\nSaved file: hatexplain_with_disagreement.csv")
