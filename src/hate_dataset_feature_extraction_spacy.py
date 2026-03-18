# ==============================================================
# hate_dataset_feature_extraction_spacy.py
#
# Purpose:
# Extract interpretable linguistic features from the HateXplain
# dataset using spaCy and measure processing time per batch.
#
# This script corresponds to:
# Module D: Linguistic Feature Extraction
#
# Input:
# hatexplain_with_disagreement.csv
#
# Output:
# hatexplain_with_features.csv
#
# ==============================================================


# --------------------------------------------------------------
# Import required libraries
# --------------------------------------------------------------

import os              # for file path operations
import re              # for regex pattern matching
import time            # for measuring processing time
import math            # for batch calculations
import pandas as pd    # for dataset manipulation
import numpy as np     # for numeric operations
import spacy           # NLP library used for linguistic analysis


# --------------------------------------------------------------
# Step 0: Load spaCy language model
# --------------------------------------------------------------
# spaCy provides:
# - tokenization
# - part-of-speech tagging
# - dependency parsing
# - sentence segmentation

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' is not installed.\n"
        "Run this first:\n"
        "python -m spacy download en_core_web_sm"
    )


# --------------------------------------------------------------
# Step 1: Define input/output file paths
# --------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "hatexplain_with_disagreement.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "hatexplain_with_features.csv"
)


# --------------------------------------------------------------
# Step 2: Load dataset
# --------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("Input dataset preview:")
print(df.head())

print("\n" + "-" * 60 + "\n")


# --------------------------------------------------------------
# Step 3: Verify required columns exist
# --------------------------------------------------------------

required_cols = [
    "id",
    "sentence",
    "disagreement_score",
    "disagreement_category"
]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")


# --------------------------------------------------------------
# Step 4: Define lexical resources
# --------------------------------------------------------------
# These lists represent interpretable linguistic markers.

# Negation cues
NEGATIONS = {
    "no","not","never","none","nothing","nowhere","neither","nor",
    "cannot","can't","dont","don't","doesnt","doesn't",
    "didnt","didn't","wont","won't","wouldnt","wouldn't"
}

# Hedging / uncertainty words
HEDGES = {
    "maybe","perhaps","possibly","probably","likely","unlikely",
    "seems","appear","might","may","could","generally","sometimes"
}

# Hedging phrases
HEDGE_PHRASES = {
    "i think",
    "i guess",
    "i believe",
    "in my opinion"
}

# Ambiguity markers
AMBIGUITY_CUES = {
    "some","someone","something","maybe","perhaps","kind","sort",
    "stuff","things","whatever"
}

# Sarcasm / informal markers
SARCASM_MARKERS = {
    "yeah right",
    "sure",
    "obviously",
    "lol",
    "lmao",
    "haha"
}

# Pronoun sets
FIRST_PERSON_PRONOUNS = {"i","me","my","mine","we","us","our"}
SECOND_PERSON_PRONOUNS = {"you","your","yours","u","ur"}
THIRD_PERSON_PRONOUNS = {"he","she","they","them","their","it"}


# --------------------------------------------------------------
# Step 5: Helper functions
# --------------------------------------------------------------

def contains_phrase(text, phrase_set):
    """
    Count occurrences of phrases within a text.
    """
    text_lower = text.lower()

    count = 0

    for phrase in phrase_set:
        if phrase in text_lower:
            count += 1

    return count


def get_dependency_depth(token):
    """
    Compute approximate depth of a token in dependency tree.
    """

    depth = 0
    current = token

    while current.head != current:
        depth += 1
        current = current.head

    return depth


def safe_divide(a, b):
    """
    Prevent division by zero errors.
    """
    return a / b if b != 0 else 0.0


# --------------------------------------------------------------
# Step 6: Feature extraction function
# --------------------------------------------------------------

def extract_features(doc, raw_text):

    """
    Extract linguistic features from a spaCy document.
    """

    text = str(raw_text)

    tokens = [t for t in doc if not t.is_space]
    alpha_tokens = [t for t in tokens if t.is_alpha]

    token_texts = [t.text.lower() for t in tokens]
    alpha_texts = [t.text.lower() for t in alpha_tokens]

    # ----------------------------------------------------------
    # Surface-level features
    # ----------------------------------------------------------

    sentence_length_tokens = len(tokens)
    sentence_length_chars = len(text)

    avg_token_length = (
        np.mean([len(t.text) for t in tokens])
        if tokens else 0
    )

    # ----------------------------------------------------------
    # Negation
    # ----------------------------------------------------------

    negation_count = sum(
        1 for tok in token_texts if tok in NEGATIONS
    )

    negation_binary = 1 if negation_count > 0 else 0


    # ----------------------------------------------------------
    # Hedging
    # ----------------------------------------------------------

    hedge_count = (
        sum(1 for tok in alpha_texts if tok in HEDGES)
        + contains_phrase(text, HEDGE_PHRASES)
    )

    hedging_binary = 1 if hedge_count > 0 else 0


    # ----------------------------------------------------------
    # Pronouns
    # ----------------------------------------------------------

    first_person = sum(1 for tok in token_texts if tok in FIRST_PERSON_PRONOUNS)
    second_person = sum(1 for tok in token_texts if tok in SECOND_PERSON_PRONOUNS)
    third_person = sum(1 for tok in token_texts if tok in THIRD_PERSON_PRONOUNS)

    pronoun_total = first_person + second_person + third_person


    # ----------------------------------------------------------
    # Ambiguity
    # ----------------------------------------------------------

    ambiguity_count = sum(
        1 for tok in alpha_texts if tok in AMBIGUITY_CUES
    )

    ambiguity_binary = 1 if ambiguity_count > 0 else 0


    # ----------------------------------------------------------
    # Sarcasm indicators
    # ----------------------------------------------------------

    sarcasm_count = (
        contains_phrase(text, SARCASM_MARKERS)
        + text.count("lol")
        + text.count("haha")
    )

    sarcasm_binary = 1 if sarcasm_count > 0 else 0


    # ----------------------------------------------------------
    # POS-based features
    # ----------------------------------------------------------

    noun_count = sum(1 for t in tokens if t.pos_ in {"NOUN","PROPN"})
    verb_count = sum(1 for t in tokens if t.pos_ in {"VERB","AUX"})
    adj_count = sum(1 for t in tokens if t.pos_ == "ADJ")
    adv_count = sum(1 for t in tokens if t.pos_ == "ADV")


    # ----------------------------------------------------------
    # Syntactic complexity
    # ----------------------------------------------------------

    dependency_depths = [get_dependency_depth(t) for t in tokens]

    max_depth = max(dependency_depths) if dependency_depths else 0
    mean_depth = np.mean(dependency_depths) if dependency_depths else 0


    # ----------------------------------------------------------
    # Lexical diversity
    # ----------------------------------------------------------

    lexical_diversity = safe_divide(
        len(set(alpha_texts)),
        len(alpha_texts)
    )


    # ----------------------------------------------------------
    # Return features as dictionary
    # ----------------------------------------------------------

    return {

        "sentence_length_tokens": sentence_length_tokens,
        "sentence_length_chars": sentence_length_chars,
        "avg_token_length": avg_token_length,

        "negation_count": negation_count,
        "negation_binary": negation_binary,

        "hedging_count": hedge_count,
        "hedging_binary": hedging_binary,

        "pronoun_total_count": pronoun_total,

        "ambiguity_count": ambiguity_count,
        "ambiguity_binary": ambiguity_binary,

        "sarcasm_marker_count": sarcasm_count,
        "sarcasm_binary": sarcasm_binary,

        "noun_count": noun_count,
        "verb_count": verb_count,
        "adj_count": adj_count,
        "adv_count": adv_count,

        "max_dependency_depth": max_depth,
        "mean_dependency_depth": mean_depth,

        "lexical_diversity": lexical_diversity
    }


# --------------------------------------------------------------
# Step 7: Run spaCy pipeline in batches
# --------------------------------------------------------------

print("Running spaCy feature extraction...")

batch_size = 64

texts = df["sentence"].fillna("").astype(str).tolist()

total_texts = len(texts)
total_batches = math.ceil(total_texts / batch_size)

print(f"Total texts: {total_texts}")
print(f"Batch size: {batch_size}")
print(f"Total batches: {total_batches}")

print("\n" + "-" * 60 + "\n")

features = []

overall_start_time = time.time()


for batch_num, start_idx in enumerate(range(0, total_texts, batch_size), start=1):

    end_idx = min(start_idx + batch_size, total_texts)

    batch_texts = texts[start_idx:end_idx]

    batch_start = time.time()

    docs = list(nlp.pipe(batch_texts, batch_size=batch_size))

    for local_idx, doc in enumerate(docs):

        global_idx = start_idx + local_idx

        features.append(
            extract_features(doc, texts[global_idx])
        )

    batch_time = time.time() - batch_start

    processed = end_idx

    print(
        f"Batch {batch_num}/{total_batches} | "
        f"Rows {start_idx}-{end_idx-1} | "
        f"Batch time: {batch_time:.2f}s | "
        f"Processed: {processed}/{total_texts}"
    )


total_time = time.time() - overall_start_time

print("\nFeature extraction completed in", round(total_time,2),"seconds")


# --------------------------------------------------------------
# Step 8: Merge features with original dataset
# --------------------------------------------------------------

features_df = pd.DataFrame(features)

final_df = pd.concat(
    [
        df.reset_index(drop=True),
        features_df
    ],
    axis=1
)

print("\nFinal dataset preview:")
print(final_df.head())

print("\nDataset shape:", final_df.shape)


# --------------------------------------------------------------
# Step 9: Save output dataset
# --------------------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

final_df.to_csv(OUTPUT_PATH, index=False)

print("\nProcessed feature dataset saved to:")
print(OUTPUT_PATH)