# ==============================================================
# Import required libraries
# ==============================================================

import os                      # Used for working with file and folder paths
import pandas as pd            # Used for loading, saving, and handling tables
import re                      # Used for simple text pattern matching

# Make pandas show all columns when printing in the console
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


# ==============================================================
# Step 1: Define small word lists for linguistic features
# ==============================================================

# Words that often indicate negation
NEGATION_WORDS = {
    "no", "not", "never", "none", "nothing", "nobody", "neither",
    "nowhere", "hardly", "scarcely", "barely", "can't", "cannot",
    "won't", "don't", "didn't", "isn't", "wasn't", "shouldn't",
    "couldn't", "wouldn't"
}

# Words or phrases that often indicate hedging / uncertainty
HEDGE_WORDS = {
    "maybe", "perhaps", "probably", "possibly", "guess", "seems",
    "seem", "might", "could", "may", "apparently", "likely",
    "sort of", "kind of", "i think", "i guess"
}

# Common English pronouns
PRONOUN_WORDS = {
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "mine", "yours", "ours", "theirs"
}


# ==============================================================
# Step 2: Create helper functions
# ==============================================================

def clean_text(text):
    """
    Makes text easier to analyze.

    What this function does:
    1. Handles missing values safely
    2. Converts text to lowercase

    Example:
    "I Loved This Phone!" -> "i loved this phone!"
    """

    # If the value is missing, return an empty string
    if pd.isna(text):
        return ""

    # Convert text to lowercase
    return str(text).lower()


def tokenize_words(text):
    """
    Splits a sentence into words.

    Example:
    "I loved this phone!" -> ["i", "loved", "this", "phone"]

    The regex keeps words and contractions like "don't".
    """

    return re.findall(r"\b\w+(?:'\w+)?\b", clean_text(text))


def count_sentence_length(text):
    """
    Counts how many words are in the sentence.

    Example:
    "I loved this phone" -> 4
    """

    words = tokenize_words(text)
    return len(words)


def count_negations(text):
    """
    Counts how many negation words appear in the sentence.

    Example:
    "The service was not good" -> 1
    """

    words = tokenize_words(text)
    return sum(1 for word in words if word in NEGATION_WORDS)


def count_pronouns(text):
    """
    Counts how many pronouns appear in the sentence.

    Example:
    "I loved this phone" -> 1
    """

    words = tokenize_words(text)
    return sum(1 for word in words if word in PRONOUN_WORDS)


def count_hedges(text):
    """
    Counts hedge words and hedge phrases.

    Some hedges are single words:
        maybe, perhaps, might

    Some hedges are phrases:
        i think, i guess, kind of, sort of
    """

    clean = clean_text(text)
    words = tokenize_words(clean)

    # Count single-word hedges
    single_word_hedges = sum(
        1 for word in words
        if word in HEDGE_WORDS and " " not in word
    )

    # Count multi-word hedge phrases
    multi_word_hedges = sum(
        1 for phrase in HEDGE_WORDS
        if " " in phrase and phrase in clean
    )

    return single_word_hedges + multi_word_hedges


def count_question_marks(text):
    """
    Counts how many question marks are in the sentence.

    Example:
    "Really?" -> 1
    "Why??" -> 2
    """

    return clean_text(text).count("?")


def count_exclamation_marks(text):
    """
    Counts how many exclamation marks are in the sentence.

    Example:
    "Great!" -> 1
    "Amazing!!" -> 2
    """

    return clean_text(text).count("!")


# ==============================================================
# Step 3: Define file paths
# ==============================================================

# Find the main project folder automatically
# This script is inside /src, so we go up two levels
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input file created by the disagreement script
INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "dataset_with_disagreement.csv"
)

# Output file that will contain disagreement + linguistic features
OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "dataset_with_features.csv"
)


# ==============================================================
# Step 4: Load the processed dataset
# ==============================================================

# Read the CSV into a pandas DataFrame
df = pd.read_csv(INPUT_PATH)

# Print confirmation messages
print("Dataset loaded successfully.")
print(f"Number of rows: {len(df)}")
print(f"Columns found: {list(df.columns)}")
print("\n" + "-" * 60 + "\n")


# ==============================================================
# Step 5: Check that required columns exist
# ==============================================================

# We must have a sentence column because features are extracted from text
if "sentence" not in df.columns:
    raise ValueError("The input dataset does not contain a 'sentence' column.")

# We also expect disagreement columns from the previous script
required_cols = ["disagreement_score", "disagreement_category"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")


# ==============================================================
# Step 6: Extract linguistic features from the sentence column
# ==============================================================

# Feature 1: sentence length (number of words)
df["sentence_length"] = df["sentence"].apply(count_sentence_length)

# Feature 2: number of negation words
df["negation_count"] = df["sentence"].apply(count_negations)

# Feature 3: number of pronouns
df["pronoun_count"] = df["sentence"].apply(count_pronouns)

# Feature 4: number of hedge words / hedge phrases
df["hedge_count"] = df["sentence"].apply(count_hedges)

# Feature 5: number of question marks
df["question_mark_count"] = df["sentence"].apply(count_question_marks)

# Feature 6: number of exclamation marks
df["exclamation_mark_count"] = df["sentence"].apply(count_exclamation_marks)


# ==============================================================
# Step 7: Create a cleaner preview table
# ==============================================================

preview_cols = [
    "sentence",
    "disagreement_score",
    "disagreement_category",
    "sentence_length",
    "negation_count",
    "pronoun_count",
    "hedge_count",
    "question_mark_count",
    "exclamation_mark_count"
]

preview_df = df[preview_cols].copy()

print("Feature extraction completed.")
print("Preview of dataset with features:")
print(preview_df.to_string(index=False))
print("\n" + "-" * 60 + "\n")


# ==============================================================
# Step 8: Quick feature diagnostics
# ==============================================================

# This groups the data by High/Low disagreement
# and computes the average value of each feature in each group
print("Feature diagnostics: average feature values by disagreement category\n")

diagnostics = df.groupby("disagreement_category")[
    [
        "sentence_length",
        "negation_count",
        "pronoun_count",
        "hedge_count",
        "question_mark_count",
        "exclamation_mark_count"
    ]
].mean()

print(diagnostics.to_string())
print("\n" + "-" * 60 + "\n")


# ==============================================================
# Step 9: Numeric summary of feature columns
# ==============================================================

# This gives count, mean, std, min, max, etc. for numeric columns
print("Numeric summary of dataset:\n")

numeric_summary = df[
    [
        "disagreement_score",
        "sentence_length",
        "negation_count",
        "pronoun_count",
        "hedge_count",
        "question_mark_count",
        "exclamation_mark_count"
    ]
].describe()

print(numeric_summary.to_string())
print("\n" + "-" * 60 + "\n")


# ==============================================================
# Step 10: Small manual test sentence
# ==============================================================

# This is just to confirm the feature functions work as expected
test_sentence = "Yeah great job breaking it!"

print("Manual test sentence:")
print(test_sentence)
print("sentence_length =", count_sentence_length(test_sentence))
print("negation_count =", count_negations(test_sentence))
print("pronoun_count =", count_pronouns(test_sentence))
print("hedge_count =", count_hedges(test_sentence))
print("question_mark_count =", count_question_marks(test_sentence))
print("exclamation_mark_count =", count_exclamation_marks(test_sentence))
print("\n" + "-" * 60 + "\n")


# ==============================================================
# Step 11: Save the new dataset
# ==============================================================

df.to_csv(OUTPUT_PATH, index=False)

print(f"Feature dataset saved to: {OUTPUT_PATH}")


# ==============================================================
# Step 12: Keep variables visible in Spyder Variable Explorer
# ==============================================================

# These lines help Spyder keep these variables available
# so you can click them in the Variable Explorer
df
preview_df
diagnostics
numeric_summary