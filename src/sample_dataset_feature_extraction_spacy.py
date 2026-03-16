import os
import pandas as pd
import spacy

# Show all columns clearly in console output
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Load spaCy English model
# Keep POS tagging and lemmatization
# Disable parser and NER because we are not using them right now
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

# Lexicons for interpretable lexical features
NEGATION_WORDS = {
    "no", "not", "never", "none", "nothing", "nobody", "neither",
    "nowhere", "hardly", "scarcely", "barely", "can't", "cannot",
    "won't", "don't", "didn't", "isn't", "wasn't", "shouldn't",
    "couldn't", "wouldn't"
}

HEDGE_WORDS = {
    "maybe", "perhaps", "probably", "possibly", "guess", "seem",
    "might", "could", "may", "apparently", "likely",
    "sort of", "kind of", "i think", "i guess"
}


def extract_features(doc):
    """
    Extract linguistic features from one spaCy Doc.

    Key spaCy terms:
    - Doc: processed text object
    - token: one unit inside the Doc, usually a word or punctuation mark
    - token.text: original token text
    - token.lemma_: base form of the token
    - token.pos_: part-of-speech tag
    """

    text = doc.text.lower()

    # Count real words, excluding punctuation and spaces
    sentence_length = sum(
        1 for token in doc if not token.is_punct and not token.is_space
    )

    # POS-based features
    pronoun_count = sum(1 for token in doc if token.pos_ == "PRON")
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    verb_count = sum(1 for token in doc if token.pos_ in {"VERB", "AUX"})
    adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")

    # Negation count using token text + lemma
    negation_count = sum(
        1 for token in doc
        if token.text.lower() in NEGATION_WORDS
        or token.lemma_.lower() in NEGATION_WORDS
    )

    # Hedge count using token text + lemma for single words
    single_word_hedges = sum(
        1 for token in doc
        if (
            token.text.lower() in HEDGE_WORDS
            or token.lemma_.lower() in HEDGE_WORDS
        )
        and " " not in token.text
    )

    # Hedge count for multi-word phrases
    multi_word_hedges = sum(
        1 for phrase in HEDGE_WORDS
        if " " in phrase and phrase in text
    )

    hedge_count = single_word_hedges + multi_word_hedges

    # Punctuation features
    question_mark_count = text.count("?")
    exclamation_mark_count = text.count("!")

    return pd.Series({
        "sentence_length": sentence_length,
        "pronoun_count": pronoun_count,
        "noun_count": noun_count,
        "verb_count": verb_count,
        "adjective_count": adjective_count,
        "negation_count": negation_count,
        "hedge_count": hedge_count,
        "question_mark_count": question_mark_count,
        "exclamation_mark_count": exclamation_mark_count
    })


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "dataset_with_disagreement.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "dataset_with_features_spacy.csv"
)

df = pd.read_csv(INPUT_PATH)

required_cols = ["sentence", "disagreement_score", "disagreement_category"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Convert sentence column into clean strings
sentences = df["sentence"].fillna("").astype(str).tolist()

# Process all sentences efficiently with spaCy
docs = list(nlp.pipe(sentences, batch_size=100))

# Extract features from each processed Doc
feature_rows = [extract_features(doc) for doc in docs]
features_df = pd.DataFrame(feature_rows)

# Join features to original dataset
df = pd.concat([df, features_df], axis=1)

# Preview useful columns
preview_cols = [
    "sentence",
    "disagreement_score",
    "disagreement_category",
    "sentence_length",
    "pronoun_count",
    "noun_count",
    "verb_count",
    "adjective_count",
    "negation_count",
    "hedge_count",
    "question_mark_count",
    "exclamation_mark_count"
]

print(df[preview_cols].to_string(index=False))

# Save output
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")

# Keep visible in Spyder Variable Explorer
df
features_df