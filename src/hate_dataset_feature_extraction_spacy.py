
# ==============================================================
# hate_dataset_feature_extraction_spacy.py
# Extract only spaCy-derived linguistic features
# ==============================================================

import os
import time
import math
import pandas as pd
import numpy as np
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Install spaCy model first: python -m spacy download en_core_web_sm")

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "hatexplain_with_disagreement.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "hatexplain_with_features.csv")

# Load dataset
df = pd.read_csv(INPUT_PATH)

print("Dataset preview:")
print(df.head())


def safe_divide(a, b):
    return a / b if b else 0.0


def dependency_depth(token):
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
    return depth


def extract_features(doc, text):
    tokens = [t for t in doc if not t.is_space]
    alpha_tokens = [t for t in tokens if t.is_alpha]
    sents = list(doc.sents)

    lemmas = [t.lemma_.lower() for t in alpha_tokens]
    depths = [dependency_depth(t) for t in tokens] if tokens else [0]

    # Surface / length
    sentence_length_tokens = len(tokens)
    sentence_length_chars = len(text)
    avg_token_length = np.mean([len(t.text) for t in tokens]) if tokens else 0.0
    sentence_count = len(sents) if sents else 1
    avg_sentence_length = safe_divide(sentence_length_tokens, sentence_count)

    # Coarse POS counts
    noun_count = sum(t.pos_ == "NOUN" for t in tokens)
    proper_noun_count = sum(t.pos_ == "PROPN" for t in tokens)
    verb_count = sum(t.pos_ == "VERB" for t in tokens)
    aux_count = sum(t.pos_ == "AUX" for t in tokens)
    adj_count = sum(t.pos_ == "ADJ" for t in tokens)
    adv_count = sum(t.pos_ == "ADV" for t in tokens)
    pronoun_count = sum(t.pos_ == "PRON" for t in tokens)
    det_count = sum(t.pos_ == "DET" for t in tokens)
    adp_count = sum(t.pos_ == "ADP" for t in tokens)
    part_count = sum(t.pos_ == "PART" for t in tokens)
    cconj_count = sum(t.pos_ == "CCONJ" for t in tokens)
    sconj_count = sum(t.pos_ == "SCONJ" for t in tokens)
    num_count = sum(t.pos_ == "NUM" for t in tokens)

    # Fine-grained grammatical tags from spaCy
    modal_count = sum(t.tag_ == "MD" for t in tokens)
    wh_word_count = sum(t.tag_ in {"WDT", "WP", "WP$", "WRB"} for t in tokens)

    # Dependency counts
    negation_count = sum(t.dep_ == "neg" for t in tokens)
    subject_count = sum(t.dep_ in {"nsubj", "nsubjpass", "csubj", "expl"} for t in tokens)
    object_count = sum(t.dep_ in {"dobj", "obj", "iobj", "pobj"} for t in tokens)
    passive_subject_count = sum(t.dep_ == "nsubjpass" for t in tokens)
    root_count = sum(t.dep_ == "ROOT" for t in tokens)
    clause_count = sum(t.dep_ in {"ccomp", "xcomp", "advcl", "relcl", "acl"} for t in tokens)
    conjunction_dep_count = sum(t.dep_ == "conj" for t in tokens)
    prep_dep_count = sum(t.dep_ == "prep" for t in tokens)
    agent_dep_count = sum(t.dep_ == "agent" for t in tokens)

    # Tree / syntactic complexity
    max_dependency_depth = max(depths)
    mean_dependency_depth = float(np.mean(depths))
    avg_children_per_token = float(np.mean([len(list(t.children)) for t in tokens])) if tokens else 0.0

    # Ratios
    lexical_diversity = safe_divide(len(set(lemmas)), len(lemmas))
    content_word_count = sum(t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"} for t in tokens)
    content_word_ratio = safe_divide(content_word_count, len(tokens))
    noun_verb_ratio = safe_divide(noun_count + proper_noun_count, verb_count + aux_count)
    pronoun_ratio = safe_divide(pronoun_count, len(tokens))
    punctuation_ratio = safe_divide(sum(t.is_punct for t in tokens), len(tokens))

    # Entity / sentence structure
    entity_count = len(doc.ents)
    avg_entities_per_sentence = safe_divide(entity_count, sentence_count)

    # Punctuation directly from text
    exclamation_count = text.count("!")
    question_count = text.count("?")
    comma_count = text.count(",")
    semicolon_count = text.count(";")
    colon_count = text.count(":")
    repeated_punctuation_binary = int(("!!" in text) or ("??" in text) or ("?!" in text) or ("!?" in text))

    return {
        "sentence_length_tokens": sentence_length_tokens,
        "sentence_length_chars": sentence_length_chars,
        "avg_token_length": round(avg_token_length, 3),
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 3),

        "noun_count": noun_count,
        "proper_noun_count": proper_noun_count,
        "verb_count": verb_count,
        "aux_count": aux_count,
        "adj_count": adj_count,
        "adv_count": adv_count,
        "pronoun_count": pronoun_count,
        "det_count": det_count,
        "adp_count": adp_count,
        "part_count": part_count,
        "cconj_count": cconj_count,
        "sconj_count": sconj_count,
        "num_count": num_count,

        "modal_count": modal_count,
        "wh_word_count": wh_word_count,

        "negation_count": negation_count,
        "subject_count": subject_count,
        "object_count": object_count,
        "passive_subject_count": passive_subject_count,
        "root_count": root_count,
        "clause_count": clause_count,
        "conjunction_dep_count": conjunction_dep_count,
        "prep_dep_count": prep_dep_count,
        "agent_dep_count": agent_dep_count,

        "max_dependency_depth": max_dependency_depth,
        "mean_dependency_depth": round(mean_dependency_depth, 3),
        "avg_children_per_token": round(avg_children_per_token, 3),

        "lexical_diversity": round(lexical_diversity, 3),
        "content_word_count": content_word_count,
        "content_word_ratio": round(content_word_ratio, 3),
        "noun_verb_ratio": round(noun_verb_ratio, 3),
        "pronoun_ratio": round(pronoun_ratio, 3),
        "punctuation_ratio": round(punctuation_ratio, 3),

        "entity_count": entity_count,
        "avg_entities_per_sentence": round(avg_entities_per_sentence, 3),

        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "comma_count": comma_count,
        "semicolon_count": semicolon_count,
        "colon_count": colon_count,
        "repeated_punctuation_binary": repeated_punctuation_binary,
    }


# Batch processing
texts = df["sentence"].fillna("").astype(str).tolist()
batch_size = 64
total_texts = len(texts)
total_batches = math.ceil(total_texts / batch_size)

print(f"\nTotal texts: {total_texts}")
print(f"Batch size: {batch_size}")
print(f"Total batches: {total_batches}\n")

features = []
overall_start = time.time()

for start_idx in range(0, total_texts, batch_size):
    end_idx = min(start_idx + batch_size, total_texts)
    batch_texts = texts[start_idx:end_idx]
    docs = list(nlp.pipe(batch_texts, batch_size=batch_size))

    for i, doc in enumerate(docs):
        features.append(extract_features(doc, batch_texts[i]))

total_time = time.time() - overall_start
print(f"Feature extraction completed in {total_time:.2f}s")

# Merge and save
features_df = pd.DataFrame(features)
final_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

print("\nFinal dataset preview:")
print(final_df.head())
print("\nDataset shape:", final_df.shape)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("\nProcessed feature dataset saved to:")
print(OUTPUT_PATH)












