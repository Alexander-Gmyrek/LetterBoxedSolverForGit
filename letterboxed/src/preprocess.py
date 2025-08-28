# src/preprocess.py

import os
import string
import pickle
from collections import defaultdict

#RAW_WORDS_PATH = "../data/en_US.dic"
RAW_WORDS_PATH = "letterboxed/data/en_US.dic"
#CLEANED_WORDS_PATH = "../data/words_cleaned.txt"
CLEANED_WORDS_PATH = "letterboxed/data/words_cleaned.txt"
#OUTPUT_PICKLE_PATH = "../data/preprocessed.pkl"
OUTPUT_PICKLE_PATH = "letterboxed/data/preprocessed.pkl"

def load_raw_words(path):
    words = []
    with open(path, "r") as file:
        for line in file:
            word = line.strip().split("/")[0]  # Remove any trailing annotations
            word = word.lower()
            if word.isalpha():  # Only alphabetic
                words.append(word)
    return words


def is_valid_word(word):
    if len(word) < 3:
        return False
    if not all(c in string.ascii_lowercase for c in word):
        return False
    for i in range(len(word) - 1):
        if word[i] == word[i+1]:
            return False
    return True

def clean_words(words):
    return [word for word in words if is_valid_word(word)]

def build_letter_to_words(words):
    letter_to_words = defaultdict(set)
    for word in words:
        for letter in set(word):
            letter_to_words[letter].add(word)
    return letter_to_words

def build_word_to_letter_set(words):
    return {word: set(word) for word in words}

def build_word_to_transitions(words):
    transitions = {}
    for word in words:
        pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
        transitions[word] = pairs
    return transitions

def save_cleaned_words(words, path):
    with open(path, "w") as f:
        for word in words:
            f.write(word + "\n")

def save_preprocessed_data(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_preprocessed(path=OUTPUT_PICKLE_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)
    
import string

def build_word_features(words):
    word_features = {}
    for word in words:
        freq = [0] * 26
        for c in word:
            freq[ord(c) - ord('a')] += 1
        word_features[word] = {
            "length": len(word),
            "first_letter": word[0],
            "last_letter": word[-1],
            "letter_freq": freq
        }
    return word_features


def main():
    print("Loading raw words...")
    raw_words = load_raw_words(RAW_WORDS_PATH)

    print("Cleaning word list...")
    cleaned_words = clean_words(raw_words)
    save_cleaned_words(cleaned_words, CLEANED_WORDS_PATH)

    print("Building indexes...")
    letter_to_words = build_letter_to_words(cleaned_words)
    word_to_letter_set = build_word_to_letter_set(cleaned_words)
    word_to_transitions = build_word_to_transitions(cleaned_words)

    print("Building features...")
    word_features = build_word_features(cleaned_words)

    print("Saving preprocessed data...")
    save_preprocessed_data({
        "words": cleaned_words,
        "letter_to_words": letter_to_words,
        "word_to_letter_set": word_to_letter_set,
        "word_to_transitions": word_to_transitions,
        "word_features": word_features,
    }, OUTPUT_PICKLE_PATH)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
