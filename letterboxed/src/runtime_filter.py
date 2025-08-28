# src/runtime_filter.py

import pickle
from puzzle import puzzle_to_side_map
from typing import List, Dict, Set, Tuple
from utils import timed

#PREPROCESSED_DATA_PATH = "../data/preprocessed.pkl"
PREPROCESSED_DATA_PATH = "letterboxed/data/preprocessed.pkl"

def load_preprocessed(path=PREPROCESSED_DATA_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_valid_words(puzzle: List[List[str]], pre: dict) -> List[str]:
    """
    Filters and returns a list of valid words based on the current puzzle.
    """
    with timed("Runtime filtering"):
        # Setup
        puzzle_letters = set(letter.lower() for side in puzzle for letter in side)
        side_map = puzzle_to_side_map(puzzle)

        valid_words = []

        for word in pre["words"]:
            # Coarse filter: must only use letters in puzzle
            if not pre["word_to_letter_set"][word].issubset(puzzle_letters):
                continue

            # Fine filter: must not use same-side transitions
            transitions = pre["word_to_transitions"][word]
            if all(side_map[a] != side_map[b] for (a, b) in transitions):
                valid_words.append(word)

        return sorted(valid_words)

# Quick test runner
if __name__ == "__main__":
    import json
    from puzzle import load_puzzle, print_puzzle

    print("Loading puzzle and preprocessed data...")
    puzzle = load_puzzle()
    print_puzzle(puzzle)

    pre = load_preprocessed()
    valid_words = get_valid_words(puzzle, pre)

    print(f"\n{len(valid_words)} valid words found.")
    print("Examples:", valid_words[:10])
