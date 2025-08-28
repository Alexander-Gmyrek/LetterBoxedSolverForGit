# src/test_filter_speed.py

from runtime_filter import get_valid_words as filter_letter_to_word
from preprocess import load_preprocessed
from puzzle import load_puzzle, print_puzzle, puzzle_to_side_map
from utils import timed

import pickle
import time

# Reference version from word-to-letter_set
def get_valid_words_set_based(puzzle, pre):
    puzzle_letters = set(letter.lower() for side in puzzle for letter in side)
    side_map = puzzle_to_side_map(puzzle)
    valid_words = []

    for word in pre["words"]:
        if not pre["word_to_letter_set"][word].issubset(puzzle_letters):
            continue
        transitions = pre["word_to_transitions"][word]
        if all(side_map[a] != side_map[b] for (a, b) in transitions):
            valid_words.append(word)

    return sorted(valid_words)

def run_speed_test():
    print("Loading puzzle and preprocessed data...")
    puzzle = load_puzzle()
    print_puzzle(puzzle)

    pre = load_preprocessed()

    print("\n--- Benchmarking ---")

    # Approach 1: word-to-letter-set (reference)
    start = time.perf_counter()
    result1 = get_valid_words_set_based(puzzle, pre)
    t1 = time.perf_counter() - start
    print(f"Set-based approach took {t1:.4f} seconds")

    # Approach 2: letter-to-words (index-based)
    start = time.perf_counter()
    result2 = filter_letter_to_word(puzzle, pre)
    t2 = time.perf_counter() - start
    print(f"Letter-index approach took {t2:.4f} seconds")

    # Sanity check
    if set(result1) != set(result2):
        print("\n Mismatch in results!")
        print("Words only in set-based:", set(result1) - set(result2))
        print("Words only in letter-index:", set(result2) - set(result1))
    else:
        print("\n Both approaches returned the same words.")

    print(f"\n Speedup: {t1/t2:.2f}x faster" if t2 < t1 else f"\n Slower by: {t2/t1:.2f}x")

if __name__ == "__main__":
    run_speed_test()
