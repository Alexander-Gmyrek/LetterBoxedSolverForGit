# src/puzzle.py

import random
import json
import string
from typing import List, Dict

#PUZZLE_SAVE_PATH = "../puzzles/sample_puzzles.json"
PUZZLE_SAVE_PATH = "letterboxed/puzzles/sample_puzzles.json"

def generate_puzzle(seed=None) -> List[List[str]]:
    """
    Generate a Letter Boxed puzzle: 4 sides, each with 3 unique letters.
    Returns a list of 4 lists.
    """
    if seed is not None:
        random.seed(seed)

    letters = random.sample(string.ascii_uppercase, 12)
    # ensure we have at least 2 vouwels and if we have a q we have a u
    vowels = set("AEIOU")
    while sum(1 for l in letters if l in vowels) < 2 or ('Q' in letters and 'U' not in letters):
        letters = random.sample(string.ascii_uppercase, 12)
    random.shuffle(letters)

    return [letters[i:i+3] for i in range(0, 12, 3)]

def puzzle_to_side_map(puzzle: List[List[str]]) -> Dict[str, int]:
    """
    Maps each letter to its side index (0-3).
    """
    side_map = {}
    for i, side in enumerate(puzzle):
        for letter in side:
            side_map[letter.lower()] = i  # use lowercase for matching words
    return side_map

def save_puzzle(puzzle: List[List[str]], path=PUZZLE_SAVE_PATH):
    with open(path, "w") as f:
        json.dump({"sides": puzzle}, f, indent=2)

def load_puzzle(path=PUZZLE_SAVE_PATH) -> List[List[str]]:
    with open(path, "r") as f:
        data = json.load(f)
        return data["sides"]

def print_puzzle(puzzle: List[List[str]]):
    print("Letter Boxed Puzzle:")
    for i, side in enumerate(puzzle):
        print(f"Side {i+1}: {' '.join(side)}")

if __name__ == "__main__":
    puzzle = generate_puzzle(seed=42)
    print_puzzle(puzzle)
    save_puzzle(puzzle)
    print("Puzzle saved to", PUZZLE_SAVE_PATH)
