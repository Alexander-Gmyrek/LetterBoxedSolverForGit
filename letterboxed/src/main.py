# src/test_puzzle.py

import os
import torch
import math
import json
import string
import random
import pickle

from model       import WordScoringModel, ValueNet
from environment import LetterBoxedRLEnv
from puzzle      import load_puzzle, print_puzzle, puzzle_to_side_map
from runtime_filter import get_valid_words


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA  = 0.99


def load_checkpoint(path, device=DEVICE):
    """Load policy + value net from a .pt checkpoint."""
    ckpt = torch.load(path, map_location=device)
    policy = WordScoringModel().to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    value = ValueNet(state_dim=28).to(device)
    value.load_state_dict(ckpt["value_state"])
    value.eval()

    return policy, value


def make_state_vector(env):
    """–1 if letter not in puzzle, 0 if used, 1 if unused."""
    return torch.tensor([
        -1 if c not in env.puzzle_letters else
         0 if c in env.used_letters      else
         1
        for c in string.ascii_lowercase
    ], dtype=torch.float32, device=DEVICE)


def solve_with_lookahead(puzzle, policy, value_net, pre):
    """Run one episode on the given puzzle with one‐step lookahead."""
    device = DEVICE
    policy.to(device)
    side_map   = puzzle_to_side_map(puzzle)
    valid_words = get_valid_words(puzzle, pre)
    env = LetterBoxedRLEnv(puzzle, valid_words)

    solution = []
    print_puzzle(puzzle)
    print()

    while not env.is_done():
        candidates = env.get_possible_words()
        if not candidates:
            print("⛔ Stuck: no valid moves!")
            break

        best_f, best_w = -1e9, None

        best_f, best_w = -float("inf"), None

        for w in candidates:
            # Estimate immediate reward
            r0 = env.estimate_reward(w)

            # Simulate move
            snap = env.clone_state()
            env.submit(w)

            # Compute extra features
            possible_now = len(env.get_possible_words())
            total_valid = len(valid_words)
            rem_ratio = math.log1p(possible_now) / math.log1p(total_valid) if total_valid > 1 else 0.0
            normalized_len = len(w) / 15.0

            # Build full state vector
            sv_raw = env.get_state_vector()
            full_state_vec = sv_raw + [rem_ratio, normalized_len]
            sv_tensor = torch.tensor(full_state_vec, dtype=torch.float32).to(device)

            # Predict future value
            v_next = value_net(sv_tensor).item()
            env.restore_state(snap)

            # Rank move
            f = r0 + GAMMA * v_next
            if f > best_f:
                best_f, best_w = f, w

        if best_w is None:
            print("⛔ Couldn’t select a word!")
            break

        print(f"→ {best_w:10s}   score=f({best_f:.2f})")
        env.submit(best_w)
        solution.append(best_w)

    print("\n✅ Final solution:", solution)
    print("   Letters covered:", "".join(sorted(env.used_letters)))
    return solution


def main():
    # 1) choose puzzle
    PUZZLE_PATH = "letterboxed/puzzles/sample_puzzles.json"
    puzzle = load_puzzle(PUZZLE_PATH)

    # 2) load preprocessing (word list + features)
    with open("letterboxed/data/preprocessed.pkl", "rb") as f:
        pre = pickle.load(f)

    # 3) load your best checkpoint
    CKPT = "checkpoints/Test_8_3_A_Star_Improved_smaller/best_avg_125ep.pt"
    policy, value_net = load_checkpoint(CKPT)

    

    # 4) solve!
    solve_with_lookahead(puzzle, policy, value_net, pre)


if __name__ == "__main__":
    main()
