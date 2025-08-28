from time import sleep
import torch
import torch.nn as nn
import torch.optim as optim
import string
import math
import os
import random

from model import WordScoringModel, get_hidden_activations, ValueNet
from environment import LetterBoxedRLEnv
from runtime_filter import get_valid_words
from puzzle import generate_puzzle, puzzle_to_side_map, print_puzzle
from preprocess import load_preprocessed
from debug_utils import EpisodeTracer, get_all_hidden_acts


def one_hot(char):
    vec = [0] * 26
    if char in string.ascii_lowercase:
        vec[ord(char) - ord('a')] = 1
    return vec

def state_vector_from_env(env):
    return [
        -1 if l not in env.puzzle_letters else
         0 if l in env.used_letters else 1
        for l in string.ascii_lowercase
    ]

def build_input_vector(word, state_vector, word_features, side_map, remaining_count):
    feat = word_features[word]
    
    # Word features
    length = [feat["length"] / 15]
    #first = one_hot(feat["first_letter"])
    last = one_hot(feat["last_letter"])
    log_remaining = [remaining_count]
    presence = [1 if x > 0 else 0 for x in feat["letter_freq"]]



    # Side info: each letter gets a 4-dim one-hot based on its side
    side_vec = []
    for side_id in range(4):
        side_letters = [
            1 if side_map.get(chr(ord('A') + i), -1) == side_id else 0
            for i in range(26)
        ]
        side_vec.extend(side_letters)

    return torch.tensor(length + last + presence + log_remaining + state_vector + side_vec, dtype=torch.float32)


def train_loop(episodes=10, lr=1e-3, test_name="default_test", debug_every=1000, done=True):
    global os
    from collections import deque
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_net = ValueNet().to(device)
    opt_v = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    replay_buffer = deque(maxlen=100000)  # store (state_vec, return) pairs

    gamma = 0.99
    output_dir = os.path.join("activations", test_name)
    os.makedirs(output_dir, exist_ok=True)
    pre = load_preprocessed()
    ckpt_dir   = os.path.join("checkpoints", test_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_avg   = -float("inf")
    window     = episodes // 10

    
    model = WordScoringModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_total = -float("inf")
    reward_history = []
    completions = 0
    for ep in range(1, episodes + 1):
        puzzle = generate_puzzle()
        side_map = puzzle_to_side_map(puzzle)
        valid_words = get_valid_words(puzzle, pre)
        env = LetterBoxedRLEnv(puzzle, valid_words)
        word_features = pre["word_features"]

        total_reward = 0
        steps = []
        episode_history = []
        tracer = None
        if debug_every and ep % debug_every == 0:
            tracer = EpisodeTracer(test_name)

        step_idx = 0

        while not env.is_done() or not done:
            state_vec = state_vector_from_env(env)
            valid_words_now = env.get_possible_words()

            
            if tracer:
                acts_by_layer = get_all_hidden_acts(model, X)
                tracer.log_step(step_idx, valid_words_now, scores, acts_by_layer)
            step_idx += 1
            if not valid_words_now:
                #print(f"No valid words left! Ending episode {ep}.")
                total_reward = total_reward - 25
                rem_ratio = 0.0
                chosen_word = ""
                extra_feats = [rem_ratio, 0.0]
                full_state_vec = state_vec + extra_feats
                episode_history.append((torch.tensor(full_state_vec, dtype=torch.float32), -25))
                break

            inputs = []
            for w in valid_words_now:
                vec = build_input_vector(w, state_vec, word_features, side_map, remaining_count=((math.log1p(env.count_remaining_after_word(w))/math.log1p(len(valid_words)))))
                inputs.append(vec)

            X = torch.stack(inputs).to(device)

            best_f, best_w = -float("inf"), None
            for idx, (w, x_vec) in enumerate(zip(valid_words_now, inputs)):
                # immediate reward
                r0 = env.estimate_reward(w)

                # simulate
                backup = env.clone_state()
                env.submit(w)
                next_sv = torch.tensor(state_vector_from_env(env) + [math.log1p(len(env.get_possible_words()))/math.log1p(len(valid_words))] + [len(w)], dtype=torch.float32).to(device)
                v_next = value_net(next_sv).item()
                env.restore_state(backup)

                f = r0 + gamma * v_next
                if f > best_f:
                    best_f, best_w, best_idx = f, w, idx

            chosen_word = best_w


            next_state, reward, done = env.submit(chosen_word)


            rem_ratio = 0.0
            if valid_words:
                possible_now = len(env.get_possible_words())
                rem_ratio = math.log1p(possible_now) / math.log1p(len(valid_words))

            extra_feats = [rem_ratio, len(chosen_word)/15]
            full_state_vec = state_vec + extra_feats
            episode_history.append((torch.tensor(full_state_vec, dtype=torch.float32), reward))
            total_reward += reward
            steps.append((X[best_idx], reward))
            if tracer:
                tracer.finalize()
        if done:
            completions += 1
        reward_history.append(total_reward)

        if len(reward_history) >= window:
            # compute running average over last `window` episodes
            running_avg = sum(reward_history[-window:]) / window

            # scale threshold (optional)—for example, require +1% improvement:
            threshold = 1.01  # no extra scaling for now
            if running_avg > best_avg * threshold:
                best_avg = running_avg

                # save checkpoint
                torch.save({
                    "episode":        ep,
                    "running_avg":    running_avg,
                    "model_state":    model.state_dict(),
                    "optimizer_state":optimizer.state_dict(),
                    "value_state":    value_net.state_dict(),
                }, os.path.join(ckpt_dir, f"best_avg_{window}ep.pt"))

                print(f"💾 Saved new best running‐avg {running_avg:.3f} at ep {ep}")
        
        if total_reward > best_total:
            best_total = total_reward
            print(f"⭐ New best (Ep {ep}) R={total_reward:.2f} Words: {env.used_words} \n State: {next_state}")
        elif ep % 25 == 0:
            avg_reward = sum(reward_history[-25:]) / 25
            print(f"📈 Episode {ep}: Avg reward (last 25): {avg_reward:.2f}")
        if ep % 100 == 0:
            print(f"Completion rate {completions/100}")
            completions = 0
        if ep % 500 == 0:
            print(f"\n🧠 EPISODE {ep} TRACE:")
            print_puzzle(puzzle)
            for step_num, (X_batch, reward) in enumerate(steps):
                scores = model(X_batch)
                if scores.dim() == 0 or len(scores) < 2:
                    continue  # Skip logging when there's only one or zero valid words

                topk = torch.topk(scores, k=min(5, len(scores)))
                print(f" Step {step_num+1}:")
                for idx in topk.indices:
                    w = valid_words_now[idx]
                    s = scores[idx].item()
                    marker = "✅" if w == chosen_word else "  "
                    print(f"   {marker} {w:<10} -> Score: {s:.2f}")
            print(f"Final reward: {total_reward:.2f}, Words: {env.used_words}")
        if ep % 1000 == 0:
            with torch.no_grad():
                debug_X = X.clone()
                debug_words = valid_words_now[:]
                scores = model(debug_X)

                if scores.ndim == 0:
                    scores = scores.unsqueeze(0)

                topk = torch.topk(scores, k=min(10, len(scores)))
                selected = topk.indices.cpu().numpy()

                # Filter out-of-bounds indices
                selected = [i for i in selected if i < len(debug_words)]

                if selected:  # Only proceed if there's something to visualize
                    word_samples = [debug_words[i] for i in selected]
                    activations = get_hidden_activations(model, debug_X[selected])

                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    # Plot each layer separately
                    for i, layer_activations in enumerate(activations, start=1):
                        plt.figure(figsize=(10, 6))
                        sns.heatmap(
                            layer_activations.cpu().numpy(),
                            annot=False,
                            cmap="viridis",
                            cbar=True,
                            xticklabels=False,
                            yticklabels=word_samples
                        )
                        plt.title(f"Hidden Layer {i} Activations (Ep {ep})")
                        plt.savefig(os.path.join(output_dir, f"ep{ep}_heat_layer{i}.png"))
                        plt.close()

                else:
                    print(f"[Ep {ep}] Skipping heatmap: no valid selected words.")
        # Update value network
        R = 0
        for state_vec, reward in reversed(episode_history):
            R = reward + gamma * R
            replay_buffer.append((state_vec, R))
        for i, (s, r) in enumerate(replay_buffer):
            if not isinstance(s, torch.Tensor):
                print(f"Non-tensor at index {i}: {type(s)}")

        if len(replay_buffer) >= 256:
            batch = random.sample(replay_buffer, k=256)
            states, returns = zip(*batch)  # states is a tuple of Tensors

            # Stack first, *then* move to GPU/CPU
            S = torch.stack([torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s for s in states]).to(device)

            # Returns is a tuple of floats
            T = torch.tensor(returns, dtype=torch.float32, device=device)

            # Forward + backward
            pred_v = value_net(S)
            loss_v = loss_fn(pred_v, T)

            opt_v.zero_grad()
            loss_v.backward()
            opt_v.step()

        for vec, reward in steps:
            pred = model(vec.unsqueeze(0))
            target = torch.tensor([reward], dtype=torch.float32).to(device)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with open(f"{test_name}_rewards.csv", "w") as f:
        for r in reward_history:
            f.write(f"{r}\n")
    print("Training complete — saving final model.")
    final_ckpt = os.path.join(ckpt_dir, "final_model.pt")
    torch.save({
        "episode": episodes,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "value_net_state": value_net.state_dict(),
        "best_total": best_total,
    }, final_ckpt)


if __name__ == "__main__":
    test_name = "Test_8_4_A_Star_Improved_smaller"
    train_loop(episodes=5000, lr=1e-3, test_name=test_name)