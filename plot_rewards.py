import matplotlib.pyplot as plt
import numpy as np

# --- Load rewards ---
rewards = np.loadtxt("Test_8_4_A_Star_Improved_smaller_rewards.csv")

# --- Parameters ---
window = 100
episodes = len(rewards)
x = np.arange(episodes)

# --- Reward metrics ---
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

# --- Completion/failure detection ---
completions = (rewards > 0).astype(int)
failures = (rewards < -9).astype(int)

completion_rate = np.convolve(completions, np.ones(window)/window, mode='valid')
failure_rate = np.convolve(failures, np.ones(window)/window, mode='valid')

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left Y-axis (rewards)
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward", color='blue')
#ax1.fill_between(x, rewards, alpha=0.2, color='skyblue', label="Episode Reward")
ax1.plot(x[window-1:], moving_avg, label="Moving Avg Reward", color='orange', linewidth=2)
ax1.tick_params(axis='y', labelcolor='blue')

# Right Y-axis (rates)
ax2 = ax1.twinx()
ax2.set_ylabel("Rate", color='green')
ax2.plot(x[window-1:], completion_rate, label="Completion Rate", color='green', linewidth=2)
ax2.plot(x[window-1:], failure_rate, label="Failure Rate", color='red', linewidth=2)

ax2.tick_params(axis='y', labelcolor='green')

# Legends and layout
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.92))
plt.title("Training Progress: Rewards & Completion/Failure Rates")
plt.tight_layout()
plt.grid(True)
plt.show()
