import os
import imageio
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def get_all_hidden_acts(model, x):
    """
    Run x through model.net and collect the activation
    immediately after every ReLU layer.
    Returns: list of Tensors [batch_size×num_neurons, …]
    """
    acts = []
    z = x
    with torch.no_grad():
        for layer in model.net:
            z = layer(z)
            if isinstance(layer, torch.nn.ReLU):
                acts.append(z.detach().cpu())
    return acts

class EpisodeTracer:
    """
    For one episode, logs:
      - step_i_layer_j.png  (heatmap for layer j at step i)
      - episode.gif         (animation of layer 1 across steps)
    """
    def __init__(self, test_name="run", save_root="activations", make_gif=True):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(save_root, f"{test_name}extra/{test_name}_{ts}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.make_gif = make_gif
        self.frame_paths = []

    def log_step(self, step_idx, words, scores, acts_by_layer):
        """
        Save a composite heatmap of all hidden layers for step `step_idx`, and
        record one frame per step for the first layer to build the GIF.
        """
        n = len(words)
        m = scores.numel()

        # 1) If there are no candidate words or no scores, log a placeholder
        if n == 0 or m == 0:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "NO VALID WORDS", ha="center", va="center")
            ax.axis("off")
            fname = f"step{step_idx:02d}_empty.png"
            fpath = os.path.join(self.out_dir, fname)
            fig.savefig(fpath); plt.close(fig)
            if self.make_gif:
                self.frame_paths.append(fpath)
            return

        # 2) Determine k safely from both words and scores
        k = min(10, n, m)
        if k <= 0:
            # nothing to pick
            return

        # 3) Grab top-k indices
        topk = torch.topk(scores, k=k)
        raw = topk.indices.cpu().tolist()
        if isinstance(raw, int):
            top_idx = [raw]
        else:
            top_idx = raw

        # 4) Sanity-filter indices
        top_idx = [i for i in top_idx if 0 <= i < n]

        # 5) Selected words
        sel_words = [words[i] for i in top_idx]

        # 6) Build composite heatmap of all layers
        fig, axs = plt.subplots(1, len(acts_by_layer), figsize=(4*len(acts_by_layer), 4))
        for li, ax in enumerate(axs):
            layer_act = acts_by_layer[li][top_idx]  # (k × neurons)
            im = ax.imshow(layer_act.numpy(), aspect="auto", cmap="viridis")
            ax.set_title(f"L{li+1}")
            ax.set_yticks(range(len(sel_words)))
            ax.set_yticklabels(sel_words)
            ax.set_xticks([])
        fig.suptitle(f"Step {step_idx}")
        fname = f"step{step_idx:02d}_all_layers.png"
        fpath = os.path.join(self.out_dir, fname)
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(fpath)
        plt.close(fig)

        # 7) Add to GIF frames (use the first layer's image)
        if self.make_gif:
            self.frame_paths.append(fpath)



    def finalize(self):
        # build episode.gif from collected frame_paths
        if self.make_gif and self.frame_paths:
            gif_path = os.path.join(self.out_dir, "episode.gif")
            frames = [imageio.imread(p) for p in sorted(self.frame_paths)]
            imageio.mimsave(gif_path, frames, duration=0.8)
            print(f"🎞  Saved episode animation → {gif_path}")
