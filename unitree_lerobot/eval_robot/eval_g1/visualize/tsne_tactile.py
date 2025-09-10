from typing import Literal, List, Tuple
import tqdm
import logging
from dataclasses import dataclass, field
from pprint import pformat
from dataclasses import asdict

from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import RobotConfig, ROBOT_CONFIGS

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, Optional
from collections import defaultdict


@dataclass
class VisualizeConfig:
    repo_ids: List[str] = field(default_factory=list)
    arm_type: str = "g1"
    hand_type: str = "inspire"
    tactile_enc_type: Literal["image", "state"] = "image"


def parse_tactile_signals(
    robot_cfg: RobotConfig,
    tactile_enc_type: str,
    step: dict,
    prefixs: List[str] = ["left_tactile", "right_tactile"],
):
    tactile_signals = {}

    if tactile_enc_type == "image":
        for tac_name in robot_cfg.tactiles:
            if any(tac_name.startswith(p) for p in prefixs):
                tactile_img = step.get(f"observation.images.{tac_name}", None)
                if tactile_img is not None:
                    tactile_signal = tactile_img[0, :, :].numpy()  # (H, W) shape, use first channel
                    tactile_signals[tac_name] = tactile_signal

    elif tactile_enc_type == "state":
        total_pixels = sum([
            h * w for tac_name, (c, h, w) in robot_cfg.tactile_to_image_shape.items()
            if any(tac_name.startswith(p) for p in prefixs)
        ])
        empty_data = np.zeros((total_pixels,), dtype=np.float32).reshape(len(prefixs), -1)

        # Reconstruct raw tactile data from state
        start_idx = len(robot_cfg.motors)
        tactile_state = step["observation.state"][start_idx:]
        for i, (tac_name, pixel_indices) in enumerate(robot_cfg.tactile_to_state_indices.items()):
            tactile_data = tactile_state[i].item()
            tac_channel = 0 if tac_name.startswith(prefixs[0]) else 1
            empty_data[tac_channel, pixel_indices] = tactile_data

        # indexing and reshaping into images
        flatten_data = empty_data.flatten()
        idx = 0
        for tac_name, (channel, height, width) in robot_cfg.tactile_to_image_shape.items():
            if any(tac_name.startswith(p) for p in prefixs):
                size = height * width
                tactile_signal = flatten_data[idx:idx + size].reshape((height, width))
                tactile_signals[tac_name] = tactile_signal
                idx += size

    return tactile_signals


def collect_tactile(
    robot_cfg: RobotConfig,
    tactile_enc_type: str,
    dataset: LeRobotDataset,
) -> np.ndarray:
    tactile_signal_dict = defaultdict(list)

    n_step = sum([
        dataset.episode_data_index["to"][i].item() - dataset.episode_data_index["from"][i].item()
        for i in range(dataset.num_episodes)
    ])

    # Iterate through episodes and steps
    pbar = tqdm.tqdm(total=n_step, desc=f"Collecting from {dataset.repo_id}")
    for epi_idx in range(dataset.num_episodes):
        # init pose
        from_idx = dataset.episode_data_index["from"][epi_idx].item()
        to_idx = dataset.episode_data_index["to"][epi_idx].item()

        for step_idx in range(from_idx, to_idx):
            step = dataset[step_idx]
            tactile_signal = parse_tactile_signals(robot_cfg, tactile_enc_type, step)
            for tac_name, tactile_data in tactile_signal.items():
                tactile_signal_dict[tac_name].append(tactile_data)
            pbar.update(1)

    # Convert lists to flattened arrays -> (n_steps, feature_dim)
    tactile_signal_arrays = []
    for i, tac_name in enumerate(sorted(tactile_signal_dict.keys())):
        tactile_array = np.array(tactile_signal_dict[tac_name]).reshape(n_step, -1)
        tactile_signal_arrays.append(tactile_array)
    tactile_signal_array = np.concatenate(tactile_signal_arrays, axis=1)  # (n_steps, total_feature_dim)

    return tactile_signal_array


def plot_tsne_tactile(
    data: Dict[str, np.ndarray],
    perplexity: float = 30.0,
    standardize: bool = True,
    pca_dim: Optional[int] = None,
    random_state: int = 42,
    figsize: Tuple[int, int] = (7, 7),
    alpha: float = 0.8,
    save_path: Optional[str] = "tactile_tsne_by_key.png",
    show: bool = False,
):
    """
    Takes a dict of ndarrays with shape (n_steps, n_features)
    and visualizes the data using t-SNE, separated by keys.
    Each key is assigned a different color.
    """
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError("data must be a non-empty dict, and values must be (n_steps, n_features) ndarrays.")

    names = sorted(list(data.keys()))

    arrays = []
    labels = []
    N_total = 0

    # ----------------------------
    # 1) Merge inputs + Labeling
    # ----------------------------
    for idx, k in enumerate(names):
        if k not in data:
            raise KeyError(f"'{k}' is not in data.")
        v = np.asarray(data[k])
        if v.ndim != 2:
            raise ValueError(f"{k}: must be a 2D array (n_steps, n_features). Current shape={v.shape}")
        arrays.append(v)
        labels.extend([k] * v.shape[0])  # Record labels by key
        N_total += v.shape[0]

    X = np.concatenate(arrays, axis=0)  # (sum(n_steps), total_features)
    labels = np.array(labels)

    # ----------------------------
    # 2) Handle missing values
    # ----------------------------
    all_nan = np.isnan(X).all(axis=0)
    if all_nan.any():
        keep_mask = ~all_nan
        X = X[:, keep_mask]

    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            med = np.nanmedian(col)
            if np.isnan(med):
                med = 0.0
            col[np.isnan(col)] = med
            X[:, j] = col

    # ----------------------------
    # 3) Standardization
    # ----------------------------
    if standardize and X.size > 0:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    # ----------------------------
    # 4) PCA (optional)
    # ----------------------------
    if pca_dim is not None:
        from sklearn.decomposition import PCA
        d = int(min(pca_dim, X.shape[1]))
        if d >= 1:
            pca = PCA(n_components=d, random_state=random_state)
            X = pca.fit_transform(X)

    # ----------------------------
    # 5) t-SNE
    # ----------------------------
    N = X.shape[0]
    if N < 2:
        raise ValueError("At least 2 samples are required to perform t-SNE.")

    max_perp_rule = max(2, (N - 1) // 3)
    max_perp_sklearn = max(2, N - 1)
    safe_perp = min(perplexity, max_perp_rule, max_perp_sklearn - 1)
    if safe_perp < 2:
        safe_perp = 2.0

    tsne = TSNE(
        n_components=2,
        perplexity=float(safe_perp),
        learning_rate="auto",
        init="pca",
        metric="euclidean",
        random_state=random_state,
        verbose=0,
    )
    Y = tsne.fit_transform(X)

    # ----------------------------
    # 6) Visualization: color by key
    # ----------------------------
    plt.figure(figsize=figsize)
    unique_keys = np.unique(labels)
    cmap = plt.get_cmap("tab10")  # Up to 10 colors, can be extended if needed

    for idx, k in enumerate(unique_keys):
        mask = labels == k
        plt.scatter(Y[mask, 0], Y[mask, 1], s=12, alpha=alpha, color=cmap(idx % 10), label=k)

    handles = [matplotlib.patches.Patch(color=cmap(idx % 10), label=k) for idx, k in enumerate(unique_keys)]
    plt.legend(handles=handles, title="Task")
    plt.title(f"Tactile t-SNE Visualization (features={X.shape[1]}, perp={safe_perp:.1f})", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return Y, labels


@parser.wrap()
def visualize_main(cfg: VisualizeConfig):
    logging.info(pformat(asdict(cfg)))

    # Load robot configuration based on arm and hand type
    robot_config = ROBOT_CONFIGS[
        f"Unitree"
        f"_{cfg.arm_type.capitalize()}"
        f"_{cfg.hand_type.capitalize()}"
    ]

    # Collect tactile data
    file_name = f"tactile_signals_{len(cfg.repo_ids)}repo.npz"
    if not os.path.exists(file_name):
        logging.info("Collecting tactile data...")
        total_tactile_signals = {}
        for repo_id in cfg.repo_ids:
            dataset = LeRobotDataset(repo_id=repo_id)
            logging.info(f"Repository ID: {repo_id}")
            tactile_signals = collect_tactile(robot_config, cfg.tactile_enc_type, dataset)
            total_tactile_signals[repo_id] = tactile_signals
        np.savez_compressed(file_name, **total_tactile_signals)
    else:
        logging.info(f"Loading saved tactile data from '{file_name}'...")
        total_tactile_signals = dict(np.load(file_name, allow_pickle=True))

    # Plot t-SNE
    plot_tsne_tactile(
        total_tactile_signals,
        perplexity=30.0,
        standardize=True,
        pca_dim=50,
        random_state=42,
        figsize=(8, 8),
        alpha=0.2,
        save_path=f"tactile_tsne_{len(cfg.repo_ids)}.png",
        show=False,  # optionally display the plot
    )

    logging.info("End of analysis")


if __name__ == "__main__":
    init_logging()
    visualize_main()

"""
Usage:
python visualize_tactile.py \
--repo_ids "[\"eunjuri/towel_strong_train\", \"eunjuri/towel_weak_train\"]" \
--arm_type g1 \
--hand_type inspire \
--tactile_enc_type image
"""
