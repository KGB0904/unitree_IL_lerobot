from typing import Literal, List, Optional
import logging
from dataclasses import dataclass, field
from pprint import pformat
from dataclasses import asdict

from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import RobotConfig, ROBOT_CONFIGS
from unitree_lerobot.eval_robot.eval_g1.visualize.utils import parse_tactile_signals

import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt


@dataclass
class VisualizeConfig:
    repo_ids: List[str] = field(default_factory=list)
    arm_type: str = "g1"
    hand_type: str = "inspire"
    tactile_enc_type: Literal["image", "state"] = "image"


def collect_info(
    robot_cfg: RobotConfig,
    tactile_enc_type: str,
    dataset: LeRobotDataset,
) -> dict:
    info_dict = {
        "repo_id": dataset.repo_id,
        "total_steps": 0,
        "episode_length": [],
    }

    for epi_idx in range(dataset.num_episodes):
        from_idx = dataset.episode_data_index["from"][epi_idx].item()
        to_idx = dataset.episode_data_index["to"][epi_idx].item()

        # episode info
        episode_length = to_idx - from_idx
        info_dict["total_steps"] += episode_length
        info_dict["episode_length"].append(episode_length)

        # tactile info
        for step_idx in range(from_idx, to_idx):
            if epi_idx == 0 and step_idx == from_idx:
                step = dataset[from_idx]
                tactile_signal = parse_tactile_signals(robot_cfg, tactile_enc_type, step)
                info_dict[f"tactile_names"] = list(tactile_signal.keys())

    return info_dict


def plot_histograms(
    infos: List[dict],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    x-axis: episode length
    y-axis: count
    """
    plt.figure(figsize=(10, 6))

    all_lengths = []
    labels = []
    for info in infos:
        lengths = info["episode_length"]
        all_lengths.append(lengths)
        labels.append(info["repo_id"])

    plt.hist(all_lengths, bins=30, label=labels, alpha=0.7)
    plt.xlabel("Episode Length")
    plt.ylabel("Count")
    plt.title("Histogram of Episode Lengths")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        logging.info(f"Saved histogram to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


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
    logging.info("Collecting info...")
    infos = []
    for repo_id in cfg.repo_ids:
        dataset = LeRobotDataset(repo_id=repo_id)
        info = collect_info(robot_config, cfg.tactile_enc_type, dataset)
        infos.append(info)
        print(info)

    # Plot histograms for episode lengths
    plot_histograms(
        infos,
        save_path=f"hist_episode_{len(cfg.repo_ids)}repo.png",
        show=False,  # optionally display the plot
    )

    logging.info("End of analysis")


if __name__ == "__main__":
    init_logging()
    visualize_main()

"""
Usage:
python histogram_tactile.py \
--repo_ids "[\"eunjuri/towel_strong_train\", \"eunjuri/towel_weak_train\"]" \
--arm_type g1 \
--hand_type inspire \
--tactile_enc_type image
"""
