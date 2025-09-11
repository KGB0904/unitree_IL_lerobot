from typing import List
import numpy as np

from unitree_lerobot.utils.constants import RobotConfig


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