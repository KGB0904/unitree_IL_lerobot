''''
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/common/robot_devices/control_utils.py
'''

import json
import logging
import shutil
import time
from collections import OrderedDict
from contextlib import nullcontext
from copy import copy
from dataclasses import asdict
from multiprocessing import Array, Lock
from pathlib import Path
from pprint import pformat

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn

from cilab_unitree_lerobot.lerobot.lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_arm import G1_29_ArmController
from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_unitree import Dex3_1_Controller, Gripper_Controller
#from unitree_lerobot.eval_robot.eval_g1.robot_control.robot_hand_inspire import Inspire_Controller
from unitree_lerobot.eval_robot.eval_g1.eval_real_config import EvalRealConfig

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


# copy from lerobot.common.robot_devices.control_utils import predict_action
def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            # if "images" in name:
            #     observation[name] = observation[name].type(torch.float32) / 255
            #     observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def get_motor_slices(eval_config) -> OrderedDict:
    arm_type = eval_config.get("arm_type")
    hand_type = eval_config.get("hand_type")

    if arm_type == "g1":
        if hand_type == "inspire":
            return OrderedDict(
                [
                    ("left_arm", slice(0, 7)),
                    ("left_hand", slice(7, 13)),
                    ("right_arm", slice(13, 20)),
                    ("right_hand", slice(20, 26)),
                ]
            )
        if hand_type == "dex3":
            return OrderedDict(
                [
                    ("left_arm", slice(0, 7)),
                    ("right_arm", slice(7, 14)),
                    ("left_hand", slice(14, 21)),
                    ("right_hand", slice(21, 28)),
                ]
            )
        if hand_type == "gripper":
            return OrderedDict(
                [
                    ("left_arm", slice(0, 7)),
                    ("right_arm", slice(7, 14)),
                    ("left_hand", slice(14, 15)),
                    ("right_hand", slice(15, 16)),
                ]
            )

    raise NotImplementedError(
        f"Unsupported motor slice configuration for arm_type={arm_type}, hand_type={hand_type}"
    )


def slice_array(array: np.ndarray, slc: slice) -> np.ndarray:
    start = slc.start or 0
    stop = slc.stop if slc.stop is not None else len(array)
    if start >= len(array):
        return array[0:0]
    stop = min(stop, len(array))
    return array[start:stop]


def tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def eval_policy(
    policy: torch.nn.Module,
    dataset: LeRobotDataset,
):
    
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()

    # send_real_robot: If you want to read observations from the dataset and send them to the real robot, set this to True.  
    # (This helps verify whether the model has generalization ability to the environment, as there are inevitably differences between the real environment and the training data environment.)
    eval_config = {
        'arm_type': 'g1',
        'hand_type': "inspire",
        'tactile_enc_type': "image",
        'send_real_robot': False,
    }

    robot_config = ROBOT_CONFIGS[
        f"Unitree"
        f"_{eval_config['arm_type'].capitalize()}"
        f"_{eval_config['hand_type'].capitalize()}"
    ]

    # init pose
    from_idx = dataset.episode_data_index["from"][0].item()
    step = dataset[from_idx]
    to_idx = dataset.episode_data_index["to"][0].item()

    camera_names = robot_config.cameras
    tactile_names = getattr(robot_config, "tactiles", [])
    ground_truth_actions = []
    predicted_actions = []
    samples = []

    motor_slices = get_motor_slices(eval_config)
    frequency = 50.0

    image_key_to_color = {v: k for k, v in robot_config.camera_to_image_key.items()}

    output_root = Path("eval_outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    episode_idx = 0
    episode_dir = output_root / f"episode_{episode_idx:04d}"
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)

    colors_dir = episode_dir / "colors"
    colors_dir.mkdir(parents=True, exist_ok=True)

    tactiles_dir = episode_dir / "tactiles" if tactile_names else None
    if tactiles_dir is not None:
        tactiles_dir.mkdir(parents=True, exist_ok=True)

    has_carpet_tactiles = any(name.startswith("carpet") for name in tactile_names)
    carpet_dir = episode_dir / "carpet_tactiles" if has_carpet_tactiles else None
    if carpet_dir is not None:
        carpet_dir.mkdir(parents=True, exist_ok=True)

    for cam_name in camera_names:
        color_key = image_key_to_color.get(cam_name, cam_name)
        (colors_dir / color_key).mkdir(parents=True, exist_ok=True)

    if tactiles_dir is not None:
        for tactile_name in tactile_names:
            if tactile_name.startswith("carpet"):
                continue
            (tactiles_dir / tactile_name).mkdir(parents=True, exist_ok=True)

    if eval_config['send_real_robot']:
        # arm
        arm_ctrl = G1_29_ArmController()
        init_left_arm_pose = step['observation.state'][:14].cpu().numpy()

        # hand
        if eval_config['hand_type'] == "dex3":
            left_hand_array = Array('d', 7, lock = True)          # [input]
            right_hand_array = Array('d', 7, lock = True)         # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)  # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False) # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
            init_left_hand_pose = step['observation.state'][14:21].cpu().numpy()
            init_right_hand_pose = step['observation.state'][21:].cpu().numpy()

        elif eval_config['hand_type'] == "gripper":
            left_hand_array = Array('d', 1, lock=True)             # [input]
            right_hand_array = Array('d', 1, lock=True)            # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Gripper_Controller(left_hand_array, right_hand_array, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)
            init_left_hand_pose = step['observation.state'][14].cpu().numpy()
            init_right_hand_pose = step['observation.state'][15].cpu().numpy()
        elif eval_config['hand_type'] == "inspire":
            left_hand_array = Array('d', 6, lock = True)      # [input]
            right_hand_array = Array('d', 6, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
            init_left_hand_pose = step['observation.state'][14:20].cpu().numpy()
            init_right_hand_pose = step['observation.state'][20:].cpu().numpy()
        else:
            pass

    #===============init robot=====================
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == 's':

        if eval_config['send_real_robot']:
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            print("init robot pose")
            arm_ctrl.ctrl_dual_arm(init_left_arm_pose, np.zeros(14))
            left_hand_array[:] = init_left_hand_pose
            right_hand_array[:] = init_right_hand_pose

            print("wait robot to pose")
            time.sleep(1)

        for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
            step = dataset[step_idx]
            observation = {}

            for cam_name in camera_names:
                observation[f"observation.images.{cam_name}"] = step[f"observation.images.{cam_name}"]
            observation["observation.state"] = step["observation.state"]
            for tac_name in tactile_names:
                if eval_config["tactile_enc_type"] == "image":
                    observation[f"observation.images.{tac_name}"] = step[f"observation.images.{tac_name}"]
                elif eval_config["tactile_enc_type"] == "state":
                    observation["observation.state"] = torch.cat(
                        [
                            observation["observation.state"],
                            step[f"observation.state.{tac_name}"].unsqueeze(0),
                        ],
                        dim=0,
                    )

            action = predict_action(
                observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
            )

            action = action.cpu().numpy()

            ground_truth_action = tensor_to_numpy(step["action"])
            state_vector = tensor_to_numpy(step["observation.state"])

            ground_truth_actions.append(ground_truth_action)
            predicted_actions.append(action)

            frame_idx = step_idx - from_idx
            sample = {
                "timestamp": float(frame_idx) / frequency,
                "states": {},
                "actions": {},
                "predicted_actions": {},
                "colors": {},
                "tactiles": {},
                "carpet_tactiles": {},
            }

            for part_name, part_slice in motor_slices.items():
                sliced_state = slice_array(state_vector, part_slice)
                sliced_gt = slice_array(ground_truth_action, part_slice)
                sliced_pred = slice_array(action, part_slice)
                sample["states"][part_name] = {"qpos": sliced_state.tolist()}
                sample["actions"][part_name] = {"qpos": sliced_gt.tolist()}
                sample["predicted_actions"][part_name] = {"qpos": sliced_pred.tolist()}

            for cam_name in camera_names:
                obs_key = f"observation.images.{cam_name}"
                if obs_key not in step:
                    continue

                img_tensor = step[obs_key]
                image_np = tensor_to_numpy(img_tensor)
                if image_np.ndim == 3 and image_np.shape[0] in {1, 3, 4}:
                    image_np = np.transpose(image_np, (1, 2, 0))
                if image_np.ndim == 2:
                    image_np = image_np[:, :, None]
                if image_np.dtype != np.uint8:
                    image_np = np.clip(image_np, 0.0, 1.0)
                    image_np = (image_np * 255).astype(np.uint8)
                if image_np.shape[-1] == 4:
                    image_np = image_np[..., :3]
                if image_np.shape[-1] == 1:
                    image_np = np.repeat(image_np, 3, axis=-1)
                bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) if image_np.shape[-1] == 3 else image_np
                color_key = image_key_to_color.get(cam_name, cam_name)
                image_filename = f"{frame_idx:06d}.png"
                image_path = colors_dir / color_key / image_filename
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), bgr_image)
                sample["colors"][color_key] = (Path("colors") / color_key / image_filename).as_posix()

            if eval_config["tactile_enc_type"] == "image":
                for tactile_name in tactile_names:
                    obs_key = f"observation.images.{tactile_name}"
                    if obs_key not in step:
                        continue

                    tactile_tensor = step[obs_key]
                    tactile_np = tensor_to_numpy(tactile_tensor)
                    if tactile_np.ndim == 3 and tactile_np.shape[0] in {1, 3, 4}:
                        tactile_np = np.transpose(tactile_np, (1, 2, 0))
                    tactile_np = np.asarray(tactile_np)

                    if tactile_name.startswith("carpet"):
                        if carpet_dir is None:
                            continue
                        tactile_filename = f"{tactile_name}_{frame_idx:06d}.npy"
                        tactile_path = carpet_dir / tactile_filename
                        np.save(tactile_path, tactile_np)
                        sample["carpet_tactiles"][tactile_name] = (
                            Path("carpet_tactiles") / tactile_filename
                        ).as_posix()
                    else:
                        if tactiles_dir is None:
                            continue
                        tactile_filename = f"{frame_idx:06d}.npy"
                        tactile_path = tactiles_dir / tactile_name / tactile_filename
                        tactile_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(tactile_path, tactile_np)
                        sample["tactiles"][tactile_name] = (
                            Path("tactiles") / tactile_name / tactile_filename
                        ).as_posix()
            elif eval_config["tactile_enc_type"] == "state":
                for tactile_name in tactile_names:
                    state_key = f"observation.state.{tactile_name}"
                    if state_key not in step:
                        continue

                    tactile_state = tensor_to_numpy(step[state_key])
                    sample["states"][tactile_name] = {"qpos": tactile_state.tolist()}
            else:
                raise NotImplementedError(
                    f"Unsupported tactile encoding: {eval_config['tactile_enc_type']}"
                )

            samples.append(sample)

            if eval_config['send_real_robot']:
                # exec action
                arm_ctrl.ctrl_dual_arm(action[:14], np.zeros(14))
                if eval_config['hand_type'] == "dex3":
                    left_hand_array[:] = action[14:21]
                    right_hand_array[:] = action[21:]
                elif eval_config['hand_type'] == "gripper":
                    left_hand_array[:] = action[14]
                    right_hand_array[:] = action[15]
            
                time.sleep(1/frequency)

        ground_truth_actions = np.array(ground_truth_actions)
        predicted_actions = np.array(predicted_actions)

        episode_dict = {
            "text": {"goal": getattr(dataset, "meta", {}).get("task", "")},
            "metadata": {"frequency": frequency},
            "data": samples,
        }

        with (episode_dir / "data.json").open("w", encoding="utf-8") as json_file:
            json.dump(episode_dict, json_file, indent=2)

        # Get the number of timesteps and action dimensions
        n_timesteps, n_dims = ground_truth_actions.shape

        # Create a figure with subplots for each action dimension
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4*n_dims), sharex=True)
        fig.suptitle('Ground Truth vs Predicted Actions')

        # Plot each dimension
        for i in range(n_dims):
            ax = axes[i] if n_dims > 1 else axes

            ax.plot(ground_truth_actions[:, i], label='Ground Truth', color='blue')
            ax.plot(predicted_actions[:, i], label='Predicted', color='red', linestyle='--')
            ax.set_ylabel(f'Dim {i+1}')
            ax.legend()

        # Set common x-label
        axes[-1].set_xlabel('Timestep')

        plt.tight_layout()
        # plt.show()

        time.sleep(1)
        plt.savefig('figure.png')
        plt.close(fig)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id = cfg.repo_id)

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(policy, dataset)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
