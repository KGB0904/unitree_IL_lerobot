
import dataclasses
from collections import OrderedDict
from typing import List, Dict

@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: List[str]
    cameras: List[str]
    camera_to_image_key:Dict[str, str]
    json_state_data_name: List[str]
    json_action_data_name: List[str]


@dataclasses.dataclass(frozen=True)
class TactileRobotConfig(RobotConfig):
    tactiles: List[str]
    tactile_to_image_shape: Dict[str, tuple]  # Maps tactile names to their image shapes (height, width)


Z1_CONFIG = RobotConfig(
    motors=[
        "kLeftWaist",
        "kLeftShoulder",
        "kLeftElbow",
        "kLeftForearmRoll",
        "kLeftWristAngle",
        "kLeftWristRotate",
        "kLeftGripper",
        "kRightWaist",
        "kRightShoulder",
        "kRightElbow",
        "kRightForearmRoll",
        "kRightWristAngle",
        "kRightWristRotate",
        "kRightGripper",
    ],
    cameras=[
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_high', 'color_1': 'cam_left_wrist' ,'color_2': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm'],
    json_action_data_name = ['left_arm', 'right_arm']
)


Z1_SINGLE_CONFIG = RobotConfig(
    motors=[
        "kWaist",
        "kShoulder",
        "kElbow",
        "kForearmRoll",
        "kWristAngle",
        "kWristRotate",
        "kGripper",
    ],
    cameras=[
        "cam_high",
        "cam_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_high', 'color_1': 'cam_wrist'},
    json_state_data_name = ['left_arm', 'right_arm'],
    json_action_data_name = ['left_arm', 'right_arm']
)


G1_GRIPPER_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
        ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)


G1_DEX3_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandThumb0",
        "kLeftHandThumb1",
        "kLeftHandThumb2",
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)


G1_INSPIRE_CONFIG = TactileRobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kLeftHandPinky",
        "kLeftHandRing",
        "kLeftHandMiddle",
        "kLeftHandIndex",
        "kLeftHandThumbBend",
        "kLeftHandThumbRotation",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandPinky",
        "kRightHandRing",
        "kRightHandMiddle",
        "kRightHandIndex",
        "kRightHandThumbBend",
        "kRightHandThumbRotation",
    ],
    cameras=[
        "cam_left_high",
    ],
    tactiles=[
        "left_tactile_little_finger_tip",  # 3*3
        "left_tactile_little_finger_nail",  # 12*8
        "left_tactile_little_finger_pad",  # 10*8
        "left_tactile_ring_finger_tip",  # 3*3
        "left_tactile_ring_finger_nail",  # 12*8
        "left_tactile_ring_finger_pad",  # 10*8
        "left_tactile_middle_finger_tip",  # 3*3
        "left_tactile_middle_finger_nail",  # 12*8
        "left_tactile_middle_finger_pad",  # 10*8
        "left_tactile_index_finger_tip",  # 3*3
        "left_tactile_index_finger_nail",  # 12*8
        "left_tactile_index_finger_pad",  # 10*8
        "left_tactile_thumb_tip",  # 3*3
        "left_tactile_thumb_nail",  # 12*8
        "left_tactile_thumb_middle",  # 3*3
        "left_tactile_thumb_pad",  # 12*8
        "left_tactile_palm",  # 8*14
        "right_tactile_little_finger_tip",  # 3*3
        "right_tactile_little_finger_nail",  # 12*8
        "right_tactile_little_finger_pad",  # 10*8
        "right_tactile_ring_finger_tip",  # 3*3
        "right_tactile_ring_finger_nail",  # 12*8
        "right_tactile_ring_finger_pad",  # 10*8
        "right_tactile_middle_finger_tip",  # 3*3
        "right_tactile_middle_finger_nail",  # 12*8
        "right_tactile_middle_finger_pad",  # 10*8
        "right_tactile_index_finger_tip",  # 3*3
        "right_tactile_index_finger_nail",  # 12*8
        "right_tactile_index_finger_pad",  # 10*8
        "right_tactile_thumb_tip",  # 3*3
        "right_tactile_thumb_nail",  # 12*8
        "right_tactile_thumb_middle",  # 3*3
        "right_tactile_thumb_pad",  # 12*8
        "right_tactile_palm",  # 8*14
    ],
    tactile_to_image_shape = OrderedDict({
        "left_tactile_little_finger_tip": (3, 3, 3),
        "left_tactile_little_finger_nail": (3, 12, 8),
        "left_tactile_little_finger_pad": (3, 10, 8),
        "left_tactile_ring_finger_tip": (3, 3, 3),
        "left_tactile_ring_finger_nail": (3, 12, 8),
        "left_tactile_ring_finger_pad": (3, 10, 8),
        "left_tactile_middle_finger_tip": (3, 3, 3),
        "left_tactile_middle_finger_nail": (3, 12, 8),
        "left_tactile_middle_finger_pad": (3, 10, 8),
        "left_tactile_index_finger_tip": (3, 3, 3),
        "left_tactile_index_finger_nail": (3, 12, 8),
        "left_tactile_index_finger_pad": (3, 10, 8),
        "left_tactile_thumb_tip": (3, 3, 3),
        "left_tactile_thumb_nail": (3, 12, 8),
        "left_tactile_thumb_middle": (3, 3, 3),
        "left_tactile_thumb_pad": (3, 12, 8),
        "left_tactile_palm": (3, 8, 14),
        "right_tactile_little_finger_tip": (3, 3, 3),
        "right_tactile_little_finger_nail": (3, 12, 8),
        "right_tactile_little_finger_pad": (3, 10, 8),
        "right_tactile_ring_finger_tip": (3, 3, 3),
        "right_tactile_ring_finger_nail": (3, 12, 8),
        "right_tactile_ring_finger_pad": (3, 10, 8),
        "right_tactile_middle_finger_tip": (3, 3, 3),
        "right_tactile_middle_finger_nail": (3, 12, 8),
        "right_tactile_middle_finger_pad": (3, 10, 8),
        "right_tactile_index_finger_tip": (3, 3, 3),
        "right_tactile_index_finger_nail": (3, 12, 8),
        "right_tactile_index_finger_pad": (3, 10, 8),
        "right_tactile_thumb_tip": (3, 3, 3),
        "right_tactile_thumb_nail": (3, 12, 8),
        "right_tactile_thumb_middle": (3, 3, 3),
        "right_tactile_thumb_pad": (3, 12, 8),
        "right_tactile_palm": (3, 8, 14),
    }),
    camera_to_image_key = {'color_0': 'cam_left_high'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)


ROBOT_CONFIGS = {
    "Unitree_Z1_Single": Z1_SINGLE_CONFIG,
    "Unitree_Z1_Dual": Z1_CONFIG,
    "Unitree_G1_Gripper": G1_GRIPPER_CONFIG,
    "Unitree_G1_Dex3": G1_DEX3_CONFIG,
    "Unitree_G1_Inspire": G1_INSPIRE_CONFIG,
}