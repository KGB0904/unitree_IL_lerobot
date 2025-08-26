#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLP Policy Configuration

This file defines all configuration parameters for the MLP policy.
This is a simplified version of the ACT configuration adapted for a simple MLP.

Main configuration groups:
1. Input/Output structure settings
2. MLP architecture settings
3. Training settings

Configuration usage:
- Default parameters are optimized for general robot manipulation tasks
- Modify input_shapes, output_shapes according to environment/sensors
- Adjust learning rate, batch size, etc. based on training performance
"""

from dataclasses import dataclass, field
from typing import Union

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType


@PreTrainedConfig.register_subclass("mlp")
@dataclass
class MLPConfig(PreTrainedConfig):
    """
    Configuration class for simple MLP policy

    Default values are optimized for general robot manipulation tasks.

    The most frequently changed parameters are those that depend on environment/sensors.
    These are `input_shapes` and `output_shapes`.

    Notes on inputs and outputs:
        - At least one of the following is required:
            - One or more keys starting with "observation.image" (for input)
              AND/OR
            - "observation.environment_state" key for input
        - If there are multiple keys starting with "observation.images.", they are treated as multiple camera views.
          Currently, all images must have the same shape.
        - "observation.state" key is optional, for robot proprioceptive state.
        - "action" is required as output key.

    Args:
        n_obs_steps: Number of environment steps to pass to the policy (current step and previous steps)
        chunk_size: Size of action prediction "chunk" (in environment steps)
        n_action_steps: Number of action steps to execute in the environment per policy call
            This should not be larger than chunk size. For example, if chunk size is 100,
            you can set this to 50. This means the model predicts 100 steps of actions,
            executes 50 in the environment, and discards the remaining 50.
        input_shapes: Dictionary defining the shape of the policy's input data. Keys are input data names,
            values are lists representing the dimensions of that data. For example, "observation.image"
            represents camera input with dimensions [3, 96, 96], meaning 3 color channels and 96x96 resolution.
            Important: `input_shapes` does not include batch or time dimensions.
        output_shapes: Dictionary defining the shape of the policy's output data. Keys are output data names,
            values are lists representing the dimensions of that data. For example, "action" represents
            14-dimensional action with shape [14]. Important: `output_shapes` does not include batch or time dimensions.
        input_normalization_modes: Dictionary where keys represent modalities (e.g., "observation.state") and
            values specify the normalization mode to apply. Available modes are "mean_std"
            (subtract mean, divide by std) and "min_max" (rescale to [-1, 1] range).
        output_normalization_modes: Similar dictionary to `normalize_input_modes` but for
            unnormalizing to original scale. This is also used for normalizing training targets.
        dim_model: Main hidden dimension for MLP layers
        n_encoder_layers: Number of MLP layers to use
        dropout: Dropout rate for MLP layers
                optimizer_lr: Learning rate for optimizer
        optimizer_weight_decay: Weight decay for optimizer
        optimizer_lr_backbone: Learning rate for backbone (if applicable)
        use_amp: Whether to use automatic mixed precision
    """

    # Input/output structure settings
    n_obs_steps: int = 1          # Number of observation steps (currently only 1 supported)
    chunk_size: int = 1           # Action prediction chunk size (1 for MLP)
    n_action_steps: int = 1       # Number of action steps to execute at once (1 for MLP)

    # Normalization mapping settings
    # Define normalization methods for each data type
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,    # Visual data: mean-std normalization
            "STATE": NormalizationMode.MEAN_STD,     # State data: mean-std normalization
            "ACTION": NormalizationMode.MEAN_STD,    # Action data: mean-std normalization
        }
    )

    # Input/output features (inherited from PreTrainedConfig)
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    # Image processing settings (like ACT)
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = True  # Whether to replace batch normalization with group normalization in the backbone

    # VAE settings (like ACT)
    use_vae: bool = True  # Whether to use VAE encoder
    latent_dim: int = 32  # VAE latent dimension
    kl_weight: float = 10.0  # Weight for KL divergence loss

    # MLP architecture settings
    dim_model: int = 512          # Hidden layer dimension
    n_encoder_layers: int = 4     # Number of hidden layers
    dropout: float = 0.1          # Dropout rate

    # Training settings
    optimizer_lr: float = 1e-5                    # Base learning rate
    optimizer_weight_decay: float = 1e-4          # Weight decay
    optimizer_lr_backbone: float = 1e-5           # Backbone learning rate (currently same as base)
    use_amp: bool = False                         # Whether to use automatic mixed precision

    def __post_init__(self):
        """Validate configuration after initialization"""
        super().__post_init__()

        # Input validation (incomplete)
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        
        # Validate vision backbone
        image_features = [key for key in self.input_features.keys() if key.startswith("observation.images.")]
        if image_features and not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        """Return optimizer preset configuration"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        """Return scheduler preset configuration (currently None)"""
        return None

    def validate_features(self) -> None:
        """Validate that at least one input feature is provided"""
        if not self.input_features:
            raise ValueError("You must provide at least one input feature.")
        
        # Check if we have at least one image or environment state
        image_features = [key for key in self.input_features.keys() if key.startswith("observation.images.")]
        env_state_features = [key for key in self.input_features.keys() if key.startswith("observation.environment_state")]
        
        if not image_features and not env_state_features:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
    
    @property
    def image_features(self) -> list[str]:
        """Get list of image feature keys from input_features"""
        return [key for key in self.input_features.keys() if key.startswith("observation.images.")]

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        """Get robot state feature from input_features (like ACT)"""
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        """Get environment state feature from input_features (like ACT)"""
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def action_feature(self) -> PolicyFeature | None:
        """Get action feature from output_features (like ACT)"""
        for _, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION:
                return ft
        return None

    @property
    def observation_delta_indices(self) -> None:
        """Observation delta indices (currently None)"""
        return None

    @property
    def action_delta_indices(self) -> list:
        """Action delta indices - returns list of indices up to chunk size"""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Reward delta indices (currently None)"""
        return None
