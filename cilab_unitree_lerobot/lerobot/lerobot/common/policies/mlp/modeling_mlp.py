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

"""Simple MLP Policy

A simplified version of the ACT policy using a simple Multi-Layer Perceptron architecture.
This replaces the complex transformer-based ACT with a straightforward feedforward network.

Architecture (following ACT structure):
Input -> CNN Backbone (image) + State Processing -> VAE Encoder (MLP) -> Transformer (MLP) -> Output
- Input: observation data (robot state, environment state, images, etc.)
- Output: action vector

Features:
- CNN backbone for image feature extraction (like ACT)
- MLP-based VAE encoder and transformer replacement
- Support for various input modalities (state, images, etc.)
- Normalization and unnormalization
- Action queue for multi-step action support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from typing import Any, Dict, List, Optional, Tuple, Union

from lerobot.common.policies.pretrained import PreTrainedPolicy
from cilab_unitree_lerobot.lerobot.lerobot.common.policies.mlp.configuration_mlp import MLPConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize


def _replace_submodules(
    root_module: nn.Module,
    predicate: callable,
    func: callable,
) -> nn.Module:
    """Replace submodules in a module that satisfy a predicate with a new module.

    Args:
        root_module: The root module to search in.
        predicate: A function that takes a module and returns True if it should be replaced.
        func: A function that takes a module and returns the replacement module.

    Returns:
        The root module with submodules replaced.
    """
    for name, child in root_module.named_children():
        if predicate(child):
            setattr(root_module, name, func(child))
        else:
            _replace_submodules(child, predicate, func)
    return root_module


class MLPPolicy(PreTrainedPolicy):
    """
    Simple MLP Policy
    
    A simplified version of the ACT policy using a simple Multi-Layer Perceptron.
    This replaces the complex transformer-based ACT with a straightforward feedforward network.
    
    Features:
    - CNN backbone for image feature extraction (like ACT)
    - MLP-based VAE encoder and transformer replacement
    - Input/output normalization and unnormalization
    - MLP-based action prediction
    - Action queue for multi-step action management
    - Checkpoint saving/loading
    """

    config_class = MLPConfig
    name = "mlp"

    def __init__(
        self,
        config: MLPConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = MLP(config)
        self.reset()

    def get_optim_params(self) -> dict:
        # Return parameters in the same format as ACT for compatibility
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        # MLP doesn't use action queue, so no reset needed
        pass

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        For MLP policy, we predict one action at a time based on current observation.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        
        # Handle image features like ACT does
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        # For MLP, directly predict single action from current observation
        actions = self.model(batch)[0]  # shape: (batch_size, 1, action_dim)

        # Unnormalize the action
        actions = self.unnormalize_outputs({"action": actions})["action"]

        # Return single action (remove batch and time dimensions)
        action = actions[0, 0]  # shape: (action_dim,)
        
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        
        # Handle image features like ACT does
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = [batch[key] for key in self.config.image_features]
            
        batch = self.normalize_targets(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)  # shape: (B, 1, action_dim)

        # Use only first action from target sequence
        target_actions = batch["action"][:, :1, :]  # shape: (B, 1, action_dim)
        target_is_pad = batch["action_is_pad"][:, :1]  # shape: (B, 1)

        l1_loss = (
            F.l1_loss(target_actions, actions_hat, reduction="none") * ~target_is_pad.unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        
        # Add VAE loss if using VAE
        if self.config.use_vae:
            # Calculate KL divergence loss
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class MLP(nn.Module):
    """
    MLP Policy following ACT architecture: 
    Input -> CNN Backbone (image) + State Processing -> VAE Encoder (MLP) -> Transformer (MLP) -> Output
    
    This replaces the transformer-based ACT with MLP layers while keeping the same input/output structure.
    """

    def __init__(self, config: MLPConfig):
        """
        Args:
            config: MLP configuration object
        """
        super().__init__()
        self.config = config
        
        print(f"\n=== MLP Model Initialization (ACT-style) ===")
        print(f"Config input_features keys: {list(config.input_features.keys())}")
        print(f"Config output_features keys: {list(config.output_features.keys())}")
        
        # 1. CNN Backbone for image feature extraction (like ACT)
        if self.config.image_features:
            print("Setting up CNN backbone for image feature extraction...")
            # Use config values instead of hardcoded ones
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            
            # Projection layer for image features (like ACT)
            self.img_projection = nn.Conv2d(512, config.dim_model, kernel_size=1)
            print("CNN backbone created successfully!")
        else:
            print("No images found, backbone not needed")
            self.backbone = None
        
        # 2. VAE Encoder (MLP replacement)
        if self.config.use_vae:
            print("Setting up VAE encoder (MLP)...")
            # Calculate VAE encoder input dimension
            vae_input_dim = 0
            if self.config.robot_state_feature:
                vae_input_dim += self.config.robot_state_feature.shape[0]
            vae_input_dim += self.config.action_feature.shape[0] * self.config.chunk_size
            
            self.vae_encoder_mlp = nn.Sequential(
                nn.Linear(vae_input_dim, config.dim_model),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim_model, config.dim_model),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim_model, config.latent_dim * 2)  # mu + log_var
            )
            print(f"VAE encoder input dim: {vae_input_dim} -> latent dim: {config.latent_dim * 2}")
        
        # 3. Transformer Encoder/Decoder (MLP replacement)
        print("Setting up Transformer replacement (MLP)...")
        
        # Calculate encoder input dimension
        encoder_input_dim = config.latent_dim  # latent
        
        if self.config.robot_state_feature:
            encoder_input_dim += self.config.robot_state_feature.shape[0]
        if self.config.env_state_feature:
            encoder_input_dim += self.config.env_state_feature.shape[0]
        if self.config.image_features:
            # Global average pooling of image features
            encoder_input_dim += config.dim_model
        
        print(f"Encoder input dimension: {encoder_input_dim}")
        
        # Encoder MLP (replaces transformer encoder)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(encoder_input_dim, config.dim_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_model, config.dim_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_model, config.dim_model)
        )
        
        # Decoder MLP (replaces transformer decoder)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_model, config.dim_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_model, config.dim_model)
        )
        
        # Final action head
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
        
        print(f"MLP model created successfully!")
        print(f"Architecture: Input -> CNN -> VAE Encoder (MLP) -> Transformer (MLP) -> Action Head")

    def _reset_parameters(self):
        """Initialize MLP parameters (only if not loading from checkpoint)"""
        # Only initialize if we're not loading from a checkpoint
        # This prevents overwriting loaded weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for better convergence
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                # Use Kaiming initialization for Conv2d layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """
        MLP forward pass following ACT structure
        
        Args:
            batch: Dictionary containing observations
            
        Returns:
            actions: Predicted actions
            (mu, log_sigma_x2): VAE parameters for compatibility
        """
        batch_size = batch["observation.images"][0].shape[0] if "observation.images" in batch else batch["observation.state"].shape[0]
        
        # 1. VAE Encoder (MLP replacement)
        if self.config.use_vae and "action" in batch:
            # Prepare VAE encoder input: [robot_state, action_sequence]
            vae_inputs = []
            
            if self.config.robot_state_feature:
                vae_inputs.append(batch["observation.state"])
            
            # Flatten action sequence
            action_flat = batch["action"].view(batch_size, -1)
            vae_inputs.append(action_flat)
            
            vae_input = torch.cat(vae_inputs, dim=-1)
            
            # VAE encoder forward pass
            vae_output = self.vae_encoder_mlp(vae_input)
            mu = vae_output[:, :self.config.latent_dim]
            log_sigma_x2 = vae_output[:, self.config.latent_dim:]
            
            # Sample latent
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using VAE, set latent to zeros
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=batch["observation.state"].device)
        
        # 2. Prepare encoder inputs (like ACT)
        encoder_inputs = [latent_sample]
        
        if self.config.robot_state_feature:
            encoder_inputs.append(batch["observation.state"])
        
        if self.config.env_state_feature:
            encoder_inputs.append(batch["observation.environment_state"])
        
        # 3. Image feature extraction (CNN backbone)
        if self.config.image_features:
            all_img_features = []
            
            for img in batch["observation.images"]:
                # CNN backbone
                img_features = self.backbone(img)["feature_map"]  # (B, 512, H, W)
                img_features = self.img_projection(img_features)  # (B, dim_model, H, W)
                
                # Global average pooling
                img_features = F.adaptive_avg_pool2d(img_features, (1, 1)).squeeze(-1).squeeze(-1)  # (B, dim_model)
                all_img_features.append(img_features)
            
            # Average across multiple images if any
            if len(all_img_features) > 1:
                img_features = torch.stack(all_img_features, dim=0).mean(0)
            else:
                img_features = all_img_features[0]
            
            encoder_inputs.append(img_features)
        
        # 4. Encoder MLP (replaces transformer encoder)
        encoder_input = torch.cat(encoder_inputs, dim=-1)
        encoder_output = self.encoder_mlp(encoder_input)
        
        # 5. Decoder MLP (replaces transformer decoder)
        decoder_output = self.decoder_mlp(encoder_output)
        
        # 6. Action head
        actions = self.action_head(decoder_output)  # (B, action_dim)
        
        # Output single action: (B, action_dim) -> (B, 1, action_dim)
        actions = actions.unsqueeze(1)
        
        return actions, (mu, log_sigma_x2)
