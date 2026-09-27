"""Small keypoint regressor used only for data-path competence checks."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class _ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(8, output_channels), num_channels=output_channels),
            nn.GELU(),
        )


class PilotKeypointRegressor(nn.Module):
    """A compact direct-coordinate network for the 16-sample overfit gate.

    This is deliberately not the final reportable architecture. Its job is to
    establish that each modality, crop transform, and target path can learn.
    """

    def __init__(self, num_keypoints: int) -> None:
        super().__init__()
        if num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")
        self.num_keypoints = num_keypoints
        self.features = nn.Sequential(
            _ConvBlock(3, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 96),
            _ConvBlock(96, 128),
            # Four stride-2 blocks map the frozen 256x256 pilot input to 16x16.
            # Fixed average pooling avoids AdaptiveAvgPool2d's nondeterministic
            # CUDA backward path while preserving the intended 4x4 feature map.
            nn.AvgPool2d(kernel_size=4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.GELU(),
            nn.Linear(512, num_keypoints * 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 256, 256):
            raise ValueError("inputs must have the frozen shape (B, 3, 256, 256)")
        output = cast(Tensor, self.head(self.features(inputs)))
        return output.reshape(inputs.shape[0], self.num_keypoints, 2)
