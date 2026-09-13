"""Foundation smoke tests for the locked research environment."""

import torch

from saga_pose import __version__


def test_package_version() -> None:
    """The source package is importable through the project environment."""
    assert __version__ == "0.1.0"


def test_torch_tensor_operation() -> None:
    """PyTorch executes a deterministic basic tensor operation."""
    values = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(values.square(), torch.tensor([1.0, 4.0, 9.0]))
