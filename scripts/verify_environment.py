"""Report whether the locked research environment can use the local GPU."""

from __future__ import annotations

import json
import platform
import sys

import torch


def main() -> None:
    """Print a machine-readable environment report and fail if CUDA is unavailable."""
    cuda_available = torch.cuda.is_available()
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cudnn": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
    }
    if cuda_available:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        report["device"] = {
            "index": device,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_gib": round(props.total_memory / 1024**3, 2),
        }

    print(json.dumps(report, indent=2))
    if not cuda_available:
        raise SystemExit("CUDA is unavailable; verify the driver and PyTorch accelerator build.")


if __name__ == "__main__":
    main()
