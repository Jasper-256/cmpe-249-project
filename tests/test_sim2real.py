import torch

from saga_pose.data.sim2real import (
    Sim2RealConfig,
    augment_batch,
    clipping_from_rgb,
    invert_points,
    roll_only_config,
    sample_affine,
    transform_points,
    warp,
)


def _dot_image(points: torch.Tensor, size: int = 256) -> torch.Tensor:
    image = torch.zeros(points.shape[0], 1, size, size)
    for index, (x, y) in enumerate(points.round().long().tolist()):
        image[index, 0, y - 1 : y + 2, x - 1 : x + 2] = 1.0
    return image


def test_warp_moves_pixels_where_transform_points_says() -> None:
    generator = torch.Generator().manual_seed(0)
    points = torch.tensor([[100.0, 140.0], [150.0, 90.0], [128.0, 170.0], [90.0, 110.0]])
    affine = sample_affine(4, Sim2RealConfig(), generator)
    warped = warp(_dot_image(points), affine)
    expected = transform_points(points[:, None], affine)[:, 0]
    flat = warped.flatten(1).argmax(dim=1)
    found = torch.stack([flat % 256, flat // 256], dim=1).float()
    assert torch.all((found - expected).abs() <= 1.5)


def test_invert_points_round_trips() -> None:
    generator = torch.Generator().manual_seed(1)
    points = torch.rand(3, 18, 2) * 255
    affine = sample_affine(3, Sim2RealConfig(), generator)
    assert torch.allclose(
        invert_points(transform_points(points, affine), affine), points, atol=1e-3
    )


def test_roll_only_changes_geometry_but_not_values() -> None:
    generator = torch.Generator().manual_seed(2)
    inputs = {
        "rgb": torch.rand(2, 3, 256, 256) * 0.5,
        "event": torch.zeros(2, 3, 256, 256),
        "clipping": torch.zeros(2, 16, 16),
    }
    inputs["event"][:, :, 100:110, 100:110] = 1.0
    keypoints = torch.full((2, 5, 2), 128.0)
    out, moved, affine = augment_batch(inputs, keypoints, roll_only_config(), generator)
    assert out["rgb"].shape == inputs["rgb"].shape
    assert torch.allclose(
        affine[:, :, :2] @ affine[:, :, :2].transpose(1, 2), torch.eye(2).expand(2, 2, 2), atol=1e-5
    )
    # Mass is conserved up to interpolation: no dropout or noise events were added.
    assert abs(float(out["event"].sum() / inputs["event"].sum()) - 1.0) < 0.05
    assert moved.shape == keypoints.shape


def test_full_augmentation_stays_in_range_and_is_seeded() -> None:
    inputs = {
        "rgb": torch.rand(4, 3, 256, 256),
        "event": torch.zeros(4, 3, 256, 256),
        "clipping": torch.zeros(4, 16, 16),
    }
    keypoints = torch.rand(4, 18, 2) * 255
    first = augment_batch(inputs, keypoints, Sim2RealConfig(), torch.Generator().manual_seed(7))
    second = augment_batch(inputs, keypoints, Sim2RealConfig(), torch.Generator().manual_seed(7))
    for key in ("rgb", "event", "clipping"):
        assert torch.equal(first[0][key], second[0][key])
        assert float(first[0][key].min()) >= 0.0 and float(first[0][key].max()) <= 1.0
    assert torch.equal(first[1], second[1])


def test_clipping_from_rgb_counts_saturated_pixels() -> None:
    rgb = torch.zeros(1, 3, 256, 256)
    rgb[0, 1, :16, :16] = 1.0
    grid = clipping_from_rgb(rgb)
    assert grid.shape == (1, 16, 16)
    assert float(grid[0, 0, 0]) == 1.0 and float(grid.sum()) == 1.0
