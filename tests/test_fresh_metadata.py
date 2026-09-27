from __future__ import annotations

import io
import json
import zipfile

from saga_pose.data.fresh import load_metadata


def test_load_metadata_accepts_ragged_wireframe_faces() -> None:
    payload = {
        "landmarks_3d": [[0, 0, 1]],
        "wireframe_points": [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        "wireframe_faces": [[0, 1, 2], [0, 1, 3, 2]],
        "intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "annotations": [
            {
                "filename_rgb": "a_rgb.png",
                "filename_event": "a_event.png",
                "pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "keypoints": [[0, 0]],
                "bbox": [0, 0, 1, 1],
            }
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("synthetic/example/train.json", json.dumps(payload))
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        metadata = load_metadata(archive, "synthetic/example/train.json")
    assert metadata.wireframe_faces == ((0, 1, 2), (0, 1, 3, 2))
