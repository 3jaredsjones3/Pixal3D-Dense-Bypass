from pathlib import Path
import sys
import types

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.modules.setdefault("pixal3d.pipelines.rembg", types.ModuleType("pixal3d.pipelines.rembg"))
sys.modules.setdefault(
    "pixal3d.modules.image_feature_extractor",
    types.ModuleType("pixal3d.modules.image_feature_extractor"),
)
representations_stub = types.ModuleType("pixal3d.representations")
representations_stub.Mesh = object
representations_stub.MeshWithVoxel = object
sys.modules.setdefault("pixal3d.representations", representations_stub)

from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline


def main():
    pipe = Pixal3DImageTo3DPipeline(models=None)
    pipe._device = "cuda" if torch.cuda.is_available() else "cpu"

    cases = [
        ("glasses_32", Path("experiments/dense_bypass/coords/glasses_512_r0125/pixal_coords_32.npz"), 32),
        ("glasses_64", Path("experiments/dense_bypass/coords/glasses_512_r0125/pixal_coords_64.npz"), 64),
        ("dumbbell_32", Path("experiments/dense_bypass/coords/dumbbell_r0125_512/pixal_coords_32.npz"), 32),
        ("dumbbell_64", Path("experiments/dense_bypass/coords/dumbbell_r0125_512/pixal_coords_64.npz"), 64),
    ]

    for name, path, res in cases:
        coords = pipe._prepare_external_coords(path, res)
        xyz = coords[:, 1:]
        unique = torch.unique(coords, dim=0).shape[0] == coords.shape[0]

        assert coords.ndim == 2 and coords.shape[1] == 4
        assert coords.dtype == torch.int32
        assert str(coords.device) == pipe.device
        assert bool(((xyz >= 0) & (xyz < res)).all())
        assert unique

        print({
            "name": name,
            "path": str(path),
            "shape": tuple(coords.shape),
            "dtype": str(coords.dtype),
            "device": str(coords.device),
            "min": xyz.min(dim=0).values.tolist(),
            "max": xyz.max(dim=0).values.tolist(),
            "unique": unique,
        })

    bad_float = np.array([[0, 1.2, 2, 3]], dtype=np.float32)
    try:
        pipe._prepare_external_coords(bad_float, 32)
        raise RuntimeError("bad_float unexpectedly passed")
    except ValueError as e:
        print("PASS bad_float rejected:", str(e))

    bad_bounds = np.array([[0, 32, 0, 0]], dtype=np.int32)
    try:
        pipe._prepare_external_coords(bad_bounds, 32)
        raise RuntimeError("bad_bounds unexpectedly passed")
    except ValueError as e:
        print("PASS bad_bounds rejected:", str(e))

    bad_dupe = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    try:
        pipe._prepare_external_coords(bad_dupe, 32)
        raise RuntimeError("bad_dupe unexpectedly passed")
    except ValueError as e:
        print("PASS bad_dupe rejected:", str(e))

    print("\nPASS: external coord validation helper works.")


if __name__ == "__main__":
    main()
