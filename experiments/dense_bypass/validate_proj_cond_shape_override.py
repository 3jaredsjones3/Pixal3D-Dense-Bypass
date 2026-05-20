from pathlib import Path
import sys
import types

import torch
import torch.nn as nn

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


class FakeProjGrid:
    def __init__(self, grid_resolution, image_resolution):
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution

    def to(self, device):
        self.device = device
        return self


class FakeImageCondModel(nn.Module):
    def __init__(self, grid_resolution=2, image_resolution=16, channels=3, fail=False):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.proj_grid = FakeProjGrid(grid_resolution, image_resolution)
        self.channels = channels
        self.fail = fail

    def forward(self, image, camera_angle_x, distance, mesh_scale):
        if self.fail:
            raise RuntimeError("intentional fake image-cond failure")
        values = torch.arange(
            self.grid_resolution ** 3 * self.channels,
            device=camera_angle_x.device,
            dtype=torch.float32,
        )
        return torch.ones(1, self.channels, device=camera_angle_x.device), values.reshape(-1, self.channels)


def make_pipe():
    pipe = Pixal3DImageTo3DPipeline(models=None)
    pipe._device = "cpu"
    pipe.low_vram = False
    return pipe


def test_exception_restores_exact_proj_grid():
    pipe = make_pipe()
    model = FakeImageCondModel(fail=True)
    original_grid_resolution = model.grid_resolution
    original_proj_grid = model.proj_grid
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)

    try:
        pipe.get_proj_cond_shape(model, [object()], coords, grid_resolution_override=4)
        raise RuntimeError("fake failure unexpectedly passed")
    except RuntimeError as e:
        assert "intentional fake image-cond failure" in str(e)

    assert model.grid_resolution == original_grid_resolution
    assert model.proj_grid is original_proj_grid
    print("PASS exception path restores exact proj_grid object")


def test_success_returns_sparse_tensor_and_restores():
    pipe = make_pipe()
    model = FakeImageCondModel()
    original_grid_resolution = model.grid_resolution
    original_proj_grid = model.proj_grid
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32)

    cond = pipe.get_proj_cond_shape(model, [object()], coords, grid_resolution_override=4)
    proj = cond["cond"]["proj"]

    assert proj.coords.shape == coords.shape
    assert torch.equal(proj.coords, coords)
    assert proj.feats.shape == (coords.shape[0], model.channels)
    assert model.grid_resolution == original_grid_resolution
    assert model.proj_grid is original_proj_grid
    print("PASS success path returns sparse proj features and restores exact proj_grid object")


def main():
    test_exception_restores_exact_proj_grid()
    test_success_returns_sparse_tensor_and_restores()
    print("\nPASS: get_proj_cond_shape grid override validation works.")


if __name__ == "__main__":
    main()
