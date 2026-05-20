from pathlib import Path
import sys
import types

import torch
import torch.nn as nn
from PIL import Image

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


class SampleResult:
    def __init__(self, samples):
        self.samples = samples


class FakeSampler:
    def sample(self, flow_model, noise, **kwargs):
        return SampleResult(noise)


class FakeFlowModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels


class FakeProjGrid:
    def __init__(self, grid_resolution, image_resolution):
        self.grid_resolution = grid_resolution
        self.image_resolution = image_resolution

    def to(self, device):
        self.device = device
        return self


class FakeImageCondModel(nn.Module):
    def __init__(self, grid_resolution, image_resolution=1024, channels=3):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.proj_grid = FakeProjGrid(grid_resolution, image_resolution)
        self.channels = channels

    def forward(self, image, camera_angle_x, distance, mesh_scale):
        z_global = torch.ones(1, self.channels, device=camera_angle_x.device)
        z_proj = torch.arange(
            self.grid_resolution ** 3 * self.channels,
            dtype=torch.float32,
            device=camera_angle_x.device,
        ).reshape(-1, self.channels)
        return z_global, z_proj


class ForbiddenSparseStructureFlow(nn.Module):
    resolution = 32
    in_channels = 4


class ForbiddenSparseStructureDecoder(nn.Module):
    pass


class ForbiddenShapeSLatDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample_called = False

    def upsample(self, *args, **kwargs):
        self.upsample_called = True
        raise AssertionError("shape_slat_decoder.upsample should be skipped when external_hr_coords is provided")


class DenseBypassSmokePipeline(Pixal3DImageTo3DPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_sparse_structure_called = False
        self.decode_resolution = None
        self.decode_shape_token_count = None
        self.decode_tex_token_count = None
        self.sentinel = object()

    def sample_sparse_structure(self, *args, **kwargs):
        self.sample_sparse_structure_called = True
        raise AssertionError("sample_sparse_structure should be skipped when external_coords is provided")

    def decode_latent(self, shape_slat, tex_slat, resolution):
        self.decode_resolution = resolution
        self.decode_shape_token_count = shape_slat.coords.shape[0]
        self.decode_tex_token_count = tex_slat.coords.shape[0]
        return self.sentinel


def main():
    shape_channels = 4
    tex_channels = 3
    models = {
        "sparse_structure_flow_model": ForbiddenSparseStructureFlow(),
        "sparse_structure_decoder": ForbiddenSparseStructureDecoder(),
        "shape_slat_flow_model_512": FakeFlowModel(shape_channels),
        "shape_slat_flow_model_1024": FakeFlowModel(shape_channels),
        "shape_slat_decoder": ForbiddenShapeSLatDecoder(),
        "tex_slat_flow_model_1024": FakeFlowModel(shape_channels + tex_channels),
        "tex_slat_decoder": nn.Identity(),
    }
    pipe = DenseBypassSmokePipeline(
        models=models,
        sparse_structure_sampler=FakeSampler(),
        shape_slat_sampler=FakeSampler(),
        tex_slat_sampler=FakeSampler(),
        sparse_structure_sampler_params={},
        shape_slat_sampler_params={},
        tex_slat_sampler_params={},
        shape_slat_normalization={"std": [1.0] * shape_channels, "mean": [0.0] * shape_channels},
        tex_slat_normalization={"std": [1.0] * tex_channels, "mean": [0.0] * tex_channels},
        low_vram=False,
    )
    pipe._device = "cpu"
    pipe.image_cond_model_ss = FakeImageCondModel(32)
    pipe.image_cond_model_shape_512 = FakeImageCondModel(16)
    pipe.image_cond_model_shape_1024 = FakeImageCondModel(16)
    pipe.image_cond_model_tex_1024 = FakeImageCondModel(16)

    external_coords = Path("experiments/dense_bypass/coords/glasses_512_r0125/pixal_coords_32.npz")
    external_hr_coords = Path("experiments/dense_bypass/coords/glasses_512_r0125/pixal_coords_64.npz")
    expected_hr_tokens = pipe._prepare_external_coords(external_hr_coords, 64).shape[0]

    result = pipe.run(
        image=Image.new("RGB", (8, 8), color=(0, 0, 0)),
        camera_params={"camera_angle_x": 0.8575560450553894, "distance": 2.0, "mesh_scale": 1.0},
        external_coords=external_coords,
        external_hr_coords=external_hr_coords,
        external_coords_resolution=32,
        external_hr_coords_resolution=64,
        dense_bypass_debug=True,
        preprocess_image=False,
        pipeline_type="1024_cascade",
    )

    assert result is pipe.sentinel
    assert pipe.decode_resolution == 1024
    assert pipe.decode_shape_token_count == expected_hr_tokens
    assert pipe.decode_tex_token_count == expected_hr_tokens
    assert not pipe.sample_sparse_structure_called
    assert not pipe.models["shape_slat_decoder"].upsample_called
    print("PASS: Dense Bypass run smoke returned sentinel at resolution 1024 without forbidden paths.")


if __name__ == "__main__":
    main()
