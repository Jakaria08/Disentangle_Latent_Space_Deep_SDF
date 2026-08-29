from __future__ import annotations

import torch

from conditional_spiral_unet import mesh_volume, vertex_normals


def test_tetrahedron_volume_and_normals_are_finite():
    vertices = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    volume = mesh_volume(vertices, faces)
    normals = vertex_normals(vertices, faces)
    assert torch.allclose(volume, torch.tensor([1.0 / 6.0]), atol=1.0e-6)
    assert torch.isfinite(normals).all()
    assert torch.allclose(torch.linalg.vector_norm(normals, dim=-1), torch.ones(1, 4), atol=1.0e-6)

