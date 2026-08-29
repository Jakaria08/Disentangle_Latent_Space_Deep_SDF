from __future__ import annotations

import torch

from mesh_layers import LearnablePrefixPool


def test_adaptive_support_predictor_has_nonzero_gradient_and_updates():
    indices = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]], dtype=torch.long
    )
    layer = LearnablePrefixPool(8, indices, initial_support=1.5)
    features = torch.randn(2, 4, 8, requires_grad=True)
    loss = layer(features).square().mean()
    loss.backward()
    assert layer.predictor.weight.grad is not None
    assert float(layer.predictor.weight.grad.abs().sum()) > 0.0
    before = layer.predictor.weight.detach().clone()
    torch.optim.SGD(layer.parameters(), lr=0.1).step()
    assert not torch.equal(before, layer.predictor.weight.detach())


def test_adaptive_support_is_bounded():
    indices = torch.arange(5).repeat(5, 1)
    layer = LearnablePrefixPool(4, indices, initial_support=2.0)
    sequence = torch.randn(3, 5, 5, 4)
    support = layer.support(sequence)
    assert torch.all(support >= 0.0)
    assert torch.all(support <= 4.0)

