from __future__ import annotations

import torch

from region_layout import load_region_layout


def test_selected_lamm_layout_is_exact_partition() -> None:
    layout = load_region_layout()
    assert layout.region_counts == [43, 86]
    assert len(layout.fingerprint) == 64
    for scale in layout.scales:
        active = scale.member_idx[scale.member_mask]
        assert active.numel() == 2746
        assert torch.equal(torch.sort(active).values, torch.arange(2746))

