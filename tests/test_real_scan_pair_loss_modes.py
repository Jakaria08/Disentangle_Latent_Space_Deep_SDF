import torch

import train_deep_sdf_longitudinal as longitudinal


class ZeroVelocity(torch.nn.Module):
    def forward(self, z, s, t, age_cond=None):
        return torch.zeros_like(z)


class CountingDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return torch.zeros(inputs.shape[0], 1, device=inputs.device, dtype=inputs.dtype)


def test_real_scan_pair_direction_latent_only_skips_decoder():
    decoder = CountingDecoder()
    z_source = torch.zeros(1, 4)
    z_target = torch.zeros(1, 4)

    reconstruction_loss, latent_loss = (
        longitudinal._compute_real_scan_pair_direction_losses(
            decoder=decoder,
            temporal_flow=ZeroVelocity(),
            z_source=z_source,
            source_time=0.0,
            z_target=z_target,
            target_time=1.0,
            target_condition=None,
            target_sdf_data=None,
            clamp_min=-0.1,
            clamp_max=0.1,
            compute_reconstruction=False,
            compute_latent=True,
        )
    )

    assert reconstruction_loss is None
    assert latent_loss is not None
    assert float(latent_loss.item()) == 0.0
    assert decoder.calls == 0


def test_real_scan_pair_direction_reconstruction_only_skips_latent():
    decoder = CountingDecoder()
    z_source = torch.zeros(1, 4)
    z_target = torch.ones(1, 4)
    target_sdf = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    reconstruction_loss, latent_loss = (
        longitudinal._compute_real_scan_pair_direction_losses(
            decoder=decoder,
            temporal_flow=ZeroVelocity(),
            z_source=z_source,
            source_time=0.0,
            z_target=z_target,
            target_time=1.0,
            target_condition=None,
            target_sdf_data=target_sdf,
            clamp_min=-0.1,
            clamp_max=0.1,
            compute_reconstruction=True,
            compute_latent=False,
        )
    )

    assert reconstruction_loss is not None
    assert float(reconstruction_loss.item()) == 0.0
    assert latent_loss is None
    assert decoder.calls > 0


def test_real_scan_pair_batch_latent_only_does_not_load_auxiliary_samples(monkeypatch):
    decoder = CountingDecoder()
    lat_vecs = torch.nn.Embedding(1, 4)
    with torch.no_grad():
        lat_vecs.weight.zero_()

    def _unexpected_loader(*args, **kwargs):
        raise AssertionError("auxiliary SDF samples should not be loaded in latent-only mode")

    monkeypatch.setattr(longitudinal, "_load_auxiliary_sdf_samples", _unexpected_loader)

    losses = longitudinal._compute_real_scan_pair_batch_losses(
        decoder=decoder,
        temporal_flow=ZeroVelocity(),
        lat_vecs=lat_vecs,
        unique_subjects=torch.tensor([0], dtype=torch.long),
        subject_to_scan_indices={0: [0, 1]},
        sdf_dataset=None,
        scan_to_time_cpu=torch.tensor([0.0, 1.0], dtype=torch.float32),
        scan_age_condition=torch.tensor([[0.0], [0.0]], dtype=torch.float32),
        subject_baseline_time=torch.tensor([0.0], dtype=torch.float32),
        use_age_conditioning=False,
        pairs_per_subject=1,
        num_samples=8,
        use_forward=True,
        use_backward=True,
        clamp_min=-0.1,
        clamp_max=0.1,
        compute_reconstruction=False,
        compute_latent=True,
    )

    assert losses["forward_reconstruction"] is None
    assert losses["backward_reconstruction"] is None
    assert losses["forward_latent"] is not None
    assert losses["backward_latent"] is not None
    assert decoder.calls == 0
