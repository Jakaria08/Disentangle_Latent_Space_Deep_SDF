import torch


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def gradient_reverse(x, strength=1.0):
    return GradientReverse.apply(x, strength)


def build_mlp(input_dim, hidden_dims, output_dim):
    if len(hidden_dims) == 0:
        raise ValueError("Hidden dims must be non-empty.")
    dims = [int(input_dim)] + [int(x) for x in hidden_dims] + [int(output_dim)]
    layers = []
    for idx in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*layers)


class StandardTemporalFlowMLP(torch.nn.Module):
    def __init__(self, latent_size, hidden_dims, age_condition_dim=0):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("Flow hidden dims must be non-empty.")
        self.age_condition_dim = max(0, int(age_condition_dim))
        dims = [latent_size + 2 + self.age_condition_dim] + list(hidden_dims) + [latent_size]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ReLU(inplace=True))
        self.net = torch.nn.Sequential(*layers)

    def _prepare_age_condition(self, z, age_cond):
        if self.age_condition_dim <= 0:
            return None

        if age_cond is None:
            return torch.zeros(
                z.shape[0],
                self.age_condition_dim,
                device=z.device,
                dtype=z.dtype,
            )

        age = age_cond
        if age.dim() == 1:
            age = age.unsqueeze(1)
        age = age.to(device=z.device, dtype=z.dtype)

        if age.shape[1] == self.age_condition_dim:
            return age
        if age.shape[1] == 1 and self.age_condition_dim > 1:
            return age.repeat(1, self.age_condition_dim)

        raise ValueError(
            f"Invalid age condition width: got {age.shape[1]}, "
            f"expected {self.age_condition_dim}"
        )

    def forward(self, z, s, t, age_cond=None):
        if s.dim() == 1:
            s = s.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        parts = [z, s, t]
        if self.age_condition_dim > 0:
            parts.append(self._prepare_age_condition(z, age_cond))
        x = torch.cat(parts, dim=1)
        return self.net(x)


class AdditiveVelocityFlow(torch.nn.Module):
    """Full-latent additive velocity: age + disease + residual."""

    def __init__(
        self,
        latent_size,
        flow_hidden_dims,
        age_branch_hidden_dims=None,
        disease_branch_hidden_dims=None,
        residual_branch_hidden_dims=None,
        probe_hidden_dims=None,
        age_condition_dim=1,
        disease_uses_latent=False,
        residual_uses_diagnosis=False,
    ):
        super().__init__()
        self.latent_size = int(latent_size)
        self.age_condition_dim = max(0, int(age_condition_dim))
        self.disease_uses_latent = bool(disease_uses_latent)
        self.residual_uses_diagnosis = bool(residual_uses_diagnosis)

        age_branch_hidden_dims = (
            list(flow_hidden_dims)
            if age_branch_hidden_dims is None
            else list(age_branch_hidden_dims)
        )
        disease_branch_hidden_dims = (
            list(flow_hidden_dims)
            if disease_branch_hidden_dims is None
            else list(disease_branch_hidden_dims)
        )
        residual_branch_hidden_dims = (
            list(flow_hidden_dims)
            if residual_branch_hidden_dims is None
            else list(residual_branch_hidden_dims)
        )
        probe_hidden_dims = [64] if probe_hidden_dims is None else list(probe_hidden_dims)

        self.age_net = build_mlp(2, age_branch_hidden_dims, self.latent_size)

        disease_input_dim = 3
        if self.disease_uses_latent:
            disease_input_dim += self.latent_size
        self.disease_net = build_mlp(
            disease_input_dim, disease_branch_hidden_dims, self.latent_size
        )

        residual_input_dim = self.latent_size + 2
        if self.residual_uses_diagnosis:
            residual_input_dim += 1
        self.residual_net = build_mlp(
            residual_input_dim, residual_branch_hidden_dims, self.latent_size
        )

        self.age_probe = build_mlp(self.latent_size, probe_hidden_dims, 1)
        self.disease_probe = build_mlp(self.latent_size, probe_hidden_dims, 1)
        self.age_leak_from_disease = build_mlp(self.latent_size, probe_hidden_dims, 1)
        self.age_leak_from_residual = build_mlp(self.latent_size, probe_hidden_dims, 1)
        self.disease_leak_from_age = build_mlp(self.latent_size, probe_hidden_dims, 1)
        self.disease_leak_from_residual = build_mlp(self.latent_size, probe_hidden_dims, 1)

    @property
    def is_additive_velocity_flow(self):
        return True

    def _prepare_time(self, value, z):
        if value.dim() == 1:
            value = value.unsqueeze(1)
        return value.to(device=z.device, dtype=z.dtype)

    def _prepare_disease_score(self, z, age_cond):
        if self.age_condition_dim <= 0 or age_cond is None:
            return torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        disease = age_cond
        if disease.dim() == 1:
            disease = disease.unsqueeze(1)
        disease = disease.to(device=z.device, dtype=z.dtype)
        if disease.shape[1] < 1:
            raise ValueError("Disease condition must have at least one column.")
        if disease.shape[0] == 1 and z.shape[0] != 1:
            disease = disease.repeat(z.shape[0], 1)
        if disease.shape[0] != z.shape[0]:
            raise ValueError(
                f"Disease batch size mismatch: got {disease.shape[0]}, "
                f"expected {z.shape[0]}."
            )
        return disease[:, 0:1]

    def _disease_velocity_for_score(self, z, s, t, disease_score):
        parts = [s, t, disease_score]
        if self.disease_uses_latent:
            parts.insert(0, z)
        return self.disease_net(torch.cat(parts, dim=1))

    def velocity_components(self, z, s, t, age_cond=None):
        s = self._prepare_time(s, z)
        t = self._prepare_time(t, z)
        disease_score = self._prepare_disease_score(z, age_cond)

        v_age = self.age_net(torch.cat([s, t], dim=1))

        v_dis_at_score = self._disease_velocity_for_score(z, s, t, disease_score)
        zero_score = torch.zeros_like(disease_score)
        v_dis_at_zero = self._disease_velocity_for_score(z, s, t, zero_score)
        v_disease = v_dis_at_score - v_dis_at_zero

        residual_parts = [z, s, t]
        if self.residual_uses_diagnosis:
            residual_parts.append(disease_score)
        v_residual = self.residual_net(torch.cat(residual_parts, dim=1))

        velocity = v_age + v_disease + v_residual
        return {
            "velocity": velocity,
            "age": v_age,
            "disease": v_disease,
            "disease_at_score": v_dis_at_score,
            "disease_at_zero": v_dis_at_zero,
            "residual": v_residual,
            "disease_score": disease_score,
        }

    def forward(self, z, s, t, age_cond=None):
        return self.velocity_components(z, s, t, age_cond=age_cond)["velocity"]


def build_temporal_flow(specs, latent_size, hidden_dims, age_condition_dim=0):
    flow_type = str(specs.get("FlowModelType", "standard")).strip().lower()
    if flow_type in ("standard", "temporal_mlp", "default"):
        return StandardTemporalFlowMLP(
            latent_size,
            hidden_dims,
            age_condition_dim=age_condition_dim,
        )
    if flow_type not in ("additive_velocity", "additive_model"):
        raise ValueError(f"Unknown FlowModelType='{flow_type}'.")

    return AdditiveVelocityFlow(
        latent_size=latent_size,
        flow_hidden_dims=hidden_dims,
        age_branch_hidden_dims=specs.get("AdditiveAgeBranchHiddenDims", hidden_dims),
        disease_branch_hidden_dims=specs.get("AdditiveDiseaseBranchHiddenDims", hidden_dims),
        residual_branch_hidden_dims=specs.get("AdditiveResidualBranchHiddenDims", hidden_dims),
        probe_hidden_dims=specs.get("AdditiveProbeHiddenDims", [64]),
        age_condition_dim=age_condition_dim,
        disease_uses_latent=bool(specs.get("AdditiveDiseaseUsesLatent", False)),
        residual_uses_diagnosis=bool(specs.get("AdditiveResidualUsesDiagnosis", False)),
    )
