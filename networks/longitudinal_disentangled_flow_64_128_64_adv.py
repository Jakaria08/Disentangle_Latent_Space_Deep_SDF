import math

import torch


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def gradient_reverse(x, lambd=1.0):
    return _GradientReverse.apply(x, float(lambd))


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


def _build_mlp(input_dim, hidden_dims, output_dim):
    if len(hidden_dims) == 0:
        raise ValueError("Flow hidden dims must be non-empty.")
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*layers)


class DisentangledTemporalFlowMLP(torch.nn.Module):
    """Two-time latent velocity with projected age, disease-gated, and residual blocks."""

    def __init__(
        self,
        latent_size,
        hidden_dims,
        age_condition_dim=1,
        age_dim=16,
        disease_dim=16,
        residual_dim=None,
        age_embedding_scale=0.02,
        disease_uses_latent=True,
        residual_uses_diagnosis=False,
        adv_probe_hidden_dims=None,
        diagonal_eps=1e-6,
    ):
        super().__init__()
        self.latent_size = int(latent_size)
        self.age_condition_dim = max(0, int(age_condition_dim))
        self.age_dim = int(age_dim)
        self.disease_dim = int(disease_dim)
        self.residual_dim = (
            self.latent_size - self.age_dim - self.disease_dim
            if residual_dim is None
            else int(residual_dim)
        )
        self.age_embedding_scale = float(age_embedding_scale)
        self.disease_uses_latent = bool(disease_uses_latent)
        self.residual_uses_diagnosis = bool(residual_uses_diagnosis)
        self.diagonal_eps = float(diagonal_eps)
        self.adv_probe_hidden_dims = (
            [64] if adv_probe_hidden_dims is None else list(adv_probe_hidden_dims)
        )

        if self.age_dim != 16:
            if self.age_dim <= 0:
                raise ValueError("VelocityAgeDim must be positive.")
        if self.age_dim + self.disease_dim + self.residual_dim != self.latent_size:
            raise ValueError(
                "Velocity block dimensions must sum to latent size: "
                f"{self.age_dim}+{self.disease_dim}+{self.residual_dim} "
                f"!= {self.latent_size}."
            )
        if self.disease_dim <= 0 or self.residual_dim <= 0:
            raise ValueError("Disease and residual velocity dimensions must be positive.")

        disease_input_dim = 2 + (self.latent_size if self.disease_uses_latent else 0)
        residual_input_dim = self.latent_size + 2
        if self.residual_uses_diagnosis:
            residual_input_dim += 1

        self.age_projection = torch.nn.Linear(16, self.age_dim, bias=False)
        with torch.no_grad():
            self.age_projection.weight.zero_()
            eye_dim = min(16, self.age_dim)
            self.age_projection.weight[:eye_dim, :eye_dim] = torch.eye(eye_dim)

        self.disease_net = _build_mlp(disease_input_dim, hidden_dims, self.disease_dim)
        self.residual_net = _build_mlp(residual_input_dim, hidden_dims, self.residual_dim)
        self.age_leakage_classifier = _build_mlp(
            self.age_dim, self.adv_probe_hidden_dims, 1
        )
        self.residual_leakage_classifier = _build_mlp(
            self.residual_dim, self.adv_probe_hidden_dims, 1
        )
        self.disease_classifier = _build_mlp(
            self.disease_dim, self.adv_probe_hidden_dims, 1
        )

    @property
    def is_disentangled_velocity_flow(self):
        return True

    def _prepare_time(self, value, z):
        if value.dim() == 1:
            value = value.unsqueeze(1)
        return value.to(device=z.device, dtype=z.dtype)

    def _prepare_diagnosis(self, z, age_cond):
        if self.age_condition_dim <= 0 or age_cond is None:
            return torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        diagnosis = age_cond
        if diagnosis.dim() == 1:
            diagnosis = diagnosis.unsqueeze(1)
        diagnosis = diagnosis.to(device=z.device, dtype=z.dtype)
        if diagnosis.shape[1] < 1:
            raise ValueError("Diagnosis condition must have at least one column.")
        if diagnosis.shape[0] == 1 and z.shape[0] != 1:
            diagnosis = diagnosis.repeat(z.shape[0], 1)
        if diagnosis.shape[0] != z.shape[0]:
            raise ValueError(
                f"Diagnosis batch size mismatch: got {diagnosis.shape[0]}, "
                f"expected {z.shape[0]}."
            )
        return diagnosis[:, 0:1]

    def age_embedding(self, age):
        pi = math.pi
        terms = [
            age,
            age.pow(2),
            age.pow(3),
        ]
        for freq in range(1, 7):
            terms.append(torch.sin(freq * pi * age))
            terms.append(torch.cos(freq * pi * age))
        terms.append(torch.ones_like(age))
        return self.age_embedding_scale * torch.cat(terms, dim=1)

    def age_embedding_derivative(self, age):
        pi = math.pi
        terms = [
            torch.ones_like(age),
            2.0 * age,
            3.0 * age.pow(2),
        ]
        for freq in range(1, 7):
            scale = float(freq) * pi
            terms.append(scale * torch.cos(freq * pi * age))
            terms.append(-scale * torch.sin(freq * pi * age))
        terms.append(torch.zeros_like(age))
        return self.age_embedding_scale * torch.cat(terms, dim=1)

    def age_velocity(self, s, t):
        dt = t - s
        midpoint = 0.5 * (s + t)
        diff = self.age_embedding(t) - self.age_embedding(s)
        safe_dt = torch.where(
            dt.abs() > self.diagonal_eps,
            dt,
            torch.ones_like(dt),
        )
        divided = diff / safe_dt
        diagonal = self.age_embedding_derivative(midpoint)
        basis_velocity = torch.where((dt.abs() > self.diagonal_eps), divided, diagonal)
        return self.age_projection(basis_velocity)

    def velocity_components(self, z, s, t, age_cond=None):
        s = self._prepare_time(s, z)
        t = self._prepare_time(t, z)
        diagnosis = self._prepare_diagnosis(z, age_cond)

        v_age = self.age_velocity(s, t)

        disease_parts = [s, t]
        if self.disease_uses_latent:
            disease_parts.insert(0, z)
        disease_input = torch.cat(disease_parts, dim=1)
        v_dis_raw = self.disease_net(disease_input)
        v_dis = diagnosis * v_dis_raw

        residual_parts = [z, s, t]
        if self.residual_uses_diagnosis:
            residual_parts.append(diagnosis)
        residual_input = torch.cat(residual_parts, dim=1)
        v_res = self.residual_net(residual_input)

        velocity = torch.cat([v_age, v_dis, v_res], dim=1)
        return {
            "velocity": velocity,
            "age": v_age,
            "disease_raw": v_dis_raw,
            "disease": v_dis,
            "residual": v_res,
            "diagnosis": diagnosis,
        }

    def age_leakage_logits(self, v_age, grl_lambda=1.0):
        return self.age_leakage_classifier(gradient_reverse(v_age, grl_lambda))

    def residual_leakage_logits(self, v_residual, grl_lambda=1.0):
        return self.residual_leakage_classifier(
            gradient_reverse(v_residual, grl_lambda)
        )

    def disease_logits(self, v_disease_raw):
        return self.disease_classifier(v_disease_raw)

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
    if flow_type not in ("disentangled_velocity", "disentangled_flow"):
        raise ValueError(f"Unknown FlowModelType='{flow_type}'.")

    age_dim = int(specs.get("VelocityAgeDim", 16))
    disease_dim = int(specs.get("VelocityDiseaseDim", 16))
    residual_dim = int(
        specs.get(
            "VelocityResidualDim",
            int(latent_size) - age_dim - disease_dim,
        )
    )
    return DisentangledTemporalFlowMLP(
        latent_size=latent_size,
        hidden_dims=hidden_dims,
        age_condition_dim=age_condition_dim,
        age_dim=age_dim,
        disease_dim=disease_dim,
        residual_dim=residual_dim,
        age_embedding_scale=float(specs.get("VelocityAgeEmbeddingScale", 0.02)),
        disease_uses_latent=bool(specs.get("VelocityDiseaseUsesLatent", True)),
        residual_uses_diagnosis=bool(specs.get("VelocityResidualUsesDiagnosis", False)),
        adv_probe_hidden_dims=specs.get("VelocityAdvProbeHiddenDims", [64]),
        diagonal_eps=float(specs.get("VelocityDiagonalEps", 1e-6)),
    )
