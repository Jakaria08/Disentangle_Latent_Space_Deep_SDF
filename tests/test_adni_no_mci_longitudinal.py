import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI"
    / "longitudinal_age_disease_conditioned_cocycle_shape_multiple_pairs_real_pair_shape_reconstruction"
)
BRIDGE_DIR = EXPERIMENT_DIR / "pretrained_task2_deepsdf"
LABELS_PATH = EXPERIMENT_DIR / "metadata" / "adni_no_mci_longitudinal_labels.pt"
RECORDS_PATH = EXPERIMENT_DIR / "metadata" / "adni_no_mci_longitudinal_records.csv"
VALIDATION_REPORT_PATH = EXPERIMENT_DIR / "metadata" / "input_validation_report.json"
TRAIN_SPLIT_PATH = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI"
    / "brainode_comparison_task1_manifest_original"
    / "splits"
    / "train_clean.json"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import deep_sdf.workspace as ws
import train_deep_sdf_longitudinal as longitudinal
import adni_no_mci_longitudinal_model_helpers as helpers


EXPECTED_SCAN_COUNTS = {"train": 617, "val": 39, "test": 71}
EXPECTED_SUBJECT_COUNTS = {"train": 207, "val": 13, "test": 24}


@lru_cache(maxsize=1)
def records_df():
    return pd.read_csv(RECORDS_PATH).sort_values(["split", "subject_id", "visit_order"]).reset_index(
        drop=True
    )


@lru_cache(maxsize=1)
def labels_payload():
    return torch.load(LABELS_PATH, map_location="cpu")


@lru_cache(maxsize=1)
def experiment_specs():
    return ws.load_experiment_specifications(str(EXPERIMENT_DIR))


@lru_cache(maxsize=1)
def bridge_specs():
    return ws.load_experiment_specifications(str(BRIDGE_DIR))


@lru_cache(maxsize=1)
def validation_report():
    return json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def smoke_bundle():
    exp_specs = experiment_specs()
    bridge = bridge_specs()
    arch = __import__("networks." + bridge["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(bridge["CodeLength"], **bridge["NetworkSpecs"])
    epoch = longitudinal.load_pretrained_decoder(decoder, str(BRIDGE_DIR), "best")
    flow = longitudinal.TemporalFlowMLP(
        exp_specs["CodeLength"],
        exp_specs["FlowHiddenDims"],
        age_condition_dim=int(exp_specs.get("AgeConditionDim", 0) or 0),
    )
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.zero_()
    decoder.eval()
    flow.eval()
    metadata, labels = helpers.load_metadata(EXPERIMENT_DIR)
    return helpers.LoadedModel(
        experiment_dir=EXPERIMENT_DIR,
        specs=exp_specs,
        metadata=metadata,
        labels=labels,
        device=torch.device("cpu"),
        checkpoint="bridge_decoder_zero_flow",
        checkpoint_epoch=int(epoch) if epoch is not None else -1,
        decoder=decoder,
        flow=flow,
        train_latents=None,
        align_mode=str(exp_specs.get("EvalChamferAlignMode", helpers.DEFAULT_ALIGN_MODE)),
        align_iters=int(exp_specs.get("EvalChamferAlignIters", helpers.DEFAULT_ALIGN_ITERS)),
        align_trim_quantile=float(
            exp_specs.get(
                "EvalChamferAlignTrimQuantile",
                helpers.DEFAULT_ALIGN_TRIM_QUANTILE,
            )
        ),
    )


def test_continuous_age_construction():
    frame = records_df()
    expected_years = frame["baseline_age_years"] + frame["months_from_baseline"] / 12.0
    expected_norm = (frame["continuous_age_years"] - 57.0) / (91.0 - 57.0)
    assert np.allclose(frame["continuous_age_years"], expected_years)
    assert np.allclose(frame["continuous_age_norm"], expected_norm)


def test_no_duplicate_adjacent_times():
    frame = records_df()
    for _, group in frame.groupby("subject_id", sort=True):
        times = group.sort_values("visit_order")["continuous_age_norm"].to_numpy(dtype=float)
        assert np.all(np.diff(times) > 0.0)


def test_scan_to_subject_lookup_for_adni_names():
    scan_subject_map = longitudinal._load_scan_subject_map_from_labels(
        str(LABELS_PATH), "subject_id"
    )
    frame = records_df()
    assert len(scan_subject_map) == len(frame)
    assert scan_subject_map[frame.iloc[0]["scan_id"]] == frame.iloc[0]["subject_id"]
    assert scan_subject_map[frame.iloc[len(frame) // 2]["scan_id"]] == frame.iloc[len(frame) // 2][
        "subject_id"
    ]
    assert scan_subject_map[frame.iloc[-1]["scan_id"]] == frame.iloc[-1]["subject_id"]


def test_cn_ad_encoding():
    frame = records_df()
    assert set(frame["diagnosis"].unique()) == {"CN", "AD"}
    assert set(frame["label_ad"].unique()) == {0, 1}
    mapped = frame["diagnosis"].map({"CN": 0, "AD": 1})
    assert np.array_equal(mapped.to_numpy(dtype=int), frame["label_ad"].to_numpy(dtype=int))


def test_split_isolation():
    frame = records_df()
    assert frame["split"].value_counts().to_dict() == EXPECTED_SCAN_COUNTS
    assert frame.groupby("split")["subject_id"].nunique().to_dict() == EXPECTED_SUBJECT_COUNTS
    subject_splits = frame.groupby("subject_id")["split"].nunique()
    assert int(subject_splits.max()) == 1


def test_task2_latent_ordering():
    spec = bridge_specs()
    task1_train_split = json.loads(TRAIN_SPLIT_PATH.read_text(encoding="utf-8"))
    bridge_train_split = json.loads((BRIDGE_DIR / spec["TrainSplit"]).read_text(encoding="utf-8"))
    assert bridge_train_split == task1_train_split

    latent_payload = torch.load(BRIDGE_DIR / "LatentCodes" / "best.pth", map_location="cpu")
    latent_weight = latent_payload["latent_codes"]["weight"]
    source_train_scan_ids = latent_payload["source_train_scan_ids"]
    npz = np.load(spec["SourceTask2TrainLatents"])
    npz_scan_ids = npz["scan_ids"].tolist()
    npz_latents = torch.from_numpy(npz["latents"]).float()

    assert latent_weight.shape == (617, 256)
    assert source_train_scan_ids == [Path(path).stem for path in task1_train_split]
    assert source_train_scan_ids == npz_scan_ids
    assert torch.allclose(latent_weight[:8], npz_latents[:8])


def test_decoder_bridge_loading():
    bridge = bridge_specs()
    arch = __import__("networks." + bridge["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(bridge["CodeLength"], **bridge["NetworkSpecs"])
    epoch = longitudinal.load_pretrained_decoder(decoder, str(BRIDGE_DIR), "best")
    first_param = next(decoder.parameters())
    assert epoch is not None
    assert first_param.shape[0] > 0
    assert torch.isfinite(first_param).all()


def test_variable_two_three_scan_subject_initialization():
    frame = records_df()
    grouped = helpers.grouped_subject_rows(frame)
    two_scan_subject = next(sid for sid, group in grouped.items() if len(group) == 2)
    three_scan_subject = next(sid for sid, group in grouped.items() if len(group) == 3)
    metadata, _ = helpers.load_metadata(EXPERIMENT_DIR)

    obs_two = helpers.build_observations_from_scan_ids(
        metadata, grouped[two_scan_subject]["scan_id"].tolist(), seed=0
    )
    obs_three = helpers.build_observations_from_scan_ids(
        metadata, grouped[three_scan_subject]["scan_id"].tolist(), seed=0
    )

    assert len(obs_two) == 2
    assert len(obs_three) == 3
    assert obs_two[0]["time"] < obs_two[1]["time"]
    assert obs_three[0]["time"] < obs_three[1]["time"] < obs_three[2]["time"]


def test_normalized_age_to_year_velocity_conversion():
    assert helpers.forecast_time_delta_to_years(1.0 / 34.0) == pytest.approx(1.0)
    converted = helpers.yearly_speed_from_normalized_speed(np.array([34.0, -17.0]))
    assert np.allclose(converted, np.array([1.0, -0.5]))


def test_synthetic_forward_backward_cocycle_closure():
    flow = longitudinal.TemporalFlowMLP(256, [32], age_condition_dim=1)
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.zero_()
    z = torch.randn(1, 256)
    s = torch.tensor([[0.1]], dtype=torch.float32)
    t = torch.tensor([[0.7]], dtype=torch.float32)
    cond = torch.tensor([[1.0]], dtype=torch.float32)
    z_t = longitudinal.apply_temporal_flow(flow, z, s, t, age_cond=cond)
    z_back = longitudinal.apply_temporal_flow(flow, z_t, t, s, age_cond=cond)
    assert torch.allclose(z, z_t)
    assert torch.allclose(z, z_back)


def test_small_cpu_one_shot_smoke():
    bundle = smoke_bundle()
    report = validation_report()
    assert report["status"] == "pass"

    frame = records_df()
    subject_rows = (
        frame.loc[frame["split"] == "test"]
        .groupby("subject_id", sort=True)
        .filter(lambda group: len(group) >= 2)
        .sort_values(["subject_id", "visit_order"])
    )
    subject_id = subject_rows.iloc[0]["subject_id"]
    subject_rows = subject_rows.loc[subject_rows["subject_id"] == subject_id].reset_index(drop=True)

    anchor, loss_hist, observations = helpers.fit_subject_anchor(
        bundle,
        subject_rows.iloc[:2]["scan_id"].tolist(),
        seed=3,
        num_iterations=2,
        num_samples=128,
        lr=1e-2,
        init_std=1e-2,
    )
    baseline_time = helpers.anchor_baseline_time(observations)
    target_row = subject_rows.iloc[1]
    latent = helpers.transport_direct(
        bundle,
        anchor,
        baseline_time=baseline_time,
        target_time=float(target_row["continuous_age_norm"]),
        target_label_ad=int(target_row["label_ad"]),
    )
    vertices = np.asarray(helpers.load_mesh(target_row["mesh_path"]).vertices[:16], dtype=np.float32)
    speed = helpers.implicit_surface_normal_velocity(
        bundle,
        latent,
        current_time=float(target_row["continuous_age_norm"]),
        label_ad=int(target_row["label_ad"]),
        query_points=vertices,
        eps=1e-4,
        chunk_size=8,
        yearly=True,
    )

    assert anchor.shape == (1, 256)
    assert len(loss_hist) == 2
    assert latent.shape == (1, 256)
    assert speed.shape == (16,)
    assert np.isfinite(speed).all()
