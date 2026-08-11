"""ADNI no-MCI data contract for scan-to-scan direct-flow training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from pathlib import Path
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairRecord:
    subject_id: str
    diagnosis: str
    label_ad: int
    source_scan_id: str
    target_scan_id: str
    source_time: float
    target_time: float
    target_sdf_path: str
    source_visit_order: int
    target_visit_order: int
    observed_intermediate_times: Tuple[float, ...]
    source_mesh_volume_mm3: float = float("nan")
    target_mesh_volume_mm3: float = float("nan")

    @property
    def has_observed_intermediate(self) -> bool:
        return bool(self.observed_intermediate_times)

    @property
    def age_gap_norm(self) -> float:
        return float(self.target_time - self.source_time)


@dataclass(frozen=True)
class SequenceRecord:
    subject_id: str
    diagnosis: str
    label_ad: int
    scan_ids: Tuple[str, ...]
    times: Tuple[float, ...]
    age_years: Tuple[float, ...]
    visit_orders: Tuple[int, ...]
    sdf_paths: Tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.scan_ids)

    @property
    def step_count(self) -> int:
        return max(0, self.length - 1)


@dataclass(frozen=True)
class DirectFlowDataContract:
    metadata: pd.DataFrame
    latent_maps: Mapping[str, Mapping[str, torch.Tensor]]
    pair_records: Mapping[str, Sequence[PairRecord]]
    sequence_records: Mapping[str, Sequence[SequenceRecord]]
    report: Mapping[str, object]


def load_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    required = {
        "scan_id",
        "subject_id",
        "split",
        "visit_order",
        "continuous_age_norm",
        "continuous_age_years",
        "diagnosis",
        "label_ad",
        "sdf_npz_path",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    frame = frame.copy()
    frame["scan_id"] = frame["scan_id"].astype(str)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["visit_order"] = frame["visit_order"].astype(int)
    frame["continuous_age_norm"] = frame["continuous_age_norm"].astype(float)
    frame["continuous_age_years"] = frame["continuous_age_years"].astype(float)
    frame["label_ad"] = frame["label_ad"].astype(int)
    if set(frame["diagnosis"].unique()) != {"CN", "AD"}:
        raise ValueError(
            "Expected diagnoses exactly {'CN','AD'}, got "
            f"{sorted(frame['diagnosis'].unique())}"
        )
    if set(frame["label_ad"].unique()) != {0, 1}:
        raise ValueError(
            f"Expected label_ad values [0,1], got {sorted(frame['label_ad'].unique())}"
        )
    return frame.sort_values(
        ["split", "subject_id", "continuous_age_norm", "visit_order", "scan_id"]
    ).reset_index(drop=True)


def load_latent_map(path: str | Path, expected_dim: int) -> Dict[str, torch.Tensor]:
    path = Path(path)
    payload = np.load(path, allow_pickle=False)
    if "scan_ids" not in payload or "latents" not in payload:
        raise ValueError(f"Latent archive lacks scan_ids/latents: {path}")
    scan_ids = payload["scan_ids"]
    latents = payload["latents"]
    if latents.ndim != 2 or latents.shape[1] != int(expected_dim):
        raise ValueError(
            f"Expected latent shape [N,{expected_dim}], got {latents.shape} at {path}"
        )
    if len(scan_ids) != len(latents):
        raise ValueError(f"scan_ids/latents length mismatch at {path}")
    result: Dict[str, torch.Tensor] = {}
    for scan_id, latent in zip(scan_ids, latents):
        if isinstance(scan_id, bytes):
            scan_id = scan_id.decode("utf-8")
        key = Path(str(scan_id)).stem
        if key in result:
            raise ValueError(f"Duplicate scan ID {key!r} in {path}")
        result[key] = torch.from_numpy(np.asarray(latent, dtype=np.float32))
    return result


def build_forward_pairs(
    frame: pd.DataFrame,
    *,
    source_mode: str = "all_forward_starts",
) -> List[PairRecord]:
    """Build chronological source-target records.

    ``all_forward_starts`` keeps the historical scan-to-scan contract: every
    earlier scan can supervise every later scan.  ``first_only`` is the
    anchor/one-shot contract: only the first scan for each subject is used as a
    source, while every later scan remains a target.
    """

    normalized_mode = str(source_mode).strip().lower()
    if normalized_mode in {"anchor", "baseline", "one_shot"}:
        normalized_mode = "first_only"
    if normalized_mode not in {"all_forward_starts", "first_only"}:
        raise ValueError(
            "Pair source_mode must be all_forward_starts or first_only; "
            f"got {source_mode!r}"
        )
    records: List[PairRecord] = []
    for subject_id, subject_rows in frame.groupby("subject_id", sort=True):
        subject_rows = subject_rows.sort_values(
            ["continuous_age_norm", "visit_order", "scan_id"]
        ).reset_index(drop=True)
        times = subject_rows["continuous_age_norm"].to_numpy(dtype=float)
        if any(current <= previous for previous, current in zip(times[:-1], times[1:])):
            raise ValueError(f"Non-increasing time for subject {subject_id}")
        if normalized_mode == "first_only":
            index_pairs = (
                (0, target_index) for target_index in range(1, len(subject_rows))
            )
        else:
            index_pairs = itertools.combinations(range(len(subject_rows)), 2)
        for source_index, target_index in index_pairs:
            source = subject_rows.iloc[source_index]
            target = subject_rows.iloc[target_index]
            if str(source["diagnosis"]) != str(target["diagnosis"]):
                raise ValueError(
                    f"Diagnosis changes within subject {subject_id}; this experiment "
                    "expects a fixed CN/AD condition"
                )
            intermediate_times = tuple(
                float(subject_rows.iloc[index]["continuous_age_norm"])
                for index in range(source_index + 1, target_index)
            )
            records.append(
                PairRecord(
                    subject_id=str(subject_id),
                    diagnosis=str(source["diagnosis"]),
                    label_ad=int(source["label_ad"]),
                    source_scan_id=str(source["scan_id"]),
                    target_scan_id=str(target["scan_id"]),
                    source_time=float(source["continuous_age_norm"]),
                    target_time=float(target["continuous_age_norm"]),
                    target_sdf_path=str(target["sdf_npz_path"]),
                    source_visit_order=int(source["visit_order"]),
                    target_visit_order=int(target["visit_order"]),
                    observed_intermediate_times=intermediate_times,
                    source_mesh_volume_mm3=float(
                        source["left_mesh_volume_mm3"]
                    )
                    if "left_mesh_volume_mm3" in frame.columns
                    else float("nan"),
                    target_mesh_volume_mm3=float(
                        target["left_mesh_volume_mm3"]
                    )
                    if "left_mesh_volume_mm3" in frame.columns
                    else float("nan"),
                )
            )
    return records


def build_forward_sequences(
    frame: pd.DataFrame,
    *,
    min_length: int = 2,
    start_mode: str = "all_forward_starts",
) -> List[SequenceRecord]:
    """Build chronological subject sequences for rollout supervision.

    ``all_forward_starts`` creates one suffix trajectory from every scan that
    has at least one later scan.  For ages 70, 70.5, 71 this yields
    [70, 70.5, 71] and [70.5, 71].  This keeps the old pair dataset unchanged
    while allowing repeated application of the same direct transport operator.
    """

    if int(min_length) < 2:
        raise ValueError("Sequence min_length must be at least 2")
    normalized_mode = str(start_mode).strip().lower()
    if normalized_mode not in {"all_forward_starts", "first_only"}:
        raise ValueError(
            "Sequence start_mode must be all_forward_starts or first_only; "
            f"got {start_mode!r}"
        )
    records: List[SequenceRecord] = []
    for subject_id, subject_rows in frame.groupby("subject_id", sort=True):
        subject_rows = subject_rows.sort_values(
            ["continuous_age_norm", "visit_order", "scan_id"]
        ).reset_index(drop=True)
        if len(subject_rows) < int(min_length):
            continue
        times = subject_rows["continuous_age_norm"].to_numpy(dtype=float)
        if any(current <= previous for previous, current in zip(times[:-1], times[1:])):
            raise ValueError(f"Non-increasing time for subject {subject_id}")
        if subject_rows["diagnosis"].nunique() != 1:
            raise ValueError(
                f"Diagnosis changes within subject {subject_id}; this experiment "
                "expects a fixed CN/AD condition"
            )
        start_indices = [0] if normalized_mode == "first_only" else list(
            range(0, len(subject_rows) - int(min_length) + 1)
        )
        for start_index in start_indices:
            sequence = subject_rows.iloc[start_index:].reset_index(drop=True)
            if len(sequence) < int(min_length):
                continue
            records.append(
                SequenceRecord(
                    subject_id=str(subject_id),
                    diagnosis=str(sequence.iloc[0]["diagnosis"]),
                    label_ad=int(sequence.iloc[0]["label_ad"]),
                    scan_ids=tuple(sequence["scan_id"].astype(str).tolist()),
                    times=tuple(
                        float(value)
                        for value in sequence["continuous_age_norm"].tolist()
                    ),
                    age_years=tuple(
                        float(value)
                        for value in sequence["continuous_age_years"].tolist()
                    ),
                    visit_orders=tuple(
                        int(value)
                        for value in sequence["visit_order"].tolist()
                    ),
                    sdf_paths=tuple(
                        str(value)
                        for value in sequence["sdf_npz_path"].tolist()
                    ),
                )
            )
    return records


def _remove_nan_rows(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2 or array.shape[1] < 4:
        raise ValueError(f"Expected SDF array [N,>=4], got {array.shape}")
    return array[~np.isnan(array[:, 3])]


def _balanced_sdf_samples(
    path: str | Path,
    sample_count: int,
    *,
    seed: Optional[int],
) -> torch.Tensor:
    if int(sample_count) <= 0 or int(sample_count) % 2 != 0:
        raise ValueError("sample_count must be a positive even number")
    with np.load(path) as payload:
        positive = _remove_nan_rows(np.asarray(payload["pos"]))
        negative = _remove_nan_rows(np.asarray(payload["neg"]))
    half = int(sample_count) // 2
    rng = np.random.default_rng(seed)
    pos_indices = rng.integers(0, len(positive), size=half)
    neg_indices = rng.integers(0, len(negative), size=half)
    samples = np.concatenate(
        [positive[pos_indices], negative[neg_indices]],
        axis=0,
    )
    return torch.from_numpy(np.asarray(samples, dtype=np.float32))


class DirectFlowPairDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PairRecord],
        latent_map: Mapping[str, torch.Tensor],
        samples_per_target: int,
        *,
        deterministic: bool = False,
        deterministic_seed: int = 0,
    ) -> None:
        self.records = list(records)
        self.latent_map = latent_map
        self.samples_per_target = int(samples_per_target)
        self.deterministic = bool(deterministic)
        self.deterministic_seed = int(deterministic_seed)
        required_scan_ids = {
            scan_id
            for record in self.records
            for scan_id in (record.source_scan_id, record.target_scan_id)
        }
        missing = sorted(required_scan_ids.difference(latent_map))
        if missing:
            raise KeyError(
                f"{len(missing)} source scans lack frozen latents; first={missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _seed(self, index: int, record: PairRecord) -> Optional[int]:
        if not self.deterministic:
            return None
        digest = hashlib.sha256(
            f"{self.deterministic_seed}:{index}:{record.target_scan_id}".encode()
        ).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[index]
        if record.observed_intermediate_times:
            if self.deterministic:
                intermediate_time = record.observed_intermediate_times[
                    len(record.observed_intermediate_times) // 2
                ]
            else:
                intermediate_time = random.choice(
                    record.observed_intermediate_times
                )
            observed_mask = True
        else:
            intermediate_time = 0.5 * (
                record.source_time + record.target_time
            )
            observed_mask = False
        return {
            "source_latent": self.latent_map[record.source_scan_id].clone(),
            "target_latent": self.latent_map[record.target_scan_id].clone(),
            "source_time": torch.tensor(record.source_time, dtype=torch.float32),
            "target_time": torch.tensor(record.target_time, dtype=torch.float32),
            "condition": torch.tensor(
                [record.label_ad],
                dtype=torch.float32,
            ),
            "target_samples": _balanced_sdf_samples(
                record.target_sdf_path,
                self.samples_per_target,
                seed=self._seed(index, record),
            ),
            "observed_intermediate_time": torch.tensor(
                intermediate_time,
                dtype=torch.float32,
            ),
            "observed_intermediate_mask": torch.tensor(
                observed_mask,
                dtype=torch.bool,
            ),
            "source_mesh_volume_mm3": torch.tensor(
                record.source_mesh_volume_mm3,
                dtype=torch.float32,
            ),
            "target_mesh_volume_mm3": torch.tensor(
                record.target_mesh_volume_mm3,
                dtype=torch.float32,
            ),
            "subject_id": record.subject_id,
            "diagnosis": record.diagnosis,
            "source_scan_id": record.source_scan_id,
            "target_scan_id": record.target_scan_id,
            "source_visit_order": torch.tensor(
                record.source_visit_order,
                dtype=torch.int64,
            ),
            "target_visit_order": torch.tensor(
                record.target_visit_order,
                dtype=torch.int64,
            ),
        }


class DirectFlowSequenceDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SequenceRecord],
        latent_map: Mapping[str, torch.Tensor],
        samples_per_scan: int,
        *,
        deterministic: bool = False,
        deterministic_seed: int = 0,
    ) -> None:
        self.records = list(records)
        self.latent_map = latent_map
        self.samples_per_scan = int(samples_per_scan)
        self.deterministic = bool(deterministic)
        self.deterministic_seed = int(deterministic_seed)
        required_scan_ids = {
            scan_id
            for record in self.records
            for scan_id in record.scan_ids
        }
        missing = sorted(required_scan_ids.difference(latent_map))
        if missing:
            raise KeyError(
                f"{len(missing)} sequence scans lack frozen latents; first={missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _seed(self, index: int, record: SequenceRecord, target_index: int) -> Optional[int]:
        if not self.deterministic:
            return None
        digest = hashlib.sha256(
            (
                f"{self.deterministic_seed}:{index}:"
                f"{target_index}:{record.scan_ids[target_index]}"
            ).encode()
        ).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        latents = torch.stack(
            [self.latent_map[scan_id].clone() for scan_id in record.scan_ids],
            dim=0,
        )
        target_samples = torch.stack(
            [
                _balanced_sdf_samples(
                    record.sdf_paths[target_index],
                    self.samples_per_scan,
                    seed=self._seed(index, record, target_index),
                )
                for target_index in range(1, record.length)
            ],
            dim=0,
        )
        return {
            "subject_id": record.subject_id,
            "diagnosis": record.diagnosis,
            "scan_ids": record.scan_ids,
            "visit_orders": torch.tensor(record.visit_orders, dtype=torch.int64),
            "age_years": torch.tensor(record.age_years, dtype=torch.float32),
            "times": torch.tensor(record.times, dtype=torch.float32),
            "latents": latents,
            "condition": torch.tensor([record.label_ad], dtype=torch.float32),
            "target_samples": target_samples,
        }


def direct_flow_sequence_collate(
    items: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not items:
        raise ValueError("Cannot collate an empty sequence batch")
    batch_size = len(items)
    max_length = max(int(item["latents"].shape[0]) for item in items)  # type: ignore[index,union-attr]
    latent_dim = int(items[0]["latents"].shape[1])  # type: ignore[index,union-attr]
    sample_count = int(items[0]["target_samples"].shape[1])  # type: ignore[index,union-attr]
    sample_width = int(items[0]["target_samples"].shape[2])  # type: ignore[index,union-attr]

    latents = torch.zeros(batch_size, max_length, latent_dim, dtype=torch.float32)
    times = torch.zeros(batch_size, max_length, dtype=torch.float32)
    age_years = torch.zeros(batch_size, max_length, dtype=torch.float32)
    visit_orders = torch.zeros(batch_size, max_length, dtype=torch.int64)
    scan_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    step_mask = torch.zeros(batch_size, max_length - 1, dtype=torch.bool)
    target_samples = torch.zeros(
        batch_size,
        max_length - 1,
        sample_count,
        sample_width,
        dtype=torch.float32,
    )
    conditions = torch.zeros(batch_size, 1, dtype=torch.float32)
    subject_ids: List[str] = []
    diagnoses: List[str] = []
    scan_ids: List[Tuple[str, ...]] = []

    for row, item in enumerate(items):
        item_latents = item["latents"]  # type: ignore[index]
        item_times = item["times"]  # type: ignore[index]
        item_age_years = item["age_years"]  # type: ignore[index]
        item_visit_orders = item["visit_orders"]  # type: ignore[index]
        item_samples = item["target_samples"]  # type: ignore[index]
        length = int(item_latents.shape[0])
        latents[row, :length] = item_latents
        times[row, :length] = item_times
        age_years[row, :length] = item_age_years
        visit_orders[row, :length] = item_visit_orders
        if length < max_length:
            times[row, length:] = item_times[-1]
            age_years[row, length:] = item_age_years[-1]
            visit_orders[row, length:] = item_visit_orders[-1]
        scan_mask[row, :length] = True
        step_mask[row, : length - 1] = True
        target_samples[row, : length - 1] = item_samples
        conditions[row] = item["condition"]  # type: ignore[index]
        subject_ids.append(str(item["subject_id"]))  # type: ignore[index]
        diagnoses.append(str(item["diagnosis"]))  # type: ignore[index]
        scan_ids.append(tuple(item["scan_ids"]))  # type: ignore[index]

    return {
        "subject_id": subject_ids,
        "diagnosis": diagnoses,
        "scan_ids": scan_ids,
        "visit_orders": visit_orders,
        "age_years": age_years,
        "times": times,
        "latents": latents,
        "condition": conditions,
        "target_samples": target_samples,
        "scan_mask": scan_mask,
        "step_mask": step_mask,
    }


def build_contract(
    metadata_path: str | Path,
    latent_paths: Mapping[str, str | Path],
    expected_latent_dim: int,
    *,
    pair_source_mode: str = "all_forward_starts",
    sequence_min_length: int = 2,
    sequence_start_mode: str = "all_forward_starts",
    allow_extra_latents: bool = False,
) -> DirectFlowDataContract:
    metadata = load_metadata(metadata_path)
    normalized_pair_source_mode = str(pair_source_mode).strip().lower()
    if normalized_pair_source_mode in {"anchor", "baseline", "one_shot"}:
        normalized_pair_source_mode = "first_only"
    latent_maps: Dict[str, Mapping[str, torch.Tensor]] = {}
    pair_records: Dict[str, Sequence[PairRecord]] = {}
    sequence_records: Dict[str, Sequence[SequenceRecord]] = {}
    split_report: Dict[str, object] = {}
    subject_split_membership: Dict[str, set] = {}

    for subject_id, group in metadata.groupby("subject_id"):
        subject_split_membership[str(subject_id)] = set(group["split"].astype(str))
    crossing = {
        subject_id: sorted(splits)
        for subject_id, splits in subject_split_membership.items()
        if len(splits) != 1
    }
    if crossing:
        first_subject = next(iter(crossing))
        raise ValueError(
            f"Found subjects crossing splits; first={first_subject}:{crossing[first_subject]}"
        )

    for split in ("train", "val", "test"):
        if split not in latent_paths:
            raise KeyError(f"Missing latent archive path for split {split}")
        split_frame = metadata.loc[metadata["split"] == split].copy()
        latent_map = load_latent_map(latent_paths[split], expected_latent_dim)
        scan_ids = set(split_frame["scan_id"].astype(str))
        missing = sorted(scan_ids.difference(latent_map))
        extras = sorted(set(latent_map).difference(scan_ids))
        if missing or (extras and not bool(allow_extra_latents)):
            raise ValueError(
                f"Latent/metadata mismatch for {split}: "
                f"missing={len(missing)}, extras={len(extras)}"
            )
        bad_sdf = [
            str(path)
            for path in split_frame["sdf_npz_path"]
            if not Path(str(path)).is_file()
        ]
        if bad_sdf:
            raise FileNotFoundError(
                f"{len(bad_sdf)} missing SDF files for {split}; first={bad_sdf[0]}"
            )
        pairs = build_forward_pairs(
            split_frame,
            source_mode=normalized_pair_source_mode,
        )
        sequences = build_forward_sequences(
            split_frame,
            min_length=int(sequence_min_length),
            start_mode=sequence_start_mode,
        )
        latent_maps[split] = latent_map
        pair_records[split] = pairs
        sequence_records[split] = sequences
        split_report[split] = {
            "scans": int(len(split_frame)),
            "subjects": int(split_frame["subject_id"].nunique()),
            "pairs": int(len(pairs)),
            "pair_source_mode": normalized_pair_source_mode,
            "pairs_with_observed_intermediate": int(
                sum(record.has_observed_intermediate for record in pairs)
            ),
            "sequences": int(len(sequences)),
            "sequence_steps": int(sum(record.step_count for record in sequences)),
            "diagnosis_scans": {
                key: int(value)
                for key, value in split_frame["diagnosis"].value_counts().items()
            },
        }

    report = {
        "status": "pass",
        "latent_dim": int(expected_latent_dim),
        "pair_source_mode": normalized_pair_source_mode,
        "allow_extra_latents": bool(allow_extra_latents),
        "splits": split_report,
        "subject_split_isolation": True,
        "loss_scope": [
            "real_target_sdf_prediction",
            "observed_intermediate_latent_consistency",
            "virtual_intermediate_latent_consistency",
        ],
    }
    return DirectFlowDataContract(
        metadata=metadata,
        latent_maps=latent_maps,
        pair_records=pair_records,
        sequence_records=sequence_records,
        report=report,
    )
