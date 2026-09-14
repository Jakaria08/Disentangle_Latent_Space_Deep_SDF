#!/usr/bin/env python3
"""Build ADNI-format clinical manifests for the AIBL, OASIS-3, and CALSNIC SynthSeg cohorts.

``scripts/adni_synthseg_minimal_correspondence.py`` consumes an ADNI-style clinical
CSV keyed by ``RID`` and ``VISCODE`` such that ``scan_id = RID_VISCODE`` matches the
SynthSeg segmentation folder name.  ADNI and AIBL already ship that layout; OASIS-3
and CALSNIC do not.  This script is the read-only adapter that produces it.

It never edits a source file.  For each dataset it writes

* ``<dataset>_clinical_pipeline.csv`` - the pipeline-format clinical table, and
* ``<dataset>_clinical_audit.json``   - row counts, match rates, and every drop reason,

so the conversion is auditable before any mesh is generated.

Cohort-specific notes
---------------------

AIBL
    ``ClinicalInfo.csv`` is already ADNI-format.  ``Month.bl`` is preferred over the
    visit code because it records the realised, not nominal, month of the visit.

OASIS-3
    Visit codes are ``V1``, ``V2``, ... and carry no date.  The index is the rank of
    that session's day among the subject's unique *imaging session days* across all
    modalities (MR and PET together), as listed in the OASIS-3 file inventory.  This
    rule was validated against the segmented scans: 2159 of 2162 land on a day that
    has a T1w MR session, whereas ranking MR sessions alone matches only 1748.  A
    scan whose mapped day is not a T1w MR day is dropped rather than guessed at.
    Diagnosis comes from the nearest UDS-B4 CDR visit within one year of the scan.

CALSNIC
    Scan IDs embed subject, protocol, and visit (``CALSNIC2_EDM_C087_T1w10_V1``).
    ``RID`` is everything before the final ``_V<n>`` so that ``RID_VISCODE`` rebuilds
    the folder name exactly.  The study is ALS, not Alzheimer's disease, so the
    diagnosis columns carry ``Control``/``ALS``/... rather than CN/MCI/AD; select
    ``--cohort-filter all`` downstream.  Age is recorded only at the first visit, so
    later ages are derived from the MRI dates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DAYS_PER_MONTH = 30.4375
DAYS_PER_YEAR = 365.25
CDR_WINDOW_DAYS = 365

# Columns every emitted table must carry for the mesh pipeline and the QC loader.
PIPELINE_COLUMNS = [
    "RID",
    "VISCODE",
    "DX",
    "DX.bl",
    "AGE",
    "PTGENDER",
    "month_from_viscode",
    "visit_dx_3class",
    "baseline_dx_3class",
    "cohort_group",
]

DEFAULTS: dict[str, dict[str, Path]] = {
    "aibl": {
        "segmentations": Path("/mnt/bulk10tb/AIBL/aibl_synthseg/segmentations"),
        "clinical": Path("/mnt/bulk10tb/AIBL/AIBL-STRATIFIED-MRI-938/ClinicalInfo.csv"),
        "output_dir": Path("/mnt/bulk10tb/AIBL/cohort"),
    },
    "oasis": {
        "segmentations": Path("/mnt/bulk10tb/OASIS_3_Long/oasis3_synthseg/segmentations"),
        "inventory": REPO_ROOT / "data" / "OASIS_3_long" / "oasis_info.txt",
        "clinical_zip": Path("/mnt/bulk10tb/OASIS_3_Long/OASIS3_data_files.zip"),
        "cohort_zip": Path("/mnt/bulk10tb/OASIS_3_Long/OASIS_cohort_files.zip"),
        "output_dir": Path("/mnt/bulk10tb/OASIS_3_Long/cohort"),
    },
    "calsnic": {
        "segmentations": Path("/mnt/bulk10tb/CALSNIC_Long/calsnic_synthseg/segmentations"),
        "clinical": Path("/home/jakaria/CALSNIC/Final_Data_sheet_April2025.csv"),
        "output_dir": Path("/mnt/bulk10tb/CALSNIC_Long/cohort"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULTS))
    parser.add_argument("--segmentation-root", type=Path, default=None)
    parser.add_argument("--clinical", type=Path, default=None, help="AIBL/CALSNIC clinical table.")
    parser.add_argument("--inventory", type=Path, default=None, help="OASIS-3 file inventory (oasis_info.txt).")
    parser.add_argument("--clinical-zip", type=Path, default=None, help="OASIS3_data_files.zip")
    parser.add_argument("--cohort-zip", type=Path, default=None, help="OASIS_cohort_files.zip")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def discovered_scans(segmentation_root: Path) -> pd.DataFrame:
    """One row per SynthSeg segmentation actually present on disk."""
    paths = sorted(segmentation_root.glob("*/*.synthseg.mgz"))
    if not paths:
        raise FileNotFoundError(f"No SynthSeg segmentations under {segmentation_root}")
    frame = pd.DataFrame({"scan_id": [path.parent.name for path in paths]})
    frame[["RID", "VISCODE"]] = frame["scan_id"].str.rsplit("_", n=1, expand=True)
    return frame.sort_values("scan_id", kind="stable").reset_index(drop=True)


def _finalise(frame: pd.DataFrame, scans: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restrict to scans that exist, verify the scan-id contract, and audit the result."""
    frame = frame.copy()
    frame["scan_id"] = frame["RID"].astype(str) + "_" + frame["VISCODE"].astype(str)
    present = set(scans["scan_id"])
    matched = frame.loc[frame["scan_id"].isin(present)].copy()
    missing_clinical = sorted(present.difference(set(frame["scan_id"])))
    duplicated = matched.loc[matched.duplicated("scan_id", keep=False), "scan_id"].unique().tolist()
    if duplicated:
        raise RuntimeError(f"{dataset}: duplicate clinical rows for scans {duplicated[:5]}")
    missing_columns = sorted(set(PIPELINE_COLUMNS).difference(matched.columns))
    if missing_columns:
        raise RuntimeError(f"{dataset}: emitted table is missing columns {missing_columns}")
    ordered = [*PIPELINE_COLUMNS, *[c for c in matched.columns if c not in PIPELINE_COLUMNS]]
    matched = matched.loc[:, ordered].sort_values(["RID", "month_from_viscode", "VISCODE"], kind="stable")
    visits = matched.groupby("RID")["scan_id"].nunique()
    audit = {
        "dataset": dataset,
        "segmentations_on_disk": int(len(scans)),
        "clinical_rows_in": int(len(frame)),
        "emitted_rows": int(len(matched)),
        "emitted_subjects": int(matched["RID"].nunique()),
        "segmentations_without_clinical_row": len(missing_clinical),
        "segmentations_without_clinical_examples": missing_clinical[:10],
        "subjects_with_at_least_two_visits": int((visits >= 2).sum()),
        "scans_in_multi_visit_subjects": int(visits.loc[visits >= 2].sum()),
        "visit_dx_3class_counts": {str(k): int(v) for k, v in matched["visit_dx_3class"].value_counts(dropna=False).items()},
        "baseline_dx_3class_counts": {
            str(k): int(v)
            for k, v in matched.drop_duplicates("RID")["baseline_dx_3class"].value_counts(dropna=False).items()
        },
        "missing_age_rows": int(pd.to_numeric(matched["AGE"], errors="coerce").isna().sum()),
        "missing_month_rows": int(pd.to_numeric(matched["month_from_viscode"], errors="coerce").isna().sum()),
    }
    return matched.reset_index(drop=True), audit


def build_aibl(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    scans = discovered_scans(args.segmentation_root)
    clinical = pd.read_csv(args.clinical, dtype={"RID": str, "VISCODE": str}, keep_default_na=False)
    frame = clinical.copy()
    frame["RID"] = frame["RID"].str.strip()
    frame["VISCODE"] = frame["VISCODE"].str.strip()
    # Month.bl is the realised month of the visit; the visit code is only nominal.
    frame["month_from_viscode"] = pd.to_numeric(frame["Month.bl"].replace({"": None, "NA": None}), errors="coerce")
    frame["visit_dx_3class"] = frame["DX"].map({"NL": "CN", "MCI": "MCI", "Dementia": "AD"})
    frame["baseline_dx_3class"] = frame["DX.bl"].map({"CN": "CN", "NL": "CN", "MCI": "MCI", "AD": "AD"})
    frame["cohort_group"] = frame["AUX.STRATIFICATION"].replace({"": None})
    return _finalise(frame, scans, "aibl")


def _oasis_clinical_tables(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Read the OASIS CSVs straight out of their zips; the archives are left untouched."""
    import zipfile

    wanted = {
        "demographics": "OASIS3_demographics.csv",
        "cdr": "OASIS3_UDSb4_cdr.csv",
        "healthy": "OASIS3_unchanged_CDR_cognitively_healthy.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for archive in (args.clinical_zip, args.cohort_zip):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                for key, name in wanted.items():
                    if member.endswith(name):
                        with handle.open(member) as stream:
                            tables[key] = pd.read_csv(stream, low_memory=False)
    missing = sorted(set(wanted).difference(tables))
    if missing:
        raise RuntimeError(f"OASIS clinical tables not found in the archives: {missing}")
    return tables


def build_oasis(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    scans = discovered_scans(args.segmentation_root)
    inventory = pd.read_csv(args.inventory, header=None, names=["path"])
    parts = inventory["path"].str.split("/", expand=True)
    inventory["subject"] = parts[1]
    inventory["session"] = parts[2]
    inventory["modality"] = inventory["session"].str.extract(r"_([A-Z0-9]+)_d\d+$")[0]
    inventory["day"] = pd.to_numeric(inventory["session"].str.extract(r"_d(\d+)$")[0], errors="coerce")
    inventory = inventory.dropna(subset=["day"]).astype({"day": int})
    inventory["is_t1w"] = inventory["path"].str.contains("_T1w.nii.gz", regex=False)

    session_days = inventory.groupby("subject")["day"].apply(lambda s: sorted(set(s)))
    t1w_days = (
        inventory.loc[inventory["is_t1w"] & inventory["modality"].eq("MR")]
        .groupby("subject")["day"]
        .apply(lambda s: set(s))
    )

    frame = scans.copy()
    frame["visit_index"] = pd.to_numeric(frame["VISCODE"].str.extract(r"^V(\d+)$")[0], errors="coerce")

    def session_day(row: pd.Series) -> float:
        days = session_days.get(row["RID"])
        index = row["visit_index"]
        if days is None or pd.isna(index) or int(index) > len(days):
            return np.nan
        return float(days[int(index) - 1])

    frame["days_from_entry"] = frame.apply(session_day, axis=1)
    frame["mapped_day_has_t1w_mr"] = frame.apply(
        lambda row: bool(row["RID"] in t1w_days and row["days_from_entry"] in t1w_days.get(row["RID"], set())),
        axis=1,
    )
    unmapped = frame.loc[~frame["mapped_day_has_t1w_mr"], "scan_id"].tolist()
    frame = frame.loc[frame["mapped_day_has_t1w_mr"]].copy()

    tables = _oasis_clinical_tables(args)
    cdr = tables["cdr"].copy()
    cdr["day"] = pd.to_numeric(cdr["days_to_visit"], errors="coerce")
    cdr = cdr.dropna(subset=["day"])
    healthy = set(tables["healthy"]["OASIS3_id"])
    ad_subjects = set(cdr.loc[cdr["dx1"].astype(str).str.contains("AD Dementia", case=False, na=False), "OASISID"])

    by_subject = {subject: group for subject, group in cdr.groupby("OASISID")}

    def nearest_cdr(row: pd.Series) -> tuple[float, str, float]:
        group = by_subject.get(row["RID"])
        if group is None:
            return np.nan, "", np.nan
        gap = (group["day"] - row["days_from_entry"]).abs()
        index = gap.idxmin()
        if gap.loc[index] > CDR_WINDOW_DAYS:
            return np.nan, "", float(gap.loc[index])
        return float(group.loc[index, "CDRTOT"]), str(group.loc[index, "dx1"]), float(gap.loc[index])

    resolved = frame.apply(nearest_cdr, axis=1, result_type="expand")
    frame[["CDRTOT", "clinical_dx1", "cdr_gap_days"]] = resolved

    frame["visit_dx_3class"] = np.select(
        [frame["CDRTOT"].eq(0), frame["CDRTOT"].eq(0.5), frame["CDRTOT"].ge(1)],
        ["CN", "MCI", "AD"],
        default=None,
    )
    frame["DX"] = frame["visit_dx_3class"].map({"CN": "NL", "MCI": "MCI", "AD": "Dementia"})
    first = frame.sort_values("days_from_entry").drop_duplicates("RID")[["RID", "visit_dx_3class"]]
    first = first.rename(columns={"visit_dx_3class": "baseline_dx_3class"})
    frame = frame.merge(first, on="RID", how="left")
    frame["DX.bl"] = frame["baseline_dx_3class"]

    demographics = tables["demographics"]
    frame = frame.merge(
        demographics[["OASISID", "AgeatEntry", "GENDER", "EDUC", "APOE"]],
        left_on="RID",
        right_on="OASISID",
        how="left",
    ).drop(columns="OASISID")
    frame["AGE"] = frame["AgeatEntry"] + frame["days_from_entry"] / DAYS_PER_YEAR
    frame["PTGENDER"] = frame["GENDER"].map({1: "Male", 2: "Female"})
    frame["month_from_viscode"] = frame["days_from_entry"] / DAYS_PER_MONTH
    frame["cohort_group"] = np.where(
        frame["RID"].isin(healthy),
        "CDR0_stable",
        np.where(frame["RID"].isin(ad_subjects), "AD_dementia_any", "other"),
    )

    result, audit = _finalise(frame, scans, "oasis")
    audit["visit_index_without_t1w_mr_day"] = len(unmapped)
    audit["visit_index_without_t1w_mr_day_examples"] = unmapped[:10]
    audit["scans_without_cdr_within_one_year"] = int(result["CDRTOT"].isna().sum())
    audit["cohort_group_counts"] = {str(k): int(v) for k, v in result["cohort_group"].value_counts().items()}
    return result, audit


def build_calsnic(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    scans = discovered_scans(args.segmentation_root)
    sheet = pd.read_csv(args.clinical, low_memory=False)
    sheet["subject"] = sheet["Filename"].astype(str).str.strip()
    sheet["visit_number"] = sheet["Visit Label"].astype(str).str.extract(r"(\d+)")[0]
    sheet = sheet.dropna(subset=["visit_number"])

    frame = scans.copy()
    parsed = frame["scan_id"].str.extract(
        r"^(?P<subject>CALSNIC\d+_[A-Z]+_[CP]\d+)_(?P<protocol>[^_]+)_V(?P<visit_number>\d+)$"
    )
    frame = pd.concat([frame, parsed], axis=1)
    unparsed = frame.loc[frame["subject"].isna(), "scan_id"].tolist()
    frame = frame.loc[frame["subject"].notna()].copy()

    columns = [
        "subject", "visit_number", "Study", "Site", "Patient or Control", "Diagnosis",
        "Age", "Sex", "YearsEd", "MRI_Date", "Visit_Date", "Symptom_Duration", "ALSFRS_TotalScore",
    ]
    merged = frame.merge(sheet[columns], on=["subject", "visit_number"], how="left", indicator=True)
    unmatched = merged.loc[merged["_merge"].ne("both"), "scan_id"].tolist()
    merged = merged.drop(columns="_merge")

    merged["scan_date"] = pd.to_datetime(merged["MRI_Date"], errors="coerce").fillna(
        pd.to_datetime(merged["Visit_Date"], errors="coerce")
    )
    first_date = merged.groupby("subject")["scan_date"].transform("min")
    merged["years_from_first_scan"] = (merged["scan_date"] - first_date).dt.days / DAYS_PER_YEAR
    merged["month_from_viscode"] = merged["years_from_first_scan"] * 12.0
    # Age and sex are recorded on the first visit row only; carry them across visits.
    baseline_age = pd.to_numeric(merged["Age"], errors="coerce").groupby(merged["subject"]).transform("max")
    merged["AGE"] = baseline_age + merged["years_from_first_scan"]
    merged["PTGENDER"] = merged.groupby("subject")["Sex"].transform(lambda s: s.dropna().iloc[0] if s.notna().any() else None)

    group = merged["Patient or Control"].astype(str).str.strip()
    id_group = merged["subject"].str.extract(r"_([CP])\d+$")[0].map({"C": "Control", "P": "Patient"})
    merged["cohort_group"] = np.where(group.isin(["Control", "Patient"]), group, id_group)
    # The sheet carries stray trailing spaces ("ALS " vs "ALS"), which would otherwise
    # split one diagnosis into two cohort labels.
    merged["Diagnosis"] = merged["Diagnosis"].astype("string").str.strip().replace({"": None})
    diagnosis = merged.groupby("subject")["Diagnosis"].transform(
        lambda s: s.dropna().iloc[0] if s.notna().any() else None
    )
    # ALS, not Alzheimer's: the diagnosis columns stay in the study's own vocabulary.
    merged["visit_dx_3class"] = np.where(
        merged["cohort_group"].eq("Control"), "Control", diagnosis.fillna("Patient_unspecified")
    )
    merged["baseline_dx_3class"] = merged["visit_dx_3class"]
    merged["DX"] = merged["visit_dx_3class"]
    merged["DX.bl"] = merged["baseline_dx_3class"]

    result, audit = _finalise(merged, scans, "calsnic")
    audit["scan_ids_not_parseable"] = len(unparsed)
    audit["scan_ids_not_parseable_examples"] = unparsed[:10]
    audit["scans_without_sheet_row"] = len(unmatched)
    audit["scans_without_sheet_row_examples"] = unmatched[:10]
    audit["cohort_group_counts"] = {str(k): int(v) for k, v in result["cohort_group"].value_counts(dropna=False).items()}
    audit["diagnosis_counts"] = {str(k): int(v) for k, v in result["visit_dx_3class"].value_counts(dropna=False).items()}
    return result, audit


BUILDERS = {"aibl": build_aibl, "oasis": build_oasis, "calsnic": build_calsnic}


def main() -> int:
    args = parse_args()
    defaults = DEFAULTS[args.dataset]
    args.segmentation_root = (args.segmentation_root or defaults["segmentations"]).expanduser().resolve()
    args.output_dir = (args.output_dir or defaults["output_dir"]).expanduser().resolve()
    for key in ("clinical", "inventory", "clinical_zip", "cohort_zip"):
        if getattr(args, key, None) is None and key in defaults:
            setattr(args, key, defaults[key])

    print("=" * 88)
    print(f"Clinical manifest for {args.dataset.upper()}")
    print(f"Segmentations (read-only): {args.segmentation_root}")
    print(f"Output:                    {args.output_dir}")
    print("=" * 88, flush=True)

    frame, audit = BUILDERS[args.dataset](args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.dataset}_clinical_pipeline.csv"
    json_path = args.output_dir / f"{args.dataset}_clinical_audit.json"
    frame.to_csv(csv_path, index=False)
    audit["output_csv"] = str(csv_path)
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"\nwrote {csv_path} ({len(frame):,} rows)")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
