# Exact-triangle SDF relabelling pilot

This isolated pipeline tests one causal question: does replacing the existing approximate
`PreprocessMesh` labels with exact point-to-triangle signed distances improve hippocampus
reconstruction when the query coordinates, cohort, architecture, optimizer, sampling, and seed
are held fixed?

The code lives in the repository, but every generated manifest, SDF archive, audit, runtime
config, checkpoint, latent, log, evaluation mesh, and comparison report is forced below:

`/mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/exact_sdf_relabel_v1`

The hidden test split is selected and relabelled once, but periodic training evaluation is limited
to `train` and `val`. The launcher refuses test evaluation unless `--confirm-test` is supplied.

## Fixed experiment contract

- Pilot targets: approximately 400 train, 100 validation, and 100 test scans.
- Selection unit: subject; all visits of every selected subject are retained.
- Stratification: diagnosis and within-diagnosis volume tertile.
- Query points: exactly the original xyz values, with no resampling.
- New labels: exact closest-triangle magnitude, negative inside and positive outside.
- Mesh requirement: finite, watertight, non-degenerate; orientation is corrected only in memory.
- Latent size: 256.
- Compact network: unchanged transferred 256-D global skip-SIREN, local skip-SIREN, and
  shared `32^3 x 16` grid.
- Compact schedule: 100 global adaptation + 100 local warm-up + 800 joint epochs = 1,000.
- Multires network: unchanged 8/16/24/32/48/64 shared-grid single-field model and coarse-to-fine
  schedule; total 1,000 epochs.
- Sampling and loss weights: unchanged from the current primary implementations.
- Eikonal: disabled in all four runs. This isolates the label change first.
- Checkpoints and geometry evaluation: epochs 500 and 1,000; 100 scans per available train/val
  split, resolution 256, 500 latent-fit steps, and 30,000 surface samples.
- PCA: the current 150-component PCA is retained as a descriptive reference. It was trained on
  the full training cohort, not the 400-scan pilot, so it is not a data-matched winner/loser test.

The Compact run also retains the current global-SIREN transfer policy. On the deterministic pilot,
the old global checkpoint overlaps many subjects; 19 validation scans from four subjects remain
source-unseen and are used by Compact's internal held-out-SDF check. Periodic mesh selection still
reports all 100 pilot validation scans and records source-pretraining overlap. This is acceptable for
the paired exact-versus-approximate label question because both arms have identical prior exposure,
but it is not a claim of a completely de-novo external Compact test. Multires is trained from scratch.

The sampling *policy* and thresholds are held fixed. Realized near/ultra-near membership can change
when an approximate magnitude is replaced by its exact value, and corrected signs can move a query
between `pos` and `neg`; those changes are intentional consequences of correcting the SDF labels.
The relabelling audit reports sign-partition changes while proving that the coordinate set itself did
not change.

The four runs are `compact_approx`, `compact_exact`, `multires_approx`, and `multires_exact`.
Within each architecture, the approximate and exact configurations differ only in label manifest,
label provenance, experiment name, and output directory.

## Commands

Run from the repository root. The bulk wrapper keeps temporary files and caches on the 10-TB disk,
uses a short `TMPDIR`, and prevents Python bytecode output on the SSD.

```bash
BASE=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_exact_sdf_relabel_v1
RUN=$BASE/scripts/run_on_bulk.sh
```

### 1. Validate the code/config contract without requiring generated data

```bash
bash $RUN $BASE/scripts/launch_pilot.py check
```

### 2. Select the subject-complete pilot manifest

```bash
bash $RUN $BASE/scripts/build_pilot_manifest.py
```

The actual split counts can differ slightly from 400/100/100 because subjects are never split and
all longitudinal visits are retained.

### 3. Optional one-scan backend check

Choose a scan ID from the approximate pilot manifest. This creates only that exact archive and a
partial audit; it intentionally does not create the training manifest.

```bash
bash $RUN $BASE/scripts/relabel_exact_triangle_sdf.py --scan-id 1035_bl
```

### 4. Relabel the full pilot

If the optional check was run, use `--resume`. Four workers are a reasonable starting point;
reduce this to one or two if RAM pressure is high.

```bash
bash $RUN $BASE/scripts/relabel_exact_triangle_sdf.py --workers 4 --resume
```

For a completely fresh run with no prior archive, omit `--resume`. The script writes each archive
atomically and writes the exact training manifest only when every selected scan succeeds. It never
repairs or overwrites a source mesh/SDF. A failed interrupted run is continued with `--resume`.

### 5. Independently audit the completed labels

```bash
bash $RUN $BASE/scripts/audit_exact_triangle_sdf.py --workers 4 --points-per-scan 2048
bash $RUN $BASE/scripts/launch_pilot.py check --require-data
```

The audit proves that approximate and exact manifests contain the same scans, subjects and splits;
reconstructs source query order using saved index arrays; demands bitwise-identical xyz values;
recomputes exact labels on a deterministic subset; checks sign partitions; and rechecks mesh
watertightness/orientation. Training cannot be launched through the wrapper until this audit passes.

### 6. Train

Each command is a long run. Use the actual visible CUDA ordinal; with one visible GPU this is
normally `cuda:0` even when the physical GPU was selected through `CUDA_VISIBLE_DEVICES`.

```bash
bash $RUN $BASE/scripts/launch_pilot.py train --experiment compact_approx --device cuda:0 --execute
bash $RUN $BASE/scripts/launch_pilot.py train --experiment compact_exact  --device cuda:0 --execute
bash $RUN $BASE/scripts/launch_pilot.py train --experiment multires_approx --device cuda:0 --execute
bash $RUN $BASE/scripts/launch_pilot.py train --experiment multires_exact  --device cuda:0 --execute
```

Without `--execute`, the launcher performs a dry run and prints the exact command. Resume example:

```bash
bash $RUN $BASE/scripts/launch_pilot.py train --experiment compact_exact --device cuda:0 --resume latest --execute
```

Do not resume an exact run from an approximate-run checkpoint. The global Compact SIREN warm start
is still the previously trained 256-D global checkpoint, as requested; new pilot latent codes are
initialized and optimized normally.

### 7. Validation-only evaluation and paired comparison

Periodic epoch-500/1000 evaluation already runs during training. These commands rerun a chosen
checkpoint on validation if needed:

```bash
bash $RUN $BASE/scripts/launch_pilot.py evaluate --experiment compact_exact --checkpoint best_mesh --splits val --device cuda:0 --execute
bash $RUN $BASE/scripts/launch_pilot.py evaluate --experiment multires_exact --checkpoint best_mesh --splits val --device cuda:0 --execute
```

Compare exact versus approximate on identical validation scans at epoch 1,000:

```bash
bash $RUN $BASE/scripts/compare_paired_evaluations.py --family compact --epoch 1000 --split val
bash $RUN $BASE/scripts/compare_paired_evaluations.py --family multires --epoch 1000 --split val
```

The paired report gives exact-minus-approximate changes, subject-cluster bootstrap 95% intervals,
and the fraction of scans improved. Improvement is negative for ASSD/HD95/Chamfer/volume error and
positive for F-score. Select a checkpoint/architecture from validation ASSD first, then inspect
HD95, failure rate, watertightness, components, volume error, and field-gradient diagnostics.

### 8. One final locked test evaluation

Run this only after model and checkpoint selection is frozen. Repeat for the selected experiment,
not for every candidate during development.

```bash
bash $RUN $BASE/scripts/launch_pilot.py evaluate --experiment compact_exact --checkpoint best_mesh --splits test --device cuda:0 --confirm-test --execute
```

## Expected time and storage

The pilot contains about 300 million query points if the selected scans retain the current roughly
500,000 points per archive. A measured 10,000-point exact query on this machine took about three
seconds including Python/mesh/index startup. Plan approximately 25--40 hours with one relabelling
worker or roughly 8--16 hours with four workers; filesystem speed, mesh index construction, and RAM
contention dominate. The independent 2,048-point-per-scan audit should be much shorter.

From the completed full-cohort runs, median training-only epoch times were about 45 seconds for
Compact and 8.5 seconds for multires. Scaling to roughly 400 training scans suggests about 2.5
training-only hours for Compact and under one hour for multires per 1,000-epoch run. Latent fitting,
validation every 25 epochs, and each 200-shape periodic geometry evaluation add substantial time;
budget roughly 5--9 hours per Compact pilot and 3--6 hours per multires pilot on the same GPU.
Four runs therefore represent roughly 16--30 GPU-hours if run sequentially. These are planning
ranges, not guarantees.

Expect several gigabytes for exact compressed archives plus substantially more for four sets of
checkpoints, epoch-500/1000 meshes, PCA reference meshes, and evaluation reports. Check free space
on `/mnt/bulk10tb` before starting.

## Reuse for the lateral ventricle

The selection and relabelling code is structure-agnostic. Later, give
`build_pilot_manifest.py` a lateral-ventricle source manifest, a new 10T `--output-root`, and
`--cohort-name lateral_ventricle_pilot`; the relabeller derives the exact-manifest name from the
approximate manifest and places archives under the row's `structure` value. Keep that data root and
its audit separate from hippocampus. A separate pinned experiment matrix should then be made from
the lateral-ventricle network config/AABB; do not reuse the hippocampus network matrix blindly.
