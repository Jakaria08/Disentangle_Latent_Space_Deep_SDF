# QC-Clean Direct Flow Retrain

This experiment is a clean retrain copy of
`siren_no_skip_flow_real_sdf_observed_virtual_cocycle`.

QC metadata source:

`/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle/analysis/mesh_longitudinal_qc/metadata_drop_bad_scans_min2.csv`

The SIREN decoder is still frozen. Only the longitudinal direct-flow model is
trained here. The latent NPZ files were filtered so each split contains exactly
the scan IDs present in the QC-clean metadata.

## Counts

```json
{
  "test": {
    "scans": 214,
    "subjects": 50,
    "pairs": 483,
    "pairs_with_observed_intermediate": 319,
    "diagnosis_scans": {
      "CN": 149,
      "AD": 65
    },
    "diagnosis_subjects": {
      "CN": 29,
      "AD": 21
    }
  },
  "train": {
    "scans": 1791,
    "subjects": 421,
    "pairs": 3680,
    "pairs_with_observed_intermediate": 2310,
    "diagnosis_scans": {
      "CN": 1295,
      "AD": 496
    },
    "diagnosis_subjects": {
      "CN": 261,
      "AD": 160
    }
  },
  "val": {
    "scans": 224,
    "subjects": 52,
    "pairs": 522,
    "pairs_with_observed_intermediate": 350,
    "diagnosis_scans": {
      "CN": 152,
      "AD": 72
    },
    "diagnosis_subjects": {
      "CN": 28,
      "AD": 24
    }
  }
}
```

## Commands

From repo root:

```bash
cd /home/jakaria/INR/Deep3DComp
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2 --validate-only
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2 --gpu 0 --smoke-batches 2
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2 --gpu 0
```

After training:

```bash
cd /home/jakaria/INR/Deep3DComp
/home/jakaria/anaconda3/envs/inr_sdf/bin/python evaluate_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2 --checkpoint best --split all --gpu 0
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/scripts/summarize_gap_bins.py --analysis /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/analysis/checkpoint_best
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/scripts/run_rich_visualization_analysis.py --checkpoint best --gpu 0
```

If `best.pth` is not created, evaluate `best_candidate` instead.
