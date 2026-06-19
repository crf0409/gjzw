# Three-Corpus Experiment Release

This directory contains lightweight, reviewable metadata for the completed AAFNet three-corpus experiment release.
The full ZIP package is intentionally not committed to git because it exceeds GitHub's normal file-size limit.

## Completion

- Completion audit: `True`
- Audit summary: pass=110, warn=3, fail=0
- Public CV cells: 60/60
- Public robustness probes: 60/60
- Public calibration probes: 60/60
- Public rotation probes: 60/60

## Full Package

- Local package: `release_packages/three_corpus_release_final_manual.zip`
- Release asset: `three_corpus_release_final_manual.zip`
- SHA256: `d39a9f5683dc7348599104d3640533c73bfa59b6994eedccdae3889bd5d19748`
- Size bytes: `368815033`

## Contents

- `release_packages/`: package manifest and checksum.
- `outputs/three_corpus_completion/`: final completion audit and refresh reports.
- `outputs/public_cv_parity/`: public-corpus cross-validation status.
- `outputs/public_followup_probes/`: follow-up probe summaries and split-run reports.
- `paper/figures/nature_source_data/`: CSV source data for manuscript figures.
- `manifest.json`: machine-readable index for this metadata bundle.

The final follow-up probe status source is copied from:
`paper/figures/nature_source_data/public_followup_probe_status_source.csv`
