# Thesis Repository Docs

This folder contains the release-facing documentation for the thesis repository.

## Recommended Reading Order

1. `README.md` at the repository root for the project overview.
2. `phase_one/README.md` for the Colab baseline experiments.
3. `phase_two/README.md` for the modified Mask2Former fork.
4. `UPSTREAMS.md` for provenance and attribution.

## Purpose

- Keep the repository structure understandable for reviewers.
- Record what was copied or adapted from upstream projects.
- Provide a clear starting point for reproduction.
# Repository Guide

This directory holds the documentation that makes the thesis repository understandable for reviewers and other researchers.

## Suggested Documents

- `reproducibility.md`: seed handling, hardware, and environment notes.
- `dataset_notes.md`: dataset sources, splits, and preprocessing.
- `results.md`: tables, figures, and experiment summaries.
- `upstream_provenance.md`: what was copied from Mask2Former and boundary-loss.

## Release Checklist

- `phase_one/` contains the Colab-era baseline experiments.
- `phase_two/` contains the modified Mask2Former fork.
- Notebook files are included only where they help reproduce the work.
- Generated outputs, caches, and private data are excluded from Git.
