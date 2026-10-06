# Phase Two: Mask2Former Boundary-Loss Fork

This folder contains the second phase of the thesis work: the modified Mask2Former codebase with boundary-loss integration and the thesis-specific training, evaluation, and analysis utilities.

## Upstream Sources

- Base architecture and most of the original training code: [`facebookresearch/mask2former`](https://github.com/facebookresearch/mask2former)
- Boundary-loss reference implementation: [`LIVIAETS/boundary-loss`](https://github.com/LIVIAETS/boundary-loss)

## What Lives Here

- The Mask2Former fork itself.
- Thesis-specific configs and scripts.
- The upstream documentation copied with the fork.
- Notes describing what changed relative to upstream Mask2Former.
- Any phase-two-specific experiment notes or release guidance.

## Key Paths

- `phase_two/mask2former/`, `phase_two/mask2former_video/`, `phase_two/train_net.py`, `phase_two/train_net_video.py`, `phase_two/configs/`, `phase_two/tools/`, and related files are the working phase-two code.

## Publication Guidance

- Keep the original license notices from upstream files that were copied or modified.
- Do not publish private checkpoints, raw datasets, tokens, or machine-specific caches.
- Document the exact upstream commit or release that your fork was based on.
