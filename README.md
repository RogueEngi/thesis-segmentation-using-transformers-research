# Thesis Repository: Semantic Segmentation of Remote Sensing Images using Vision Transformers

This repository accompanies the thesis project on semantic segmentation of remote sensing imagery.
It is organized into two phases:

- `phase_one/`: Colab-based baseline experiments for SETR, Segmenter, SegFormer, and Mask2Former.
- `phase_two/`: the modified Mask2Former fork with boundary-loss integration and thesis-specific utilities.

For implementation details, start with `phase_one/README.md`, `phase_two/README.md`, and `docs/README.md`.

## Phase One
The first phase of this repository is done using [MMSegmentation toolbox](https://github.com/open-mmlab/mmsegmentation) and Google's [Colab](https://colab.research.google.com/).

## Phase Two

The second phase of this repository is based on Mask2Former, the masked-attention mask transformer for universal image segmentation.
The upstream project is preserved under `phase_two/` as the code foundation and attribution source.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for Mask2Former](datasets/README.md).

See [Getting Started with Mask2Former](GETTING_STARTED.md).

## Advanced usage

See [Advanced Usage of Mask2Former](ADVANCED_USAGE.md).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Mask2Former Model Zoo](MODEL_ZOO.md).

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Former is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## Acknowledgement

Code is largely based on Mask2Former (https://github.com/facebookresearch/mask2former).

As im new to Git and Github, I took suggestions from Githubs Copilot to re-structure and write markdown files. Please inform me as if you find any related issues in this repo :grimacing:.