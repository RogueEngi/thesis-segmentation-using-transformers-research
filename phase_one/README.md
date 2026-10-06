# Phase One: Colab Baselines

This folder contains the material used for the first phase of the thesis work: the Colab-based training and evaluation workflow for SETR, Segmenter, SegFormer, and Mask2Former on the remote-sensing datasets.

## Layout

- `configs/loveda/` contains the cleaned config files that were used for the LoveDA experiments.
- `configs/vaihingen/` contains the cleaned config files that were used for the Vaihingen experiments.
- `notebooks/setup/` contains the Colab setup notebook for installing and caching dependencies.
- `notebooks/train/` contains the generic training notebooks for the phase-one models.
- `notebooks/eval/` contains the evaluation notebooks.
- `notebooks/archive/` contains scratch or redundant notebooks that are kept only for traceability.
- `dataset_converters/` contains dataset preparation scripts.
- `Modified_dataset_converters/` keeps the edited dataset conversion code that was used during experimentation.
- `docs/how_to_run_colab.txt` records the original Colab workflow notes.

## Notes

- The notebooks were written for Google Colab and still contain absolute Drive paths where needed.
- The original cache and wheel strategy was used to avoid reinstalling large packages on every session; those generated caches are intentionally not tracked in Git.
- The four training notebooks now use a dataset selector so the same file can be used for LoveDA or Vaihingen.
- If you want to rerun the experiments locally, start from the setup notebook and adjust the Drive paths to your own environment.
