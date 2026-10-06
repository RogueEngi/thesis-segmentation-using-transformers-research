# Copyright (c) Facebook, Inc. and its affiliates.
# Dataset registration for ISPRS Vaihingen semantic segmentation dataset
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# Vaihingen/ISPRS dataset with reduce_zero_label=True
# Based on MMSeg ISPRSDataset: https://github.com/open-mmlab/mmsegmentation
#
# Original annotation values: 0-6 (7 labels)
# With reduce_zero_label=True:
#   Label 0 -> 255 (ignore)
#   Labels 1-6 -> 0-5
#
# Classes (from MMSeg ISPRSDataset):
#   0: impervious_surface (was label 1)
#   1: building (was label 2)
#   2: low_vegetation (was label 3)
#   3: tree (was label 4)
#   4: car (was label 5)
#   5: clutter (was label 6)
#
# Final NUM_CLASSES = 6 (matching MMCV config)

VAIHINGEN_CATEGORIES = [
    {"name": "impervious_surface", "id": 0, "color": [255, 255, 255]},  # White
    {"name": "building", "id": 1, "color": [0, 0, 255]},                # Blue
    {"name": "low_vegetation", "id": 2, "color": [0, 255, 255]},        # Cyan
    {"name": "tree", "id": 3, "color": [0, 255, 0]},                    # Green
    {"name": "car", "id": 4, "color": [255, 255, 0]},                   # Yellow
    {"name": "clutter", "id": 5, "color": [255, 0, 0]},                 # Red
]


def _get_vaihingen_meta():
    stuff_classes = [k["name"] for k in VAIHINGEN_CATEGORIES]
    stuff_colors = [k["color"] for k in VAIHINGEN_CATEGORIES]
    
    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_vaihingen(root):
    """
    Register Vaihingen dataset for semantic segmentation.
    
    Expected directory structure:
    root/
        vaihingen/
            img_dir/
                train/
                val/
            ann_dir/
                train/
                val/
    
    The annotation images should be single-channel PNGs with pixel values 0-5.
    Class 0 (clutter) will be ignored via reduce_zero_label in the dataset mapper.
    """
    root = os.path.join(root, "vaihingen")
    meta = _get_vaihingen_meta()
    
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, "img_dir", dirname)
        gt_dir = os.path.join(root, "ann_dir", dirname)
        dataset_name = f"vaihingen_sem_seg_{name}"
        
        DatasetCatalog.register(
            dataset_name, 
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="png"
            )
        )
        MetadataCatalog.get(dataset_name).set(
            stuff_classes=meta["stuff_classes"],
            stuff_colors=meta["stuff_colors"],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # Pixels with value 255 will be ignored
        )


# Auto-register when imported
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_vaihingen(_root)
