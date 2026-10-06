# Copyright (c) Facebook, Inc. and its affiliates.
# Dataset registration for LoveDA semantic segmentation dataset
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# LoveDA dataset - Official label definition from https://github.com/Junjue-Wang/LoveDA:
# "Category labels: background – 1, building – 2, road – 3, water – 4, 
#  barren – 5, forest – 6, agriculture – 7. 
#  And the no-data regions were assigned 0 which should be ignored."
#
# Original annotation values:
#   0: no-data (IGNORE)
#   1: background
#   2: building
#   3: road
#   4: water
#   5: barren
#   6: forest
#   7: agriculture
#
# With reduce_zero_label=True:
#   Label 0 (no-data) -> 255 (ignore)
#   Labels 1-7 -> 0-6
#
# Final NUM_CLASSES = 7 (matching MMCV config)

# Categories after reduce_zero_label (7 classes, indices 0-6)
LOVEDA_CATEGORIES_NO_BG = [
    {"name": "background", "id": 0, "color": [255, 255, 255]},    # White (was label 1)
    {"name": "building", "id": 1, "color": [255, 0, 0]},          # Red (was label 2)
    {"name": "road", "id": 2, "color": [255, 255, 0]},            # Yellow (was label 3)
    {"name": "water", "id": 3, "color": [0, 0, 255]},             # Blue (was label 4)
    {"name": "barren", "id": 4, "color": [159, 129, 183]},        # Purple (was label 5)
    {"name": "forest", "id": 5, "color": [0, 255, 0]},            # Green (was label 6)
    {"name": "agricultural", "id": 6, "color": [255, 195, 128]},  # Orange (was label 7)
]


def _get_loveda_meta_no_bg():
    """Get metadata for LoveDA without background (7 classes after reduce_zero_label)."""
    stuff_classes = [k["name"] for k in LOVEDA_CATEGORIES_NO_BG]
    stuff_colors = [k["color"] for k in LOVEDA_CATEGORIES_NO_BG]
    
    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_loveda(root):
    """
    Register LoveDA dataset for semantic segmentation.
    
    Expected directory structure (from your converter):
    root/
        loveDA/
            img_dir/
                train/
                val/
                test/
            ann_dir/
                train/
                val/
    
    The annotation images should be single-channel PNGs with pixel values 0-7
    where 0 is background. The reduce_zero_label mapper will:
    - Convert class 0 (background) -> 255 (ignore)
    - Shift classes 1-7 -> 0-6
    
    Final NUM_CLASSES = 7 (building, road, water, barren, forest, agriculture, nodata)
    This matches MMCV config: num_classes=7 with reduce_zero_label=True
    """
    root = os.path.join(root, "loveDA")
    # Use metadata without background (7 classes) since we use reduce_zero_label
    meta = _get_loveda_meta_no_bg()
    
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, "img_dir", dirname)
        gt_dir = os.path.join(root, "ann_dir", dirname)
        dataset_name = f"loveda_sem_seg_{name}"
        
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
register_loveda(_root)
