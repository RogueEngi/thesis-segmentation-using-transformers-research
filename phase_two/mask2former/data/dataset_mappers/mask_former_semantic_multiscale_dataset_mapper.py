# Copyright (c) Facebook, Inc. and its affiliates.
# Modified to support multi-scale training and reduce_zero_label for remote sensing datasets
"""
Enhanced Dataset Mapper with Multi-Scale Training

This mapper implements the same augmentation strategy as MMCV/MMSegmentation:
1. Multi-scale resize (ratio_range = 0.5 to 2.0)
2. Random crop with category area constraint
3. Random flip
4. PhotoMetric-like color augmentation

This should achieve comparable results to MMCV's training pipeline.
"""

import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["MaskFormerSemanticMultiScaleDatasetMapper"]


def compute_signed_distance_map(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute signed distance transform for a binary mask.
    Based on Kervadec et al. "Boundary loss for highly unbalanced segmentation" (MIDL 2019)
    
    Args:
        binary_mask: Binary mask of shape (H, W), where 1 indicates foreground.
        
    Returns:
        Signed distance map of shape (H, W):
        - Negative values inside the foreground region (closer to center = more negative)
        - Positive values outside the foreground region
        - Zero at the boundary
    """
    # Handle empty masks (all zeros or all ones)
    if not binary_mask.any():
        # All background - return all positive (large) distances
        return np.ones_like(binary_mask, dtype=np.float32) * np.sqrt(binary_mask.shape[0]**2 + binary_mask.shape[1]**2)
    
    if binary_mask.all():
        # All foreground - return all negative distances
        return -np.ones_like(binary_mask, dtype=np.float32) * np.sqrt(binary_mask.shape[0]**2 + binary_mask.shape[1]**2)
    
    # Compute distance transform for foreground and background
    pos_mask = binary_mask.astype(bool)
    neg_mask = ~pos_mask
    
    # Distance from each background pixel to nearest foreground pixel
    dist_outside = distance_transform_edt(neg_mask)
    # Distance from each foreground pixel to nearest background pixel  
    dist_inside = distance_transform_edt(pos_mask)
    
    # Signed distance: positive outside, negative inside
    # Subtract 1 from inside distance so boundary pixels have distance 0
    signed_dist = dist_outside * neg_mask - (dist_inside - 1) * pos_mask
    
    return signed_dist.astype(np.float32)


class MaskFormerSemanticMultiScaleDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.
    
    This mapper implements:
    1. Multi-scale training (like MMCV's RandomResize with ratio_range)
    2. reduce_zero_label functionality (class 0 -> ignore, others shift down)
    3. Random crop with category area constraint
    4. Color augmentation
    
    The multi-scale training is CRITICAL for achieving MMCV-level performance!
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        image_format,
        ignore_label,
        size_divisibility,
        reduce_zero_label=True,
        # Multi-scale parameters (MMCV-style)
        min_scale=0.5,
        max_scale=2.0,
        crop_size=(512, 512),
        single_category_max_area=0.75,
        color_aug_ssd=True,
        boundary_loss_enabled=False,
    ):
        """
        Args:
            is_train: for training or inference
            image_format: an image format supported by detection_utils.read_image
            ignore_label: the label that is ignored during evaluation
            size_divisibility: pad image size to be divisible by this value
            reduce_zero_label: if True, class 0 becomes ignore and other classes shift down
            min_scale: minimum scale ratio for multi-scale training (default: 0.5)
            max_scale: maximum scale ratio for multi-scale training (default: 2.0)
            crop_size: crop size after scaling (default: (512, 512))
            single_category_max_area: max ratio of area a single category can occupy (default: 0.75)
            color_aug_ssd: whether to apply ColorAugSSD augmentation
            boundary_loss_enabled: whether to compute distance maps for boundary loss
        """
        self.is_train = is_train
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.reduce_zero_label = reduce_zero_label
        
        # Multi-scale parameters
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.crop_size = crop_size
        self.single_category_max_area = single_category_max_area
        self.color_aug_ssd = color_aug_ssd
        self.boundary_loss_enabled = boundary_loss_enabled

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Mode: {mode}")
        logger.info(f"[{self.__class__.__name__}] Multi-scale: {min_scale}x - {max_scale}x")
        logger.info(f"[{self.__class__.__name__}] Crop size: {crop_size}")
        logger.info(f"[{self.__class__.__name__}] reduce_zero_label: {reduce_zero_label}")
        logger.info(f"[{self.__class__.__name__}] color_aug_ssd: {color_aug_ssd}")
        if boundary_loss_enabled:
            logger.info(f"[{self.__class__.__name__}] Boundary loss enabled - computing distance maps")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Get parameters from config
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        # Multi-scale parameters
        min_scale = getattr(cfg.INPUT, 'MIN_SCALE', 0.5)
        max_scale = getattr(cfg.INPUT, 'MAX_SCALE', 2.0)
        crop_size = cfg.INPUT.CROP.SIZE
        single_category_max_area = cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA
        color_aug_ssd = cfg.INPUT.COLOR_AUG_SSD
        
        # Check if boundary loss is enabled
        boundary_loss_enabled = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT > 0

        ret = {
            "is_train": is_train,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "reduce_zero_label": True,  # Always True for this mapper
            "min_scale": min_scale,
            "max_scale": max_scale,
            "crop_size": tuple(crop_size),
            "single_category_max_area": single_category_max_area,
            "color_aug_ssd": color_aug_ssd,
            "boundary_loss_enabled": boundary_loss_enabled,
        }
        return ret

    def _reduce_zero_label(self, sem_seg_gt):
        """
        Apply reduce_zero_label transformation:
        - Class 0 -> ignore_label (255)
        - Class N (N > 0) -> Class N-1
        
        This matches MMCV/MMSegmentation's reduce_zero_label=True behavior.
        """
        sem_seg_gt = sem_seg_gt.copy()
        
        # Mark class 0 pixels
        zero_mask = sem_seg_gt == 0
        
        # Protect pixels that are already ignore_label
        ignore_mask = sem_seg_gt == self.ignore_label
        
        # Subtract 1 from all non-zero, non-ignore pixels
        valid_mask = ~zero_mask & ~ignore_mask
        sem_seg_gt[valid_mask] = sem_seg_gt[valid_mask] - 1
        
        # Set original class 0 pixels to ignore_label
        sem_seg_gt[zero_mask] = self.ignore_label
        
        return sem_seg_gt

    def _random_scale(self, image, sem_seg_gt):
        """
        Apply random scale augmentation (like MMCV's RandomResize with ratio_range).
        
        This scales the image by a random factor between min_scale and max_scale,
        while maintaining aspect ratio.
        """
        h, w = image.shape[:2]
        
        # Sample random scale factor
        scale = np.random.uniform(self.min_scale, self.max_scale)
        
        # Compute new size (maintain aspect ratio)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Use Detectron2's resize transform
        resize_tfm = T.ResizeTransform(h, w, new_h, new_w)
        
        image = resize_tfm.apply_image(image)
        sem_seg_gt = resize_tfm.apply_segmentation(sem_seg_gt)
        
        return image, sem_seg_gt

    def _random_crop(self, image, sem_seg_gt):
        """
        Apply random crop with category area constraint.
        
        Like MMCV's RandomCrop with cat_max_ratio.
        """
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size
        
        # If image is smaller than crop size, pad first
        if h < crop_h or w < crop_w:
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=128
            )
            sem_seg_gt = np.pad(
                sem_seg_gt,
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=self.ignore_label
            )
            h, w = image.shape[:2]
        
        # Try to find a valid crop
        max_tries = 10
        for _ in range(max_tries):
            # Random crop position
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
            
            crop_seg = sem_seg_gt[top:top+crop_h, left:left+crop_w]
            
            # Check category area constraint
            unique_classes, counts = np.unique(crop_seg, return_counts=True)
            total_pixels = crop_h * crop_w
            
            # Filter out ignore label
            valid_idx = unique_classes != self.ignore_label
            
            if valid_idx.any():
                valid_counts = counts[valid_idx]
                max_ratio = valid_counts.max() / total_pixels
                
                if max_ratio <= self.single_category_max_area:
                    break
        
        # Apply the crop
        image = image[top:top+crop_h, left:left+crop_w].copy()
        sem_seg_gt = sem_seg_gt[top:top+crop_h, left:left+crop_w].copy()
        
        return image, sem_seg_gt

    def _random_flip(self, image, sem_seg_gt, prob=0.5):
        """Apply random horizontal flip."""
        if np.random.random() < prob:
            image = np.fliplr(image).copy()
            sem_seg_gt = np.fliplr(sem_seg_gt).copy()
        return image, sem_seg_gt

    def _color_augmentation(self, image):
        """
        Apply color augmentation similar to MMCV's PhotoMetricDistortion.
        Uses Detectron2's ColorAugSSDTransform.
        """
        if self.color_aug_ssd:
            # Create augmentation
            aug = ColorAugSSDTransform(img_format=self.img_format)
            aug_input = T.AugInput(image)
            aug_input, _ = T.apply_transform_gens([aug], aug_input)
            image = aug_input.image
        return image

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticMultiScaleDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # ====================================================================
        # Apply augmentations in MMCV order:
        # 1. Reduce zero label (before any spatial transforms)
        # 2. Random scale (multi-scale training)
        # 3. Random crop
        # 4. Random flip
        # 5. Color augmentation
        # ====================================================================
        
        # 1. Reduce zero label
        if self.reduce_zero_label:
            sem_seg_gt = self._reduce_zero_label(sem_seg_gt)
        
        # 2. Multi-scale resize
        image, sem_seg_gt = self._random_scale(image, sem_seg_gt)
        
        # 3. Random crop with category area constraint
        image, sem_seg_gt = self._random_crop(image, sem_seg_gt)
        
        # 4. Random horizontal flip
        image, sem_seg_gt = self._random_flip(image, sem_seg_gt)
        
        # 5. Color augmentation
        image = self._color_augmentation(image)

        # Convert to tensors
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        # Padding if needed
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                (self.size_divisibility - image_size[1] % self.size_divisibility) % self.size_divisibility,
                0,
                (self.size_divisibility - image_size[0] % self.size_divisibility) % self.size_divisibility,
            ]
            if padding_size != [0, 0, 0, 0]:
                image = F.pad(image, padding_size, value=128).contiguous()
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])

        # Store in dataset dict
        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks (required by Mask2Former)
        sem_seg_np = sem_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = np.unique(sem_seg_np)
        classes = classes[classes != self.ignore_label]
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        masks = []
        dist_maps = []
        for class_id in classes:
            binary_mask = (sem_seg_np == class_id)
            masks.append(binary_mask)
            
            # Compute distance map if boundary loss is enabled
            if self.boundary_loss_enabled:
                dist_map = compute_signed_distance_map(binary_mask.astype(np.float32))
                dist_maps.append(dist_map)

        if len(masks) == 0:
            instances.gt_masks = torch.zeros((0, sem_seg_np.shape[-2], sem_seg_np.shape[-1]))
            if self.boundary_loss_enabled:
                instances.gt_dist_maps = torch.zeros((0, sem_seg_np.shape[-2], sem_seg_np.shape[-1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            
            # Add distance maps if boundary loss is enabled
            if self.boundary_loss_enabled:
                instances.gt_dist_maps = torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in dist_maps]
                )

        dataset_dict["instances"] = instances

        return dataset_dict
