# Copyright (c) Facebook, Inc. and its affiliates.
# Modified to support reduce_zero_label for remote sensing datasets
"""
Semantic Segmentation Evaluator with reduce_zero_label support.

This evaluator applies the same reduce_zero_label transformation to the ground truth
that is applied during training, ensuring predictions and labels are aligned.
"""
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import SemSegEvaluator


class ReduceZeroLabelSemSegEvaluator(SemSegEvaluator):
    """
    Semantic Segmentation Evaluator that applies reduce_zero_label to ground truth.
    
    This is necessary when training with reduce_zero_label=True (like for LoveDA/Vaihingen)
    because the model predicts classes 0-(N-1), but the original ground truth has classes 1-N.
    
    The transformation applied to ground truth:
    - Class 0 -> 255 (ignore)
    - Class N (N > 0) -> Class N-1
    """
    
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=None,
        num_classes=None,
        ignore_label=None,
        reduce_zero_label=True,
    ):
        """
        Args:
            dataset_name: name of the dataset to be evaluated.
            distributed: if True, will collect results from all ranks and run evaluation
                in the main process.
            output_dir: an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and return as numpy array.
            num_classes: number of classes (should be the reduced number, e.g., 7 for LoveDA).
            ignore_label: label to ignore in evaluation (default: 255).
            reduce_zero_label: if True, apply reduce_zero_label to ground truth.
        """
        # Pass sem_seg_loading_fn only if provided
        if sem_seg_loading_fn is not None:
            super().__init__(
                dataset_name,
                distributed=distributed,
                output_dir=output_dir,
                sem_seg_loading_fn=sem_seg_loading_fn,
                num_classes=num_classes,
                ignore_label=ignore_label,
            )
        else:
            super().__init__(
                dataset_name,
                distributed=distributed,
                output_dir=output_dir,
                num_classes=num_classes,
                ignore_label=ignore_label,
            )
        self.reduce_zero_label = reduce_zero_label
        self._logger = logging.getLogger(__name__)
        
        if self.reduce_zero_label:
            self._logger.info(f"[{self.__class__.__name__}] reduce_zero_label enabled: "
                            f"GT class 0 -> 255 (ignore), GT class N -> class N-1")
    
    def _apply_reduce_zero_label(self, gt_sem_seg):
        """
        Apply reduce_zero_label transformation to ground truth.
        
        Args:
            gt_sem_seg: numpy array of shape (H, W) with original class labels (0-N)
            
        Returns:
            numpy array with transformed labels (class 0->255, class N->N-1)
        """
        # Create a copy to avoid modifying the original
        gt_sem_seg = gt_sem_seg.copy()
        
        # Mark class 0 pixels (will become ignore)
        zero_mask = gt_sem_seg == 0
        
        # Protect pixels that are already ignore_label
        ignore_mask = gt_sem_seg == self._ignore_label
        
        # Shift all non-zero, non-ignore classes down by 1
        valid_mask = ~zero_mask & ~ignore_mask
        gt_sem_seg[valid_mask] = gt_sem_seg[valid_mask] - 1
        
        # Set original class 0 pixels to ignore_label
        gt_sem_seg[zero_mask] = self._ignore_label
        
        return gt_sem_seg
    
    def process(self, inputs, outputs):
        """
        Process predictions and ground truth, applying reduce_zero_label to GT.
        
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            
            # Get ground truth filename and load it
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)
            
            # Apply reduce_zero_label to ground truth
            if self.reduce_zero_label:
                gt = self._apply_reduce_zero_label(gt)
            
            # Mark ignore_label as num_classes (for confusion matrix)
            gt[gt == self._ignore_label] = self._num_classes
            
            # Update confusion matrix
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            
            # Boundary IoU (if enabled)
            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)
            
            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))


def build_reduce_zero_label_sem_seg_evaluator(dataset_name, output_dir=None, distributed=True):
    """
    Helper function to build ReduceZeroLabelSemSegEvaluator with correct settings.
    
    Args:
        dataset_name: name of the dataset
        output_dir: output directory for results
        distributed: whether to use distributed evaluation
        
    Returns:
        ReduceZeroLabelSemSegEvaluator instance
    """
    return ReduceZeroLabelSemSegEvaluator(
        dataset_name,
        distributed=distributed,
        output_dir=output_dir,
        reduce_zero_label=True,
    )

