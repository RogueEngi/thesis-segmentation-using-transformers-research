# Copyright (c) Facebook, Inc. and its affiliates.
"""
Sliding window inference for semantic segmentation.
Matches MMCV's slide mode for fair comparison.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel


__all__ = ["SemanticSegmentorWithSlide"]


class SemanticSegmentorWithSlide(nn.Module):
    """
    A SemanticSegmentor with sliding window inference enabled.
    Matches MMSegmentation's 'slide' test mode for fair comparison.
    
    During inference:
    1. Resize image (handled by data loader)
    2. Extract overlapping crops of size (crop_h, crop_w) with given stride
    3. Run model on each crop
    4. Average predictions in overlapping regions
    """

    def __init__(
        self, 
        cfg, 
        model, 
        crop_size=(512, 512), 
        stride=(341, 341),
        num_classes=None,
    ):
        """
        Args:
            cfg (CfgNode): Config
            model: A semantic segmentation model
            crop_size (tuple): Size of crops (H, W)
            stride (tuple): Stride between crops (H, W)
            num_classes (int): Number of classes. If None, inferred from cfg.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        
        self.cfg = cfg.clone()
        self.model = model
        self.crop_size = crop_size
        self.stride = stride
        
        if num_classes is None:
            num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.num_classes = num_classes
        
        # Get model properties
        self.device = next(model.parameters()).device
        self.pixel_mean = model.pixel_mean
        self.pixel_std = model.pixel_std
        self.size_divisibility = model.size_divisibility

    def forward(self, batched_inputs):
        """
        Same input/output format as the wrapped model's forward method.
        """
        return [self._inference_one_image(x) for x in batched_inputs]

    def _inference_one_image(self, input_dict):
        """
        Run sliding window inference on a single image.
        
        Args:
            input_dict: Dict with "image" (CHW tensor), "height", "width"
            
        Returns:
            Dict with "sem_seg" prediction
        """
        image = input_dict["image"].to(self.device)
        orig_height = input_dict.get("height", image.shape[1])
        orig_width = input_dict.get("width", image.shape[2])
        
        # Normalize image
        image = (image - self.pixel_mean) / self.pixel_std
        
        # Get padded image dimensions
        _, h, w = image.shape
        crop_h, crop_w = self.crop_size
        stride_h, stride_w = self.stride
        
        # Pad image if smaller than crop size
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
            _, h, w = image.shape
        
        # Calculate number of crops
        h_grids = max(h - crop_h + stride_h - 1, 0) // stride_h + 1
        w_grids = max(w - crop_w + stride_w - 1, 0) // stride_w + 1
        
        # Initialize prediction accumulator and count
        pred_accum = image.new_zeros((self.num_classes, h, w))
        count_mat = image.new_zeros((1, h, w))
        
        # Process each crop
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * stride_h
                x1 = w_idx * stride_w
                y2 = min(y1 + crop_h, h)
                x2 = min(x1 + crop_w, w)
                y1 = max(y2 - crop_h, 0)
                x1 = max(x2 - crop_w, 0)
                
                # Extract crop
                crop_img = image[:, y1:y2, x1:x2].unsqueeze(0)
                
                # Ensure size divisibility
                if self.size_divisibility > 0:
                    crop_h_pad = (self.size_divisibility - crop_img.shape[2] % self.size_divisibility) % self.size_divisibility
                    crop_w_pad = (self.size_divisibility - crop_img.shape[3] % self.size_divisibility) % self.size_divisibility
                    if crop_h_pad > 0 or crop_w_pad > 0:
                        crop_img = F.pad(crop_img, (0, crop_w_pad, 0, crop_h_pad))
                
                # Run model on crop
                with torch.no_grad():
                    crop_pred = self._inference_crop(crop_img)
                
                # Remove padding if added for divisibility
                crop_pred = crop_pred[:, :crop_h, :crop_w]
                
                # Accumulate prediction
                pred_accum[:, y1:y2, x1:x2] += crop_pred
                count_mat[:, y1:y2, x1:x2] += 1
        
        # Average predictions
        pred_accum = pred_accum / count_mat.clamp(min=1)
        
        # Remove padding
        pred_accum = pred_accum[:, :orig_height, :orig_width]
        
        # Resize to original size if needed (shouldn't be needed if loader handles resize)
        if pred_accum.shape[1] != orig_height or pred_accum.shape[2] != orig_width:
            pred_accum = F.interpolate(
                pred_accum.unsqueeze(0),
                size=(orig_height, orig_width),
                mode="bilinear",
                align_corners=False,
            )[0]
        
        return {"sem_seg": pred_accum}

    def _inference_crop(self, crop_img):
        """
        Run the model on a single crop.
        
        Args:
            crop_img: Tensor of shape (1, C, H, W), already normalized
            
        Returns:
            Tensor of shape (num_classes, H, W) - logits for each class
        """
        # Get features from backbone
        features = self.model.backbone(crop_img)
        
        # Get predictions from semantic segmentation head
        outputs = self.model.sem_seg_head(features)
        
        mask_cls_results = outputs["pred_logits"]  # (1, num_queries, num_classes+1)
        mask_pred_results = outputs["pred_masks"]  # (1, num_queries, H/4, W/4)
        
        # Upsample masks to crop size
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=crop_img.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        
        # Compute semantic segmentation from mask classification
        # This follows the same logic as the original model
        mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]  # Remove no-object class
        mask_pred = mask_pred_results.sigmoid()
        
        # Combine: (1, num_queries, num_classes) x (1, num_queries, H, W) -> (1, num_classes, H, W)
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        
        return semseg[0]  # Remove batch dimension


def build_slide_evaluator(cfg, model):
    """
    Factory function to create a sliding window evaluator.
    
    Reads slide parameters from cfg if available, otherwise uses defaults
    matching MMSegmentation's common settings.
    """
    # Try to get parameters from config, use MMCV defaults if not present
    crop_size = getattr(cfg.TEST, 'SLIDE_CROP_SIZE', (512, 512))
    stride = getattr(cfg.TEST, 'SLIDE_STRIDE', (341, 341))
    
    return SemanticSegmentorWithSlide(
        cfg=cfg,
        model=model,
        crop_size=crop_size,
        stride=stride,
    )
