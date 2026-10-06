# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def boundary_loss(
        inputs: torch.Tensor,
        dist_maps: torch.Tensor,
        num_masks: float,
        normalize: bool = False,
        normalize_scale: float = 10.0,
    ):
    """
    Compute the boundary loss using signed distance maps.
    Based on Kervadec et al. "Boundary loss for highly unbalanced segmentation" (MIDL 2019)
    
    Args:
        inputs: A float tensor of shape (N, num_points). Raw logits (before sigmoid).
                The predictions for each mask at sampled points.
        dist_maps: A float tensor with the same shape as inputs. 
                   Signed distance transform values at sampled points.
                   Negative inside GT, positive outside GT, zero at boundary.
        num_masks: Number of masks for normalization.
        normalize: If True, apply tanh normalization to distance maps.
        normalize_scale: Scale factor for normalization: tanh(dist / scale).
    Returns:
        Loss tensor (scalar)
    """
    # Apply sigmoid to get probabilities
    probs = inputs.sigmoid()
    
    # Optionally normalize distance maps to [-1, 1] range
    if normalize:
        dist_maps = torch.tanh(dist_maps / normalize_scale)
    
    # Boundary loss: probability * signed distance
    # - High prob inside GT (negative dist) → negative loss (good)
    # - High prob outside GT (positive dist) → positive loss (bad)
    loss = (probs * dist_maps).mean(1)  # Mean over points per mask
    return loss.sum() / num_masks


# Note: JIT compilation doesn't work well with optional arguments,
# so we create a simpler version for the default case
def boundary_loss_simple(
        inputs: torch.Tensor,
        dist_maps: torch.Tensor,
        num_masks: float,
    ):
    """Simple boundary loss without normalization (for JIT compatibility)."""
    probs = inputs.sigmoid()
    loss = (probs * dist_maps).mean(1)
    return loss.sum() / num_masks


def boundary_loss_weighted(
        inputs: torch.Tensor,
        dist_maps: torch.Tensor,
        class_labels: torch.Tensor,
        class_weights: torch.Tensor,
        num_masks: float,
        normalize: bool = False,
        normalize_scale: float = 10.0,
    ):
    """
    Compute per-class weighted boundary loss.
    
    Args:
        inputs: A float tensor of shape (N, num_points). Raw logits.
        dist_maps: A float tensor of shape (N, num_points). Signed distance values.
        class_labels: A long tensor of shape (N,). Class label for each mask.
        class_weights: A float tensor of shape (num_classes,). Weight for each class.
        num_masks: Number of masks for normalization.
        normalize: If True, apply tanh normalization to distance maps.
        normalize_scale: Scale factor for normalization.
    Returns:
        Loss tensor (scalar)
    """
    # Apply sigmoid to get probabilities
    probs = inputs.sigmoid()
    
    # Optionally normalize distance maps to [-1, 1] range
    if normalize:
        dist_maps = torch.tanh(dist_maps / normalize_scale)
    
    # Compute per-mask loss: mean over points
    per_mask_loss = (probs * dist_maps).mean(1)  # Shape: (N,)
    
    # Get per-class weights for each mask
    mask_weights = class_weights[class_labels]  # Shape: (N,)
    
    # Apply class weights and sum
    weighted_loss = (per_mask_loss * mask_weights).sum() / num_masks
    
    return weighted_loss


boundary_loss_jit = torch.jit.script(
    boundary_loss_simple
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, 
                 boundary_loss_enabled=False,
                 boundary_warmup_iters=0,
                 boundary_rampup_iters=0,
                 boundary_normalize=False,
                 boundary_normalize_scale=10.0,
                 boundary_class_weights=None):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boundary_loss_enabled: whether to compute boundary loss (requires distance maps in targets)
            boundary_warmup_iters: number of iterations before enabling boundary loss (curriculum learning)
            boundary_rampup_iters: number of iterations to gradually ramp up boundary loss weight after warmup
            boundary_normalize: whether to normalize distance maps with tanh
            boundary_normalize_scale: scale factor for tanh normalization
            boundary_class_weights: list of per-class weights for boundary loss (None = uniform)
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        # boundary loss configuration
        self.boundary_loss_enabled = boundary_loss_enabled
        self.boundary_warmup_iters = boundary_warmup_iters
        self.boundary_rampup_iters = boundary_rampup_iters
        self.boundary_normalize = boundary_normalize
        self.boundary_normalize_scale = boundary_normalize_scale
        
        # Per-class boundary weights
        if boundary_class_weights is not None and len(boundary_class_weights) > 0:
            self.register_buffer("boundary_class_weights", 
                                 torch.tensor(boundary_class_weights, dtype=torch.float32))
        else:
            self.boundary_class_weights = None
        
        # Track current iteration for curriculum learning
        self.register_buffer("current_iter", torch.tensor(0, dtype=torch.long))
        
        if boundary_loss_enabled:
            logger = logging.getLogger(__name__)
            logger.info(f"[SetCriterion] Boundary loss enabled")
            logger.info(f"[SetCriterion]   - warmup_iters: {boundary_warmup_iters}")
            if boundary_rampup_iters > 0:
                logger.info(f"[SetCriterion]   - rampup_iters: {boundary_rampup_iters}")
                logger.info(f"[SetCriterion]   - full weight at iter: {boundary_warmup_iters + boundary_rampup_iters}")
            logger.info(f"[SetCriterion]   - normalize: {boundary_normalize}")
            if boundary_normalize:
                logger.info(f"[SetCriterion]   - normalize_scale: {boundary_normalize_scale}")
            if self.boundary_class_weights is not None:
                logger.info(f"[SetCriterion]   - class_weights: {boundary_class_weights}")

    def get_boundary_weight_multiplier(self):
        """
        Compute the current boundary loss weight multiplier based on curriculum learning schedule.
        
        Returns:
            float: Multiplier in [0, 1] range
                - 0 during warmup period (iter < warmup_iters)
                - Linearly increases from 0 to 1 during ramp-up period
                - 1 after ramp-up is complete
        """
        current = self.current_iter.item()
        
        # During warmup: no boundary loss
        if current < self.boundary_warmup_iters:
            return 0.0
        
        # After warmup, check if we have ramp-up
        if self.boundary_rampup_iters <= 0:
            # No ramp-up: instant full weight
            return 1.0
        
        # During ramp-up: linear increase
        elapsed_after_warmup = current - self.boundary_warmup_iters
        if elapsed_after_warmup >= self.boundary_rampup_iters:
            return 1.0
        
        # Linear interpolation
        return elapsed_after_warmup / self.boundary_rampup_iters

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }
        
        # Compute boundary loss if enabled and distance maps are available
        # Get current weight multiplier (handles warmup + ramp-up schedule)
        boundary_weight_mult = self.get_boundary_weight_multiplier() if self.boundary_loss_enabled else 0.0
        boundary_active = (
            self.boundary_loss_enabled 
            and targets[0].get("dist_maps") is not None
            and boundary_weight_mult > 0
        )
        
        if boundary_active:
            dist_maps = [t["dist_maps"] for t in targets]
            target_dist_maps, _ = nested_tensor_from_tensor_list(dist_maps).decompose()
            target_dist_maps = target_dist_maps.to(src_masks)
            target_dist_maps = target_dist_maps[tgt_idx]
            
            # Sample distance maps at the same points
            target_dist_maps = target_dist_maps[:, None]  # N x 1 x H x W
            with torch.no_grad():
                point_dist = point_sample(
                    target_dist_maps,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)
            
            # Compute boundary loss (with optional per-class weighting)
            if self.boundary_class_weights is not None:
                # Get class labels for each mask
                target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
                raw_boundary_loss = boundary_loss_weighted(
                    point_logits, point_dist, target_classes,
                    self.boundary_class_weights, num_masks,
                    normalize=self.boundary_normalize,
                    normalize_scale=self.boundary_normalize_scale
                )
            elif self.boundary_normalize:
                raw_boundary_loss = boundary_loss(
                    point_logits, point_dist, num_masks,
                    normalize=True, 
                    normalize_scale=self.boundary_normalize_scale
                )
            else:
                raw_boundary_loss = boundary_loss_jit(point_logits, point_dist, num_masks)
            
            # Apply gradual ramp-up multiplier (weight_dict handles base weight, this handles schedule)
            losses["loss_boundary"] = raw_boundary_loss * boundary_weight_mult
        elif self.boundary_loss_enabled and boundary_weight_mult == 0:
            # During warmup/early ramp-up, output zero boundary loss so it appears in logs
            losses["loss_boundary"] = point_logits.new_tensor(0.0)

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def set_current_iter(self, iter_num):
        """Update current iteration for curriculum learning."""
        self.current_iter.fill_(iter_num)

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
            "boundary_loss_enabled: {}".format(self.boundary_loss_enabled),
        ]
        if self.boundary_loss_enabled:
            body.extend([
                "boundary_warmup_iters: {}".format(self.boundary_warmup_iters),
                "boundary_normalize: {}".format(self.boundary_normalize),
            ])
            if self.boundary_normalize:
                body.append("boundary_normalize_scale: {}".format(self.boundary_normalize_scale))
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
