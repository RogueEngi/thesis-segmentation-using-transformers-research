# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_reduce_zero_label_dataset_mapper import (
    MaskFormerSemanticReduceZeroLabelDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_multiscale_dataset_mapper import (
    MaskFormerSemanticMultiScaleDatasetMapper,
)

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA
from .sliding_window_inference import SemanticSegmentorWithSlide, build_slide_evaluator

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.reduce_zero_label_sem_seg_evaluator import (
    ReduceZeroLabelSemSegEvaluator,
    build_reduce_zero_label_sem_seg_evaluator,
)
