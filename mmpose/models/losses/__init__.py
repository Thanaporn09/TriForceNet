# Copyright (c) OpenMMLab. All rights reserved.
from .classfication_loss import BCELoss
from .heatmap_loss import AdaptiveWingLoss
from .mesh_loss import GANLoss, MeshLoss
from .mse_loss import JointsMSELoss, JointsOHKMMSELoss,JointsBCELoss
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss,
                              SemiSupervisionLoss, SmoothL1Loss, SoftWingLoss,
                              WingLoss)
from .cross_entropy_loss import CrossEntropyLoss

__all__ = [
    'JointsMSELoss', 'JointsOHKMMSELoss', 'HeatmapLoss', 'AELoss',
    'MultiLossFactory', 'MeshLoss', 'GANLoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss','JointsBCELoss','CrossEntropyLoss'
]
