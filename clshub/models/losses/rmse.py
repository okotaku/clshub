from typing import Optional

import torch
from torch import Tensor, nn

from mmcls.registry import MODELS


@MODELS.register_module()
class RMSELoss(nn.Module):

    def __init__(self, reduction: str = 'mean', loss_weight: int = 1.0):
        super().__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self,
                yhat: Tensor,
                y: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None):
        return torch.sqrt(self.mse(yhat.view(-1),
                                   y.view(-1))) * self.loss_weight
