from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from mmcls.models.heads.base_head import BaseHead
from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample


@MODELS.register_module()
class LinearRegHead(BaseHead):
    """Linear regression head.

    Args:
        num_classes (int): Number of targets.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='RMSELoss', loss_weight=1.0)``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss: dict = dict(type='RMSELoss', loss_weight=1.0),
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01)):
        super(LinearRegHead, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        reg_score = self.fc(pre_logits)
        return reg_score

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ClsDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.
        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[ClsDataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_label.score for i in data_samples])
        else:
            raise NotImplementedError

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        return losses

    def predict(
            self,
            feats: Tuple[torch.Tensor],
            data_samples: List[ClsDataSample] = None) -> List[ClsDataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[ClsDataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.
        Returns:
            List[ClsDataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head."""
        pred_scores = cls_score

        if data_samples is not None:
            for data_sample, score in zip(data_samples, pred_scores):
                data_sample.set_pred_score(score)
        else:
            data_samples = []
            for score in zip(pred_scores):
                data_samples.append(ClsDataSample().set_pred_score(score))

        return data_samples
