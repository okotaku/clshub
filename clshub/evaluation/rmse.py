from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from torch import nn

from mmcls.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


@METRICS.register_module()
class RMSE(BaseMetric):
    default_prefix: Optional[str] = 'rmse'

    def __init__(self,
                 collect_device: str = 'cpu',
                 target_div_val=None,
                 target_log1p: bool = False,
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.target_div_val = target_div_val
        self.target_log1p = target_log1p

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']
            result['pred_score'] = pred_label['score'].cpu()
            result['gt_label'] = gt_label['score'].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.
        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if self.target_div_val:
            target = target * self.target_div_val
        elif self.target_log1p:
            target = torch.expm1(target)
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])

            try:
                rmse = self.calculate(pred, target)
            except ValueError as e:
                # If the topk is invalid.
                raise ValueError(
                    str(e) + ' Please check the `val_evaluator` and '
                    '`test_evaluator` fields in your config file.')

            metrics['score'] = rmse.item()
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])
            rmse = self.calculate(pred, target)
            metrics['score'] = rmse.item()

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'

        mse = nn.MSELoss()
        rmse = torch.sqrt(mse(pred.view(-1), target.view(-1)))
        return rmse
