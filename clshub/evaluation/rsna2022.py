from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from mmengine.evaluator import BaseMetric
from mmengine.structures import LabelData

from mmcls.evaluation.metrics.single_label import to_tensor
from mmcls.registry import METRICS


def _precision_recall_f1_support(pred, gt_positive):
    """calculate base classification task metrics, such as  precision, recall,
    f1_score, support."""

    # ignore -1 target such as difficult sample that is not wanted
    # in evaluation results.
    # only for calculate multi-label without affecting single-label behavior
    ignored_index = gt_positive == -1
    pred[ignored_index] = 0
    gt_positive[ignored_index] = 0

    ctp = pred * gt_positive
    cfp = pred * (1 - gt_positive)
    tp_sum = ctp.sum(0)
    pred_sum = ctp.sum(0) + cfp.sum(0)
    gt_sum = gt_positive.sum(0)

    precision = tp_sum / torch.clamp(pred_sum, min=1).float() * 100
    recall = tp_sum / torch.clamp(gt_sum, min=1).float() * 100
    f1_score = 2 * precision * recall / torch.clamp(
        precision + recall, min=torch.finfo(torch.float32).eps)
    support = gt_sum
    return precision, recall, f1_score, support


@METRICS.register_module()
class RSNA2022(BaseMetric):
    default_prefix: Optional[str] = 'rsna2022'

    def __init__(self,
                 items: Sequence[str] = ('precision', 'recall', 'f1-score'),
                 collect_device: str = 'cpu',
                 groupby: bool = True,
                 prefix: Optional[str] = None) -> None:

        for item in items:
            assert item in ['precision', 'recall', 'f1-score', 'support'], \
                f'The metric {item} is not supported by `SingleLabelMetric`,' \
                ' please choose from "precision", "recall", "f1-score" and ' \
                '"support".'
        self.items = tuple(items)
        self.groupby = groupby
        self.classes = [
            'cancer', 'biopsy', 'invasive', 'difficult_negative_case'
        ]

        super().__init__(collect_device=collect_device, prefix=prefix)

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

            result['pred_score'] = pred_label['score'].clone()
            num_classes = result['pred_score'].size()[-1]

            if 'score' in gt_label:
                result['gt_score'] = gt_label['score'].clone()
            else:
                result['gt_score'] = LabelData.label_to_onehot(
                    gt_label['label'], num_classes)

            result['prediction_id'] = data_sample['prediction_id']

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.

        target = torch.stack([res['gt_score'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])

        def groupby_mean(x, prediction_id):
            x = pd.DataFrame(x)
            x = x.groupby(prediction_id).mean().values
            x = torch.Tensor(x)
            return x

        if self.groupby:
            prediction_id = [res['prediction_id'] for res in results]
            target = groupby_mean(target.numpy(), prediction_id)
            pred = groupby_mean(pred.numpy(), prediction_id)

        metric_res = self.calculate(
            pred, target, pred_indices=False, target_indices=False)

        def pack_results(precision, recall, f1_score, support):
            single_metrics = {}
            if 'precision' in self.items:
                single_metrics['precision'] = precision
            if 'recall' in self.items:
                single_metrics['recall'] = recall
            if 'f1-score' in self.items:
                single_metrics['f1-score'] = f1_score
            if 'support' in self.items:
                single_metrics['support'] = support
            return single_metrics

        result_metrics = dict()
        for k, v in pack_results(*metric_res).items():
            for i, c in enumerate(self.classes):
                result_metrics[f'{k}_{c}'] = v[i].item()
        return result_metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        pred_indices: bool = False,
        target_indices: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate the precision, recall, f1-score.
        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, num_classes)`` or a sequence of index/onehot
                format labels.
            target (torch.Tensor | np.ndarray | Sequence): The prediction
                results. A :obj:`torch.Tensor` or :obj:`np.ndarray` with
                shape ``(N, num_classes)`` or a sequence of index/onehot
                format labels.
            pred_indices (bool): Whether the ``pred`` is a sequence of
                category index labels. If True, ``num_classes`` must be set.
                Defaults to False.
            target_indices (bool): Whether the ``target`` is a sequence of
                category index labels. If True, ``num_classes`` must be set.
                Defaults to False.
        Returns:
            Tuple: The tuple contains precision, recall and f1-score.
            And the type of each item is:
            - torch.Tensor: A tensor for each metric. The shape is (1, ) if
              ``average`` is not None, and (C, ) if ``average`` is None.
        """

        def _format_label(label, is_indices):
            """format various label to torch.Tensor."""
            if isinstance(label, np.ndarray):
                assert label.ndim == 2, 'The shape `pred` and `target` ' \
                    'array must be (N, num_classes).'
                label = torch.from_numpy(label)
            elif isinstance(label, torch.Tensor):
                assert label.ndim == 2, 'The shape `pred` and `target` ' \
                    'tensor must be (N, num_classes).'
            elif isinstance(label, Sequence):
                if is_indices:
                    raise NotImplementedError
                else:
                    label = torch.stack(
                        [to_tensor(onehot) for onehot in label])
            else:
                raise TypeError(
                    'The `pred` and `target` must be type of torch.tensor or '
                    f'np.ndarray or sequence but get {type(label)}.')
            return label

        pred = _format_label(pred, pred_indices)
        target = _format_label(target, target_indices).long()

        assert pred.shape == target.shape, \
            f"The size of pred ({pred.shape}) doesn't match "\
            f'the target ({target.shape}).'

        return _precision_recall_f1_support(pred, target)
