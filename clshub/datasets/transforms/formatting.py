import warnings

import numpy as np
from mmcv.transforms import BaseTransform

from mmcls.datasets.transforms.formatting import to_tensor
from mmcls.registry import TRANSFORMS
from mmcls.structures import ClsDataSample


@TRANSFORMS.register_module()
class PackRegInputs(BaseTransform):
    """Pack the inputs data for the regression.

    **Required Keys:**
    - img
    - gt_label (optional)
    - ``*meta_keys`` (optional)
    **Deleted Keys:**
    All keys in the dict.
    **Added Keys:**
    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmcls.structures.ClsDataSample`): The annotation
      info of the sample.
    Args:
        meta_keys (Sequence[str]): The meta keys to be saved in the
            ``metainfo`` of the packed ``data_samples``.
            Defaults to a tuple includes keys:
            - ``sample_idx``: The id of the image sample.
            - ``img_path``: The path to the image file.
            - ``ori_shape``: The original shape of the image as a tuple (H, W).
            - ``img_shape``: The shape of the image after the pipeline as a
              tuple (H, W).
            - ``scale_factor``: The scale factor between the resized image and
              the original image.
            - ``flip``: A boolean indicating if image flip transform was used.
            - ``flip_direction``: The flipping direction.
    """

    def __init__(self,
                 meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)
        else:
            warnings.warn(
                'Cannot get "img" in the input dict of `PackClsInputs`,'
                'please make sure `LoadImageFromFile` has been added '
                'in the data pipeline or images have been loaded in '
                'the dataset.')

        data_sample = ClsDataSample()
        if 'gt_label' in results:
            gt_label = results['gt_label']
            # add gt scores for regression
            data_sample.set_gt_score(gt_label)

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
