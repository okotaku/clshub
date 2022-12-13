# flake8: noqa:E501
import os

import mmengine

from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS


@DATASETS.register_module()
class CSVRegression(BaseDataset):
    METAINFO = {'classes': ['target']}

    def __init__(self,
                 *args,
                 id_col: str = 'Id',
                 target_div_val=None,
                 target_log1p: bool = False,
                 debug: bool = False,
                 suffix: str = '.jpg',
                 **kwargs):
        self.id_col = id_col
        self.target_div_val = target_div_val
        self.target_log1p = target_log1p
        self.suffix = suffix
        super().__init__(*args, **kwargs)
        self.debug = debug

    def __len__(self) -> int:
        if self.debug:
            return 200
        elif self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    def load_data_list(self):
        df = mmengine.load(self.ann_file)

        if self.target_div_val:
            df[self.CLASSES] = df[self.CLASSES] / self.target_div_val
        elif self.target_log1p:
            df[self.CLASSES] = np.log1p(df[self.CLASSES])

        data_infos = []
        for image_id, gt_label in zip(df[self.id_col].values,
                                      df[self.CLASSES].values):
            filename = f'{image_id}{self.suffix}'
            import numpy as np
            info = {
                'img_path': os.path.join(self.img_prefix, filename),
                'gt_label': gt_label,
            }
            data_infos.append(info)
        return data_infos
