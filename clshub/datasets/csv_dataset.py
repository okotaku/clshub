# flake8: noqa:E501
import os

import mmengine

from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS


@DATASETS.register_module()
class CSVDataset(BaseDataset):

    def __init__(self,
                 *args,
                 id_col: str = 'Id',
                 label_col: str = 'label',
                 debug: bool = False,
                 **kwargs):
        self.id_col = id_col
        self.label_col = label_col
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

        data_infos = []
        for filename, gt_label in zip(df[self.id_col].values,
                                      df[self.label_col].values):
            info = {
                'img_path': os.path.join(self.img_prefix, filename),
                'gt_label': int(gt_label),
            }
            data_infos.append(info)
        return data_infos
