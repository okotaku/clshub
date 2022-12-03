# flake8: noqa:E501
import os

import mmengine

from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS


@DATASETS.register_module()
class RSNA2022(BaseDataset):
    CLASSES = ['cancer', 'biopsy', 'invasive', 'difficult_negative_case']

    def __init__(self, *args, debug: bool = False, **kwargs):
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
        df['prediction_id'] = df['patient_id'].astype(
            str) + '_' + df['laterality'].astype(str)

        data_infos = []
        for image_id, targets, prediction_id in zip(df.image_id,
                                                    df[self.CLASSES].values,
                                                    df.prediction_id):
            filename = f'{image_id}.jpg'
            labels = set()
            for c, target in enumerate(targets):
                if target == 1:
                    labels.add(c)
            info = {
                'img_path': os.path.join(self.img_prefix, filename),
                'gt_label': list(labels),
                'prediction_id': prediction_id
            }
            data_infos.append(info)
        return data_infos
