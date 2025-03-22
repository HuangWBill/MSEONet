# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class Vaihingen_my_dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('impervious_surface', 'building', 'low_vegetation', 'tree',
                 'car', 'clutter'),
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 0], [255, 0, 0]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
