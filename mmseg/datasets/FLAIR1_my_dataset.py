# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class FLAIR1_my_dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('building', 'pervious surface', 'impervious surface', 'bare soil', 'water', 'coniferous',
               'deciduous', 'brushwood', 'vineyard', 'herbaceous vegetation', 'agricultural land', 'plowed land'),
        palette=[[219, 14, 154], [147, 142, 123], [248, 12, 0], [169, 113, 1], [21, 83, 174], [25, 74, 38],
               [70, 228, 131], [243, 166, 13], [102, 0, 130], [85, 255, 0], [255, 243, 13], [228, 223, 124]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
