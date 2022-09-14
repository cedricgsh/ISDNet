import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DeepGlobeDataset(CustomDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """

    
    CLASSES = ('unknown', 'urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren')

    PALETTE = [[0, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], 
               [0, 0, 255], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(DeepGlobeDataset, self).__init__(
            img_suffix='_sat.jpg',
            seg_map_suffix='_mask.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)