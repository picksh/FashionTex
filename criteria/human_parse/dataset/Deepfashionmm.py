# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.datasets import DATASETS
from mmseg.datasets import CustomDataset

@DATASETS.register_module()
class Deepfashionmm(CustomDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """
    '''
    CLASSES = ('background', 'top','outer','skirt','dress','pants','leggings','headwear',
                'eyeglass','neckwear','belt','footwear','bag','hair','face','skin',
                'ring','wrist wearing','socks','gloves','necklace','rompers','earrings','tie')
    
    PALETTE = [[0, 0, 0], [121, 94, 255], [139,69 ,19],
                [250, 235, 215], [255, 250, 205], [70,130, 180],
                [70, 130, 180], [127, 255, 212], [0, 100, 0],
                [50, 205, 50], [255, 255, 0], [138 ,43 ,226],
                [255, 140, 0], [255, 0, 0], [16, 78, 139],
                [144, 238, 144], [50, 205, 174], [50, 155, 250],
                [160, 140, 88], [213, 140, 88], [90, 140, 90],
                [185, 210, 205], [130, 165, 180], [225, 141, 151]]
    ''' 
    CLASSES=('background', 'top','background',
              'background','background','background',
              'background','background','background',
              'background','background','background',
              'background','background','background',
              'background','background','background',
              'background','background','background',
              'background','background','background')
    PALETTE=[[0, 0, 0], [121, 94, 255],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],[0, 0, 0],
             [0, 0, 0],[0, 0, 0],[0, 0, 0]]

    
    '''
    CLASSES=('background','top')
    PALETTE = [[120, 120, 120], [6, 230, 230]]
    '''
    def __init__(self, **kwargs):
        super(Deepfashionmm, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_segm.png',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.img_dir)