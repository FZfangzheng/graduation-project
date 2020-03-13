from pathlib import Path
import numpy as np
import rasterio

from collections import OrderedDict
from osgeo import gdal
from utils import get_logger
import warnings





REF_PREFIX_1 = '00'
PRE_PREFIX = '01'
REF_PREFIX_2 = '02'
COARSE_PREFIX = 'M'
FINE_PREFIX = 'L'
SCALE = 1
def get_pair_path(im_dir, n_refs):
    # 将一组数据集按照规定的顺序组织好
    paths = []
    order = OrderedDict()
    order[0] = REF_PREFIX_1 + '_' + COARSE_PREFIX
    order[1] = REF_PREFIX_1 + '_' + FINE_PREFIX
    order[2] = PRE_PREFIX + '_' + COARSE_PREFIX
    order[3] = PRE_PREFIX + '_' + FINE_PREFIX

    if n_refs == 2:
        order[2] = REF_PREFIX_2 + '_' + COARSE_PREFIX
        order[3] = REF_PREFIX_2 + '_' + FINE_PREFIX
        order[4] = PRE_PREFIX + '_' + COARSE_PREFIX
        order[5] = PRE_PREFIX + '_' + FINE_PREFIX

    for prefix in order.values():
        for path in Path(im_dir).glob('*.tif'):
            if path.name.startswith(prefix):
                paths.append(path.expanduser().resolve())
                break

    if n_refs == 2:
        assert len(paths) == 6 or len(paths) == 5, len(paths)
    else:
        assert len(paths) == 3 or len(paths) == 4, len(paths)
    return paths


def load_image_pair(im_dir, n_refs):
    logger = get_logger()
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_pair_path(im_dir, n_refs)
    # 将组织好的数据转为Image对象
    images = []

    for p in paths:
        logger.info(f'There are load_image_pair 62.')
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        gdal.UseExceptions()
        logger.info(str(p))
        ds = gdal.Open(str(p))
        logger.info(f'There are load_image_pair 66.')
        im = ds.ReadAsArray().astype(np.float32)  # C*H*W (numpy.ndarray)
        logger.info(f'There are load_image_pair 68.')
        images.append(im)
        del ds

    # 对数据的尺寸进行验证
    assert images[0].shape[1] * SCALE == images[1].shape[1]
    assert images[0].shape[2] * SCALE == images[1].shape[2]
    return images

warnings.filterwarnings('ignore')
load_image_pair("/home/dataset/train/1", 1)