from pathlib import Path
import numpy as np
import rasterio
from timeit import default_timer as timer
from osgeo import gdal
import math
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from utils import make_tuple
from utils import get_logger


root_dir = Path(__file__).parents[1]
data_dir = root_dir / 'data'

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


def load_image_pair(images, im_dir, n_refs):
    logger = get_logger()
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_pair_path(im_dir, n_refs)
    # 将组织好的数据转为Image对象

    for p in paths:
        # logger.info(f'There are load_image_pair 62.')
        # gdal.PushErrorHandler('CPLQuietErrorHandler')
        # gdal.UseExceptions()
        # with rasterio.open(str(p)) as ds:
        #     logger.info(f'There are load_image_pair 66.')
        #     im = ds.read().astype(np.float32)  # C*H*W (numpy.ndarray)
        #     logger.info(f'There are load_image_pair 68.')
        #     images.append(im)
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        gdal.UseExceptions()
        ds = gdal.Open(str(p))
        im = ds.ReadAsArray().astype(np.float32)  # C*H*W (numpy.ndarray)
        images.append(im)
        del ds

    # 对数据的尺寸进行验证
    assert images[0].shape[1] * SCALE == images[1].shape[1]
    assert images[0].shape[2] * SCALE == images[1].shape[2]


def im2tensor(im):
    im = torch.from_numpy(im)
    out = im.mul_(0.0001)
    return out


class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    Pillow中的Image是列优先，而Numpy中的ndarray是行优先
    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None, n_refs=1):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        if not patch_stride:
            patch_stride = patch_size
        else:
            patch_stride = make_tuple(patch_stride)
        self.logger = get_logger()
        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.refs = n_refs

        self.image_dirs = [p for p in self.root_dir.glob('*') if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)

        # 计算出图像进行分块以后的patches的数目
        self.num_patches_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        self.num_patches_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])

        self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y
        # self.num_patches = self.num_im_pairs
        self.transform = im2tensor
        self.now_index = -1
        self.images = []

    def map_index(self, index):
        # 将全局的index映射到具体的图像对文件夹索引(id_n)，图像裁剪的列号与行号(id_x, id_y)
        id_n = index // (self.num_patches_x * self.num_patches_y)
        residual = index % (self.num_patches_x * self.num_patches_y)
        id_x = self.patch_stride[0] * (residual % self.num_patches_x)
        id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        t_start = timer()
        id_n, id_x, id_y = self.map_index(index)
        if id_n != self.now_index:
            self.images.clear()
            load_image_pair(self.images, self.image_dirs[id_n], self.refs)
            self.now_index = id_n
        patches = [None] * len(self.images)
        scales = [1, SCALE]
        for i in range(len(patches)):
            scale = scales[i % 2]
            im = self.images[i][:,
                 id_x * scale:(id_x + self.patch_size[0]) * scale,
                 id_y * scale:(id_y + self.patch_size[1]) * scale]
            patches[i] = self.transform(im)

        t_end = timer()
        self.logger.info(f'spend on read images: {t_end - t_start}s')
        return patches

    def __len__(self):
        return self.num_patches
