import os
import json
import random

import numpy as np
from PIL import Image
import torch
from monai.config import NdarrayOrTensor, KeysCollection
from monai.utils import TransformBackends
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Hashable, Mapping
from monai.data import CacheDataset
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    CropForegroundd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    ToTensord,
    ConvertToMultiChannelBasedOnBratsClassesd,
    SaveImage, Transform
)
from monai import data

from config import config
from data.loader import TwoStreamBatchSampler

# 创建自定义数据集类来返回索引
class IndexedCacheDataset(data.CacheDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        data['idx'] = index  # 添加索引到返回的数据中
        return data


class ConvertToMultiChannel(Transform):
    """
    Convert labels to multi channels based on `brats18 <https://www.med.upenn.edu/sbia/brats2018/data.html>`_ classes,
    which include TC (Tumor core), WT (Whole tumor) and ET (Enhancing tumor):
    label 1 is the necrotic and non-enhancing tumor core, which should be counted under TC and WT subregion,
    label 2 is the peritumoral edema, which is counted only under WT subregion,
    label 4 is the GD-enhancing tumor, which should be counted under ET, TC, WT subregions.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [img == 1, img == 2, img == 4]

        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

class LabelConvertToMultiChannel(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)

            result = [img == 1, img == 2, img == 4]

            d[key] = torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)
        return d

def simple_collate(batch):
    """简单的 collate 函数，返回普通 Tensor"""
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: simple_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    else:
        return batch

def split_data(datalist, basedir, type):
    """
    从JSON文件加载数据列表
    支持新格式：包含labeled/unlabeled/test三个字段
    """
    # 检查JSON格式
    with open(datalist) as f:
        json_data = json.load(f)

    data_list = json_data[type]

    data = []
    for d in data_list:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
        data.append(d)

    return data

def worker_init_fn(worker_id):
    seed = 1024
    random.seed(seed + worker_id)

def get_loader_BraTS2021(batch_size, data_dir, json_list, roi, type):
    transform_brats2021 = {
        'train': Compose([
            LoadImaged(keys=["image", "label"], image_only=True),
            LabelConvertToMultiChannel(keys="label"),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
                allow_smaller=False
            ),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        ]),

        'valid': Compose([
            LoadImaged(keys=["image", "label"], image_only=True),
            LabelConvertToMultiChannel(keys="label"),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]),

        'test': Compose([
            LoadImaged(keys=["image", "label"], image_only=True),
            LabelConvertToMultiChannel(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]),
    }

    data_list = split_data(json_list, data_dir, type)

    # 使用自定义的IndexedCacheDataset
    dataset = IndexedCacheDataset(
        data=data_list,
        cache_rate=0.0,
        transform=transform_brats2021[type]
    )

    if type == 'train':
        # 读取JSON获取labeled和unlabeled数量
        with open(json_list) as f:
            json_data = json.load(f)

        if 'labeled' in json_data and 'unlabeled' in json_data:
            # 新格式：直接使用labeled和unlabeled的数量
            labeled_num = len(json_data['labeled'])
            unlabeled_num = len(json_data['unlabeled'])
            total_num = labeled_num + unlabeled_num

            print(f"Dataset: {labeled_num} labeled + {unlabeled_num} unlabeled = {total_num} total")
            print(f"Labeled ratio: {labeled_num / total_num * 100:.1f}%")

            labeled_idxs = list(range(labeled_num))
            unlabeled_idxs = list(range(labeled_num, total_num))
        else:
            # 旧格式兼容：按比例划分
            labeled_num = int(config["labeled_rate"] / 100 * len(data_list))
            labeled_idxs = list(range(labeled_num))
            unlabeled_idxs = list(range(labeled_num, len(data_list)))

        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, config["batch_size"],
                                              config["batch_size"] - config["labeled_bs"])
        data_loader = data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            worker_init_fn=worker_init_fn,
            collate_fn=simple_collate
            )
    else:
        data_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            collate_fn=simple_collate
        )

    return data_loader


if __name__ == "__main__":
    data_dir = config["base_path"]
    json_list = config["json_path"]
    roi = (128, 128, 128)
    batch_size = 1
    data_loader = get_loader_BraTS2021(batch_size, config["base_path"], config["json_path"], config['image_shape'], 'valid')
    for batch_data in data_loader:
        data, target = batch_data["image"], batch_data["label"]
        print(data.shape)
        print(target.shape)

    print("dataset:{0}".format(len(data_loader)))
