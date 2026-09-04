import random
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import time

import numpy as np
import torch
from torch.backends import cudnn
from data.BraTS21 import get_loader_BraTS2021
from networks.MyNet_MoE.MyNet import train_MyNet_MoE
from config import config
from networks.VNet.VNet import train_VNet

parser = argparse.ArgumentParser()
parser.add_argument('--user', default='cmh', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--deterministic', default=1, type=int)
parser.add_argument('--num_worker', default=8, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 打印配置信息
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Network: {config['net']}")
    print(f"Labeled Rate: {config['labeled_rate']}%")
    print(f"Use MoE: {config['use_moe']}")
    print(f"Use VC Gating: {config['use_vc_gating']}")
    print(f"Use DSP: {config['use_dsp']}")
    print(f"Max Iterations: {config['max_iterations']}")
    print(f"{'='*60}\n")

    # 5折交叉验证
    for fold in range(1, 6):
        print(f"\n{'='*60}")
        print(f"Starting Fold {fold}/5")
        print(f"{'='*60}\n")

        # 使用新的JSON格式文件
        # json_path = os.path.join('./data', f"{config['dataname']}_{config['labeled_rate']}percent_fold_{fold}.json")
        # json_path = os.path.join('./data', f"{config['dataname']}_fold_3.json")
        json_path = os.path.join('./data', f"{config['dataname']}_80_tmp.json")

        # 检查文件是否存在
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found!")
            print(f"Please run: python data/generate_splits.py --ratios {config['labeled_rate']}")
            print(f"Skipping fold {fold}...")
            continue

        train_loader = get_loader_BraTS2021(
            config["batch_size"],
            config["root"],
            json_path,
            config['image_shape'],
            'train'
        )
        test_loader = get_loader_BraTS2021(
            config["test_batch_size"],
            config["root"],
            json_path,
            config['image_shape'],
            'test'
        )

        start_time = time.time()

        if config["net"] == 'VNet':
            train_VNet(train_loader, test_loader, fold)
        else:
            # 所有MyNet变体都使用train_MyNet_MoE
            train_MyNet_MoE(train_loader, test_loader, fold)

        end_time = time.time()
        training_time = (end_time - start_time) / 60
        print(f"\n{'='*60}")
        print(f"Fold {fold} completed in {training_time:.2f} minutes")
        print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print(f"All folds completed!")
    print(f"{'='*60}\n")