"""
数据划分脚本：支持多种标注数据比例
生成不同比例的有标签/无标签数据划分JSON文件
支持: 5%, 10%, 20%, 100%
"""

import os
import json
import random
import argparse
from pathlib import Path

def split_data_by_ratio(data_root, output_dir, labeled_ratios=[5, 10, 20, 100], num_folds=5, seed=1337):
    """
    按照不同比例划分有标签和无标签数据

    参数:
        data_root: BraTS数据根目录
        output_dir: 输出JSON文件目录
        labeled_ratios: 标注数据比例列表
        num_folds: 交叉验证折数
        seed: 随机种子
    """
    random.seed(seed)

    # 获取所有病例
    data_root = Path(data_root)
    all_cases = []

    for case_dir in data_root.iterdir():
        if case_dir.is_dir() and case_dir.name.startswith('BraTS'):
            # 检查是否包含所有必需的模态
            flair = case_dir / f"{case_dir.name}_flair.nii.gz"
            t1 = case_dir / f"{case_dir.name}_t1.nii.gz"
            t1ce = case_dir / f"{case_dir.name}_t1ce.nii.gz"
            t2 = case_dir / f"{case_dir.name}_t2.nii.gz"
            seg = case_dir / f"{case_dir.name}_seg.nii.gz"

            if all([flair.exists(), t1.exists(), t1ce.exists(), t2.exists(), seg.exists()]):
                all_cases.append(case_dir.name)

    print(f"Found {len(all_cases)} valid cases")

    # 随机打乱
    random.shuffle(all_cases)

    # 为每个比例生成数据划分
    for ratio in labeled_ratios:
        print(f"\n{'='*60}")
        print(f"Generating splits for {ratio}% labeled data")
        print(f"{'='*60}")

        # 计算每折的数据量
        fold_size = len(all_cases) // num_folds

        for fold in range(1, num_folds + 1):
            # 划分训练集和测试集
            test_start = (fold - 1) * fold_size
            test_end = fold * fold_size if fold < num_folds else len(all_cases)

            test_cases = all_cases[test_start:test_end]
            train_cases = all_cases[:test_start] + all_cases[test_end:]

            # 从训练集中划分有标签和无标签数据
            if ratio == 100:
                # 100%标注：所有训练数据都有标签
                labeled_cases = train_cases
                unlabeled_cases = []
            else:
                # 计算有标签数据数量
                num_labeled = max(1, int(len(train_cases) * ratio / 100))
                labeled_cases = train_cases[:num_labeled]
                unlabeled_cases = train_cases[num_labeled:]

            # 构建JSON数据
            data_split = {
                "labeled": [
                    {
                        "image": str(data_root / case / f"{case}_flair.nii.gz"),
                        "t1": str(data_root / case / f"{case}_t1.nii.gz"),
                        "t1ce": str(data_root / case / f"{case}_t1ce.nii.gz"),
                        "t2": str(data_root / case / f"{case}_t2.nii.gz"),
                        "label": str(data_root / case / f"{case}_seg.nii.gz")
                    }
                    for case in labeled_cases
                ],
                "unlabeled": [
                    {
                        "image": str(data_root / case / f"{case}_flair.nii.gz"),
                        "t1": str(data_root / case / f"{case}_t1.nii.gz"),
                        "t1ce": str(data_root / case / f"{case}_t1ce.nii.gz"),
                        "t2": str(data_root / case / f"{case}_t2.nii.gz"),
                        "label": str(data_root / case / f"{case}_seg.nii.gz")  # 无标签数据也保留label路径，但训练时不使用
                    }
                    for case in unlabeled_cases
                ],
                "test": [
                    {
                        "image": str(data_root / case / f"{case}_flair.nii.gz"),
                        "t1": str(data_root / case / f"{case}_t1.nii.gz"),
                        "t1ce": str(data_root / case / f"{case}_t1ce.nii.gz"),
                        "t2": str(data_root / case / f"{case}_t2.nii.gz"),
                        "label": str(data_root / case / f"{case}_seg.nii.gz")
                    }
                    for case in test_cases
                ]
            }

            # 保存JSON文件
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"brats2021_{ratio}percent_fold_{fold}.json")

            with open(output_file, 'w') as f:
                json.dump(data_split, f, indent=2)

            print(f"Fold {fold}: Labeled={len(labeled_cases)}, Unlabeled={len(unlabeled_cases)}, Test={len(test_cases)}")
            print(f"Saved to: {output_file}")

    print(f"\n{'='*60}")
    print("Data split generation completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data splits for different labeled ratios")
    parser.add_argument('--data_root', type=str,
                       default='C:/Users/mingh/my/data/BraTS_2021',
                       help='BraTS data root directory')
    parser.add_argument('--output_dir', type=str,
                       default='./data',
                       help='Output directory for JSON files')
    parser.add_argument('--ratios', type=int, nargs='+',
                       default=[5, 10, 20, 100],
                       help='Labeled data ratios (e.g., 5 10 20 100)')
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=1337,
                       help='Random seed')

    args = parser.parse_args()

    split_data_by_ratio(
        data_root=args.data_root,
        output_dir=args.output_dir,
        labeled_ratios=args.ratios,
        num_folds=args.num_folds,
        seed=args.seed
    )
