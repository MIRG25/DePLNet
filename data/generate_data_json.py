import os
import json
from config import config

if __name__ == '__main__':
    # 数据集的路径
    output_file = './brats2021_fold_5.json'

    train_data_list = []
    test_data_list = []

    tr_list_path = 'data_split/fold_5_train.txt'
    tt_list_path = 'data_split/fold_5_test.txt'
    with open(tr_list_path,'r') as f:
        tr_list = f.read().splitlines()

    with open(tt_list_path,'r') as f:
        tt_list = f.read().splitlines()

    for i in tr_list:
        data = {
            "image": [
                f"{i}/{i}_flair.nii.gz",
                f"{i}/{i}_t1ce.nii.gz",
                f"{i}/{i}_t1.nii.gz",
                f"{i}/{i}_t2.nii.gz"
            ],
            "label": f"{i}/{i}_seg.nii.gz"
        }
        train_data_list.append(data)

    for i in tt_list:
        data = {
            "image": [
                f"{i}/{i}_flair.nii.gz",
                f"{i}/{i}_t1ce.nii.gz",
                f"{i}/{i}_t1.nii.gz",
                f"{i}/{i}_t2.nii.gz"
            ],
            "label": f"{i}/{i}_seg.nii.gz"
        }
        test_data_list.append(data)

    json_data = {
        "train": train_data_list,
        "test": test_data_list
    }

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

    print("done")



