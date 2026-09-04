# coding=utf-8
import argparse
import os
import time
import logging

import setproctitle
import torch.optim

from config import config
from data.BraTS21 import get_loader_BraTS2021
from utils import test_all_case
from networks.MyNet_MoE.model_factory import create_model
import random
import numpy as np
import torch

local_time = time.strftime("%Y_%m_%d %H:%M:%S", time.localtime())
parser = argparse.ArgumentParser()
parser.add_argument('--user', default='cmh', type=str)
parser.add_argument('--mode', default='test', type=str)
parser.add_argument('-batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--test_pth', default='iter_1000.pth', type=str)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--num_worker', default=8, type=int)
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--output_dir', default='output', type=str)
parser.add_argument('--submission', default='submission', type=str)

args = parser.parse_args()

def main():
    start_time = time.time()


    model = create_model()

    fold = 1
    load_file = os.path.join('./checkpoint', config["dataname"] + '_' + str(config["labeled_rate"]),
                                       config["net"] + '_' + str(fold), args.test_pth)

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint)
        print('Successfully load checkpoint {}'.format(load_file))
    else:
        print('There is no resume file to load!')
    json_path = os.path.join(f'./data/brats2021_fold_{fold}.json')

    test_loader = get_loader_BraTS2021(config["test_batch_size"], config["root"], json_path, config['image_shape'],'test')
    # import time
    # save_seg_file = os.path.join('./seg_result', config["dataname"] + '_' + str(config["labeled_rate"]),
    #                              config["net"])
    current_time = time.strftime("%Y%m%d_%H%M%S")
    save_seg_file = os.path.join('./seg_result', f"{config['dataname']}_{current_time}")

    with torch.no_grad():
        predict_log_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log',
                                        config["dataname"] + '_predict_' + config['net'] + '_' + str(config['labeled_rate']) + '.txt')

        log_args(predict_log_file)
        print(len(test_loader))
        test_metric = test_all_case(model, test_loader, True, save_seg_file=save_seg_file)
        logging.info(
            "fold:{0}, WTDice:{1:.4f}, TCDice:{2:.4f}, ETDice:{3:.4f}, WThd:{4:.4f}, TChd:{5:.4f}, EThd:{6:.4f}, WTJI:{7:.4f}, TCJI:{8:.4f}, ETJI:{9:.4f}, WTASD:{10:.4f}, TCASD:{11:.4f}, ETASD:{12:.4f}".format(
                fold, test_metric[0], test_metric[1], test_metric[2],
                test_metric[3],
                test_metric[4],
                test_metric[5],
                test_metric[6],
                test_metric[7],
                test_metric[8],
                test_metric[9],
                test_metric[10],
                test_metric[11]
            ))
        print(test_metric)

    end_time = time.time()
    full_test_time = (end_time - start_time) / 60
    print('test full time: {:.2f} minutes!'.format(full_test_time))


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()
