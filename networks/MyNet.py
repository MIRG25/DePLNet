import os
import sys

from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from config import config
from networks.MyNet_MoE.DSPLabel import DSPModel
from monai.losses import DiceCELoss
from networks.MyNet_MoE import ramps, losses
from networks.MyNet_MoE.model_factory import create_model, get_model_info
from utils import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../data/BraTS_2021', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='MyNet', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--pseu', type=float,  default=0.1, help='pseu')
parser.add_argument('--pseu_rampup', type=float,  default=40.0, help='pseu_rampup')
parser.add_argument('--dropout_rate', type=float,  default=0.5, help='dropout rate')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_current_pseu_weight(epoch):
    return args.pseu * ramps.sigmoid_rampup(epoch, args.pseu_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data * (1 - alpha))


def train_MyNet_MoE(train_loader, test_loader, fold):
    ## make logger file
    snapshot_path_log = os.path.join('./log', config["dataname"] + '_' + str(config["labeled_rate"]), config["net"] + '_' + str(fold))
    if not os.path.exists(snapshot_path_log):
        os.makedirs(snapshot_path_log)
    snapshot_path_writer = os.path.join('./writer', config["dataname"] + '_' + str(config["labeled_rate"]), config["net"] + '_' + str(fold))
    if not os.path.exists(snapshot_path_writer):
        os.makedirs(snapshot_path_writer)
    snapshot_path_model = os.path.join('./checkpoint', config["dataname"] + '_' + str(config["labeled_rate"]), config["net"] + '_' + str(fold))
    if not os.path.exists(snapshot_path_model):
        os.makedirs(snapshot_path_model)

    date = time.strftime("%Y_%m_%d", time.localtime())
    log_file = os.path.join(snapshot_path_log, date + ".txt")
    if not os.path.exists(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w'):
            pass
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # 打印模型配置信息
    model_info = get_model_info()
    logging.info(f"Model Configuration: {model_info}")
    logging.info(f"Use MoE: {model_info['use_moe']}")
    logging.info(f"Use VC Gating: {model_info['use_vc_gating']}")
    logging.info(f"Use DSP: {model_info['use_dsp']}")

    # 使用model_factory创建模型
    model = create_model(ema=False, has_dropout=False)
    ema_model = create_model(ema=True, has_dropout=False)

    # 根据配置决定是否使用DSP
    use_dsp = config["use_dsp"]
    if use_dsp:
        dsp_module = DSPModel(num_classes=config["num_class"])
        logging.info("DSP module enabled")
    else:
        dsp_module = None
        logging.info("DSP module disabled")

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    consistency_criterion_mse = losses.softmax_mse_loss
    consistency_criterion_kl = losses.softmax_kl_loss

    # Standard Dice+CE loss (no class weights, no focal weighting)
    dice_loss_func = DiceCELoss(to_onehot_y=False, sigmoid=True)

    writer = SummaryWriter(snapshot_path_writer)

    CONS_START = 1000
    DSP_START = 3000
    BOUNDARY_DSP_STRONG = 5000
    iter_num = 0
    max_epoch = max_iterations//len(train_loader)+1
    lr_ = base_lr
    model.train()
    best_performance = 0.0

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        epoch_time1 = time.time()

        for i_batch, sampled_batch in enumerate(train_loader):
            iter_time1 = time.time()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            # Forward pass
            outputs, features, moe_aux = model(volume_batch)
            # print("+++++++++++++++++++++++++++++++++++++++++++++")
            with torch.no_grad():
                ema_output, ema_features, _ = ema_model(ema_inputs)

            # DSP模块更新（仅当use_dsp=True时）
            if use_dsp and iter_num >= DSP_START:
                with torch.no_grad():
                    ema_output_label, ema_features_label, _ = ema_model(volume_batch[:labeled_bs])
                dsp_module.update_memory_from_labeled_data(ema_output_label, ema_features_label, label_batch[:labeled_bs])

            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, config["num_class"], 128, 128, 128]).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)], _, _ = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, config["num_class"], 128, 128, 128)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)

            ## calculate the loss
            # Use standard Dice+CE loss (no class weights, no focal weighting)
            loss_seg = dice_loss_func(outputs[:labeled_bs], label_batch[:labeled_bs])
            supervised_loss = loss_seg
            print("loss_seg", loss_seg)


            # 一致性损失
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = consistency_criterion_mse(outputs[labeled_bs:], ema_output)
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            mask = (uncertainty<threshold).float()
            consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
            consistency_loss = consistency_weight * consistency_dist

            balance_loss = moe_aux["balance_loss"]
            diversity_loss = moe_aux["diversity_loss"]
            aux_loss = moe_aux["aux_loss"]

            lambda_balance = 0.5
            lambda_diversity = 0.1
            lambda_aux = 0.1
            # 如果 ET / TC还是涨不明显，再试：
            # lambda_balance = 0.02
            # lambda_diversity = 0.02
            # DSP_START = 4000
            # proto_conf threshold = 0.45
            # teacher_conf threshold = 0.75

            moe_reg_loss = lambda_balance * balance_loss + lambda_diversity * diversity_loss + lambda_aux * aux_loss

            # 总损失
            loss = supervised_loss + consistency_loss + moe_reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss.item(), iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg.item(), iter_num)
            writer.add_scalar('loss/consistency_loss', consistency_loss.item(), iter_num)
            writer.add_scalar('loss/moe_balance', balance_loss.item(), iter_num)
            writer.add_scalar('loss/moe_diversity', diversity_loss.item(), iter_num)
            writer.add_scalar('loss/moe_reg', moe_reg_loss.item(), iter_num)

            # 测试和保存
            if iter_num >= 5000 and iter_num % 5000 == 0:
                model.eval()
                test_metric = test_all_case(model, test_loader, save_seg=False, fold=fold)
                logging.info(
                    "fold:{0}, iter:{1}, WTDice:{2:.4f}, TCDice:{3:.4f}, ETDice:{4:.4f}, WThd:{5:.4f}, TChd:{6:.4f}, EThd:{7:.4f}, WTJI:{8:.4f}, TCJI:{9:.4f}, ETJI:{10:.4f}, WTASD:{11:.4f}, TCASD:{12:.4f}, ETASD:{13:.4f}".format(
                        fold, iter_num, test_metric[0], test_metric[1], test_metric[2],
                        test_metric[3], test_metric[4], test_metric[5],
                        test_metric[6], test_metric[7], test_metric[8],
                        test_metric[9], test_metric[10], test_metric[11]
                    ))
                logging.info(
                    "iter:{0}, loss:{1:.4f}, loss_seg:{2:.4f}, consistency:{3:.4f}".format(
                        iter_num, loss.item(), loss_seg.item(), consistency_loss.item()
                    ))
                dice = (test_metric[0] + test_metric[1] + test_metric[2]) / config["num_class"]
                if dice > best_performance:
                    best_performance = dice
                    save_mode_path = os.path.join(snapshot_path_model,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    torch.save(model.state_dict(), save_mode_path)
                model.train()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break

            with torch.no_grad():
                labeled_onehot = label_batch[:labeled_bs].float()  # [B_l, C, D, H, W]
                gt_ratio = labeled_onehot.mean(dim=[0, 2, 3, 4])  # [C]

                # 如果模型有 MoE gate，就把统计喂进去
                if hasattr(model, 'moe3') and hasattr(model.moe3.gate, 'update_volume_ratios_from_stats'):
                    model.moe3.gate.update_volume_ratios_from_stats(gt_ratio)
                if hasattr(model, 'moe5') and hasattr(model.moe5.gate, 'update_volume_ratios_from_stats'):
                    model.moe5.gate.update_volume_ratios_from_stats(gt_ratio)

        epoch_time2 = time.time()
        epoch_time_minute = (epoch_time2 - epoch_time1) / 60
        if iter_num >= max_iterations:
            break

    save_mode_path = os.path.join(snapshot_path_model, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    writer.close()

    return best_performance