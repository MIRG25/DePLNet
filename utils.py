import logging
import pickle

import imageio
import tensorboardX
import os
from collections import OrderedDict
import SimpleITK as sitk
import torch
import nibabel as nib
from medpy import metric
from tqdm import tqdm
import sys
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import compute_iou, compute_average_surface_distance
from config import config
from monai.transforms import Activations, AsDiscrete

post_activate = Activations(sigmoid=True)

sys.path.append(".")

from torch.utils.data import Dataset

import numpy as np

np.random.seed(0)

import random

random.seed(0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter("./runs/" + model_name.split("/")[-1].split(".h5")[0])

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']

        for col in self.header[1:]:
            self.writer.add_scalar(phase + "/" + col, float(values[col]), int(epoch))


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def combine_labels(labels):
    """
    Combine wt, tc, et into WT; tc, et into TC; et into ET
    :param labels: torch.Tensor of size (bs, 3, ?,?,?); ? is the crop size
    :return:
    """
    # whole_tumor = labels[:, :3, :, :, :].sum(1)  # could have 2 or 3
    # tumor_core = labels[:, 1:3, :, :, :].sum(1)
    # enhanced_tumor = labels[:, 2:3, :, :, :].sum(1)
    # whole_tumor[whole_tumor != 0] = 1
    # tumor_core[tumor_core != 0] = 1
    # enhanced_tumor[enhanced_tumor != 0] = 1
    # return whole_tumor, tumor_core, enhanced_tumor  # (bs, ?, ?, ?)
    whole_tumor = (labels[:, 0] | labels[:, 1] | labels[:, 2])  # WT = NCR + ED + ET
    tumor_core = (labels[:, 0] | labels[:, 2])  # TC = NCR + ET
    enhanced_tumor = labels[:, 2]  # ET
    return whole_tumor.float(), tumor_core.float(), enhanced_tumor.float()


def calculate_accuracy(outputs, targets):
    return dice_coefficient(outputs, targets)


def hausdorff(prediction: np.ndarray, reference: np.ndarray) -> float:
    try:
        return metric.hd95(prediction, reference)

    except Exception as e:
        print("Hausdorff Error: ", e)
        print(f"Hausdorff prediction does not contain the same label as gt. "
              f"Hausdorff pred labels {np.unique(prediction)} GT labels {np.unique(reference)}")
        return 100


def jacc(prediction: np.ndarray, reference: np.ndarray) -> float:
    try:
        return metric.binary.jc(prediction, reference)

    except Exception as e:
        print("Jaccard Index Error: ", e)
        print(f"Jaccard Index prediction does not contain the same label as gt. "
              f"Jaccard Index pred labels {np.unique(prediction)} GT labels {np.unique(reference)}")
        return 100


def asd(prediction: np.ndarray, reference: np.ndarray) -> float:
    try:
        return metric.asd(prediction, reference)

    except Exception as e:
        print("ASD Error: ", e)
        print(f"ASD prediction does not contain the same label as gt. "
              f"ASD pred labels {np.unique(prediction)} GT labels {np.unique(reference)}")
        return 100


def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8):
    # batch_size = targets.size(0)
    haussdor = HausdorffDistanceMetric(include_background=False, percentile=95)
    probs = post_activate(outputs)
    y_pred = (probs >= threshold)

    # y_pred = y_pred.type(torch.FloatTensor)
    wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    wt_truth, tc_truth, et_truth = combine_labels(targets)

    # tc_pred, wt_pred, et_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    # tc_truth, wt_truth, et_truth = targets[:, 0], targets[:, 1], targets[:, 2]

    # wt_pred_np = wt_pred.detach().cpu().numpy()
    # wt_truth_np = wt_truth.detach().cpu().numpy()
    # tc_pred_np = tc_pred.detach().cpu().numpy()
    # tc_truth_np = tc_truth.detach().cpu().numpy()
    # et_pred_np = et_pred.detach().cpu().numpy()
    # et_truth_np = et_truth.detach().cpu().numpy()
    res = dict()
    if not torch.any(wt_pred) and not torch.any(wt_truth):
        res["dice_wt"] = 1.0
        res["JI_wt"] = 1.0
        res["ASD_wt"] = 0.0
        res["wt_hd"] = 0.0
    elif not torch.any(wt_pred) or not torch.any(wt_truth):
        # 其中一个为空，距离指标设为无穷大，Dice设为0
        res["dice_wt"] = 0.0
        res["JI_wt"] = 0.0
        res["ASD_wt"] = 100.0
        res["wt_hd"] = 100.0
    else:
        res["dice_wt"] = metric.binary.dc(wt_pred, wt_truth)
        res["JI_wt"] = metric.binary.jc(wt_pred, wt_truth)
        res["ASD_wt"] = metric.binary.asd(wt_pred, wt_truth)
        res["wt_hd"] = metric.binary.hd95(wt_pred, wt_truth)

        # res["dice_wt"] = dice_coefficient_single_label(wt_pred, wt_truth, eps).item()
        # res["JI_wt"] = compute_iou(wt_pred.unsqueeze(0), wt_truth.unsqueeze(0), include_background=False).item()
        # res["ASD_wt"] = compute_average_surface_distance(wt_pred.unsqueeze(0), wt_truth.unsqueeze(0)).item()
        # res["wt_hd"] = haussdor(wt_pred.unsqueeze(0), wt_truth.unsqueeze(0)).item()

    if not torch.any(tc_pred) and not torch.any(tc_truth):
        res["dice_tc"] = 1.0
        res["JI_tc"] = 1.0
        res["ASD_tc"] = 0.0
        res["tc_hd"] = 0.0
    elif not torch.any(tc_pred) or not torch.any(tc_truth):
        # 其中一个为空，距离指标设为无穷大，Dice设为0
        res["dice_tc"] = 0.0
        res["JI_tc"] = 0.0
        res["ASD_tc"] = 100.0
        res["tc_hd"] = 100.0
    else:
        res["dice_tc"] = metric.binary.dc(tc_pred, tc_truth)
        res["JI_tc"] = metric.binary.jc(tc_pred, tc_truth)
        res["ASD_tc"] = metric.binary.asd(tc_pred, tc_truth)
        res["tc_hd"] = metric.binary.hd95(tc_pred, tc_truth)
        # res["dice_tc"] = dice_coefficient_single_label(tc_pred, tc_truth, eps).item()
        # res["JI_tc"] = compute_iou(tc_pred.unsqueeze(0), tc_truth.unsqueeze(0), include_background=False).item()
        # res["ASD_tc"] = compute_average_surface_distance(tc_pred.unsqueeze(0), tc_truth.unsqueeze(0)).item()
        # res["tc_hd"] = haussdor(tc_pred.unsqueeze(0), tc_truth.unsqueeze(0)).item()

    if not torch.any(et_pred) and not torch.any(et_truth):
        res["dice_et"] = 1.0
        res["JI_et"] = 1.0
        res["ASD_et"] = 0.0
        res["et_hd"] = 0.0
    elif not torch.any(et_pred) or not torch.any(et_truth):
        # 其中一个为空，距离指标设为无穷大，Dice设为0
        res["dice_et"] = 0.0
        res["JI_et"] = 0.0
        res["ASD_et"] = 100.0
        res["et_hd"] = 100.0
    else:
        res["dice_et"] = metric.binary.dc(et_pred, et_truth)
        res["JI_et"] = metric.binary.jc(et_pred, et_truth)
        res["ASD_et"] = metric.binary.asd(et_pred, et_truth)
        res["et_hd"] = metric.binary.hd95(et_pred, et_truth)
        # res["dice_et"] = dice_coefficient_single_label(et_pred, et_truth, eps).item()
        # res["JI_et"] = compute_iou(et_pred.unsqueeze(0), et_truth.unsqueeze(0), include_background=False).item()
        # res["ASD_et"] = compute_average_surface_distance(et_pred.unsqueeze(0), et_truth.unsqueeze(0)).item()
        # res["et_hd"] = haussdor(et_pred.unsqueeze(0), et_truth.unsqueeze(0)).item()

    # res["dice_wt"] = dice_coefficient_single_label(wt_pred, wt_truth, eps)
    # res["dice_tc"] = dice_coefficient_single_label(tc_pred, tc_truth, eps)
    # res["dice_et"] = dice_coefficient_single_label(et_pred, et_truth, eps)

    # if torch.sum(wt_pred) != 0.0 and torch.sum(wt_truth) != 0.0:
    #     res["wt_hd"] = haussdor(wt_pred.unsqueeze(0).cpu(), wt_truth.unsqueeze(0).cpu()).item()
    # elif torch.sum(wt_pred) == 0.0 and torch.sum(wt_truth) == 0.0:
    #     res["wt_hd"] = 0.0
    # elif (torch.sum(wt_pred) == 0.0 and torch.sum(wt_truth) != 0.0) or (torch.sum(wt_pred) != 0.0 and torch.sum(wt_truth) == 0.0):
    #     res["wt_hd"] = 347
    #
    # if torch.sum(tc_pred) != 0.0 and torch.sum(tc_truth) != 0.0:
    #     res["tc_hd"] = haussdor(tc_pred.unsqueeze(0).cpu(), tc_truth.unsqueeze(0).cpu()).item()
    # elif torch.sum(tc_pred) == 0.0 and torch.sum(tc_truth) == 0.0:
    #     res["tc_hd"] = 0.0
    # elif (torch.sum(tc_pred) == 0.0 and torch.sum(tc_truth) != 0.0) or (torch.sum(tc_pred) != 0.0 and torch.sum(tc_truth) == 0.0):
    #     res["tc_hd"] = 347
    #
    # if torch.sum(et_pred) != 0.0 and torch.sum(et_truth) != 0.0:
    #     res["et_hd"] = haussdor(et_pred.unsqueeze(0).cpu(), et_truth.unsqueeze(0).cpu()).item()
    # elif torch.sum(et_pred) == 0.0 and torch.sum(et_truth) == 0.0:
    #     res["et_hd"] = 0.0
    # elif (torch.sum(et_pred) == 0.0 and torch.sum(et_truth) != 0.0) or (torch.sum(et_pred) != 0.0 and torch.sum(et_truth) == 0.0):
    #     res["et_hd"] = 347
    #
    print(res)
    return res


def calculate_accuracy_singleLabel(outputs, targets, threshold=0.5, eps=1e-8):
    y_pred = outputs[:, 0, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, 0, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    res = dice_coefficient_single_label(y_pred, y_truth, eps)
    return res


def dice_coefficient_single_label(y_pred, y_truth, eps):
    # batch_size = y_pred.size(0)
    intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(-3, -2, -1)) + eps / 2  # axis=?, (bs, 1)
    union = torch.sum(y_pred, dim=(-3, -2, -1)) + torch.sum(y_truth, dim=(-3, -2, -1)) + eps  # (bs, 1)
    dice = 2 * intersection / union
    return dice.mean()
    # return dice / batch_size


def load_old_model(model, optimizer, saved_model_path, data_paralell=True):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path, map_location='cpu')
    epoch = checkpoint["epoch"]
    # epoch = 1
    if data_paralell:
        state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():  # remove "module."
            if "module." in k:
                node_name = k[7:]

            else:
                node_name = k
            state_dict[node_name] = v
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, epoch, optimizer


def combine_labels_predicting(output_array):
    """
    # (1, 3, 240, 240, 155)
    :param output_array: output of the model containing 3 seperated labels (3 channels)
    :return: res_array: conbined labels (1 channel)
    """
    shape = output_array.shape[-3:]

    if len(output_array.shape) == 5:
        bs = output_array.shape[0]
        res_array = np.zeros((bs,) + shape)
        res_array[output_array[:, 0, :, :, :] == 1] = 2  # 1
        res_array[output_array[:, 1, :, :, :] == 1] = 1  # 2
        res_array[output_array[:, 2, :, :, :] == 1] = 4
    elif len(output_array.shape) == 4:
        res_array = np.zeros(shape)
        res_array[output_array[0, :, :, :] == 1] = 2
        res_array[output_array[1, :, :, :] == 1] = 1
        res_array[output_array[2, :, :, :] == 1] = 4
    return res_array


def dim_recovery(img_array, orig_shape=(155, 240, 240)):
    """
    used when doing inference
    :param img_array:
    :param orig_shape:
    :return:
    """
    crop_shape = np.array(img_array.shape[-3:])
    center = np.array(orig_shape) // 2
    lower_limits = center - crop_shape // 2
    upper_limits = center + crop_shape // 2
    if len(img_array.shape) == 5:
        bs, num_labels = img_array.shape[:2]
        res_array = np.zeros((bs, num_labels) + orig_shape)
        res_array[:, :, lower_limits[0]: upper_limits[0],
        lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array
    if len(img_array.shape) == 4:
        num_labels = img_array.shape[0]
        res_array = np.zeros((num_labels,) + orig_shape)
        res_array[:, lower_limits[0]: upper_limits[0],
        lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array

    if len(img_array.shape) == 3:
        res_array = np.zeros(orig_shape)
        res_array[lower_limits[0]: upper_limits[0],
        lower_limits[1]: upper_limits[1], lower_limits[2]: upper_limits[2]] = img_array

    return res_array


def convert_stik_to_nparray(gz_path):
    sitkImage = sitk.ReadImage(gz_path)
    nparray = sitk.GetArrayFromImage(sitkImage)
    return nparray


def poly_lr_scheduler(epoch, num_epochs=300, power=0.9):
    return (1 - epoch / num_epochs) ** power


def sav_seg_res(
        save_path,
        file_name,
        images,
        pred,
        snapshot=False,
):
    """
    保存分割结果为nii.gz格式

    Args:
        save_path: 保存路径
        file_name: 文件名
        images: 输入图像
        pred: 预测结果 (logits)
        snapshot: 是否保存快照
    """
    _, _, H, W, T = images.shape
    seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

    # 使用sigmoid与训练时的监督损失一致
    pred = torch.sigmoid(pred)
    pred_img = pred[0].cpu().numpy()
    pred_img = pred_img > 0.5

    # 转换为BraTS标准格式: NCR=1, ED=2, ET=4
    seg_img[pred_img[0, :, :, :] == 1] = 1  # NCR
    seg_img[pred_img[1, :, :, :] == 1] = 2  # ED
    seg_img[pred_img[2, :, :, :] == 1] = 4  # ET

    # 使用第一个样本的t1ce作为参考（获取affine和header）
    t1ce_Path = os.path.join(config["root"], 'BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz')
    if not os.path.exists(t1ce_Path):
        # 如果默认路径不存在，尝试使用BraTS20格式
        t1ce_Path = os.path.join(config["root"], 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz')

    if os.path.exists(t1ce_Path):
        t1ce_image = nib.load(t1ce_Path)
        affine = t1ce_image.affine
        header = t1ce_image.header
    else:
        # 如果都不存在，使用默认affine
        affine = np.eye(4)
        header = None

    oname = os.path.join(save_path, str(file_name) + '.nii.gz')
    nib.save(nib.Nifti1Image(seg_img, affine=affine, header=header), oname)

    if snapshot:
        """ --- colorful figure--- """
        Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
        Snapshot_img[:, :, 0, :][np.where(pred_img[0] == 1)] = 255  # NCR - Red
        Snapshot_img[:, :, 1, :][np.where(pred_img[1] == 1)] = 255  # ED - Green
        Snapshot_img[:, :, 2, :][np.where(pred_img[2] == 1)] = 255  # ET - Blue
        for frame in range(T):
            if not os.path.exists(os.path.join(save_path, file_name)):
                os.makedirs(os.path.join(save_path, file_name))
            imageio.imwrite(os.path.join(save_path, file_name, str(frame) + '.png'), Snapshot_img[:, :, :, frame])


def test_all_case(model, test_loader, save_seg=False, fold=None):
    """
    测试所有样本并计算指标

    Args:
        model: 模型
        test_loader: 测试数据加载器
        save_seg: 是否保存分割结果
        fold: 当前fold编号（用于保存summary）

    Returns:
        tuple: (WT_dice, TC_dice, ET_dice, WT_hd, TC_hd, ET_hd, WT_JI, TC_JI, ET_JI, WT_ASD, TC_ASD, ET_ASD)
    """
    WT_dice = AverageMeter()
    TC_dice = AverageMeter()
    ET_dice = AverageMeter()
    WT_JI = AverageMeter()
    TC_JI = AverageMeter()
    ET_JI = AverageMeter()
    WT_ASD = AverageMeter()
    TC_ASD = AverageMeter()
    ET_ASD = AverageMeter()
    WT_hd = AverageMeter()
    TC_hd = AverageMeter()
    ET_hd = AverageMeter()

    test_process = tqdm(test_loader)

    for i, batch_data in enumerate(test_process):
        if config["cuda_devices"] is not None:
            inputs, targets = batch_data["image"].cuda(), batch_data["label"].cuda()

        _, _, Z, H, W = inputs.size()
        patch_size_z = config["image_shape"][0]
        patch_size_h = config["image_shape"][1]
        patch_size_w = config["image_shape"][2]
        #########get z_ind, h_ind, w_ind for sliding windows
        z_cnt = int(np.ceil((Z - patch_size_z) / (patch_size_z * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * int(patch_size_z * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size_z)

        h_cnt = int(np.ceil((H - patch_size_h) / (patch_size_h * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * int(patch_size_h * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size_h)

        w_cnt = int(np.ceil((W - patch_size_w) / (patch_size_w * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * int(patch_size_w * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size_w)
        # print(z_idx_list, h_idx_list, w_idx_list)

        pred_seg = torch.zeros(inputs.shape[0], 3, Z, H, W).float().cuda()
        one_tensor = torch.ones(inputs.shape[0], 1, patch_size_z, patch_size_h, patch_size_w).float().cuda()
        weight = torch.zeros(inputs.shape[0], 1, Z, H, W).float().cuda()

        with torch.no_grad():
            for z in z_idx_list:
                for h in h_idx_list:
                    for w in w_idx_list:
                        weight[:, :, z:z + patch_size_z, h:h + patch_size_h, w:w + patch_size_w] += one_tensor
                        x_input = inputs[:, :, z:z + patch_size_z, h:h + patch_size_h, w:w + patch_size_w]
                        if config["net"] == 'VNet':
                            outputs = model(x_input)
                        else:
                            outputs, _, _ = model(x_input)
                        seg_pred = outputs[:, :3, :, :, :]
                        pred_seg[:, :, z:z + patch_size_z, h:h + patch_size_h, w:w + patch_size_w] += seg_pred

        # 保存分割结果
        weight1 = weight.repeat(1, 3, 1, 1, 1)
        # print(pred_seg.shape)
        # print(weight1.shape)
        pred_seg = pred_seg / weight1
        pred_seg = pred_seg[:, :, :Z, :H, :W]
        for j in range(pred_seg.shape[0]):  # 遍历每个样本
            acc = calculate_accuracy(pred_seg[j:j + 1, :].cpu(), targets[j:j + 1, :].cpu())

            WT_dice.update(acc["dice_wt"], inputs.size(0))
            TC_dice.update(acc["dice_tc"], inputs.size(0))
            ET_dice.update(acc["dice_et"], inputs.size(0))
            WT_JI.update(acc["JI_wt"], inputs.size(0))
            TC_JI.update(acc["JI_tc"], inputs.size(0))
            ET_JI.update(acc["JI_et"], inputs.size(0))
            WT_ASD.update(acc["ASD_wt"], inputs.size(0))
            TC_ASD.update(acc["ASD_tc"], inputs.size(0))
            ET_ASD.update(acc["ASD_et"], inputs.size(0))
            WT_hd.update(acc["wt_hd"], inputs.size(0))
            TC_hd.update(acc["tc_hd"], inputs.size(0))
            ET_hd.update(acc["et_hd"], inputs.size(0))
        # print(acc["dice_wt"], acc["dice_tc"], acc["dice_et"], acc["wt_hd"], acc["tc_hd"], acc["et_hd"])
        # 保存分割结果到本地
        if save_seg:
            save_seg_file = os.path.join('./seg_result', config["dataname"] + '_' + str(config["labeled_rate"]),
                                         config["net"])
            if not os.path.exists(save_seg_file):
                os.makedirs(save_seg_file)
            sav_seg_res(save_seg_file, i, inputs, pred_seg)
        # logging.info(
        #     "item: {0:d}, WTDice:{1:.4f}, TCDice:{2:.4f}, ETDice:{3:.4f}, WThd:{4:.4f}, TChd:{5:.4f}, EThd:{6:.4f}".format(
        #         i, acc['dice_wt'].item(), acc['dice_tc'].item(), acc['dice_et'].item(), acc['wt_hd'], acc['tc_hd'], acc['et_hd']))

    # 保存测试结果到summary.txt
    if fold is not None:
        summary_path = os.path.join('./log', config["dataname"] + '_' + str(config["labeled_rate"]),
                                   config["net"] + '_' + str(fold), 'summary.txt')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        with open(summary_path, 'a') as f:
            import time
            f.write(f"\n{'='*80}\n")
            f.write(f"Test Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Fold: {fold}\n")
            f.write(f"Model: {config['net']}\n")
            f.write(f"Labeled Rate: {config['labeled_rate']}%\n")
            f.write(f"\nDice Scores:\n")
            f.write(f"  WT: {WT_dice.avg:.4f}\n")
            f.write(f"  TC: {TC_dice.avg:.4f}\n")
            f.write(f"  ET: {ET_dice.avg:.4f}\n")
            f.write(f"  Avg: {(WT_dice.avg + TC_dice.avg + ET_dice.avg) / 3:.4f}\n")
            f.write(f"\nHausdorff Distance (95%):\n")
            f.write(f"  WT: {WT_hd.avg:.4f}\n")
            f.write(f"  TC: {TC_hd.avg:.4f}\n")
            f.write(f"  ET: {ET_hd.avg:.4f}\n")
            f.write(f"\nJaccard Index:\n")
            f.write(f"  WT: {WT_JI.avg:.4f}\n")
            f.write(f"  TC: {TC_JI.avg:.4f}\n")
            f.write(f"  ET: {ET_JI.avg:.4f}\n")
            f.write(f"\nAverage Surface Distance:\n")
            f.write(f"  WT: {WT_ASD.avg:.4f}\n")
            f.write(f"  TC: {TC_ASD.avg:.4f}\n")
            f.write(f"  ET: {ET_ASD.avg:.4f}\n")
            f.write(f"{'='*80}\n")

        logging.info(f"Test results saved to {summary_path}")

    return WT_dice.avg, TC_dice.avg, ET_dice.avg, WT_hd.avg, TC_hd.avg, ET_hd.avg, WT_JI.avg, TC_JI.avg, ET_JI.avg, WT_ASD.avg, TC_ASD.avg, ET_ASD.avg