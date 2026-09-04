import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss2(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def softmax_kl_loss(input_logits, target_logits, temperature=1.0):
    """数值稳定的softmax KL散度计算"""
    assert input_logits.size() == target_logits.size()

    # 添加数值稳定性检查
    if torch.isnan(input_logits).any() or torch.isinf(input_logits).any():
        print("WARNING: input_logits contains NaN or Inf")
        input_logits = torch.nan_to_num(input_logits, nan=0.0, posinf=1e6, neginf=-1e6)

    if torch.isnan(target_logits).any() or torch.isinf(target_logits).any():
        print("WARNING: target_logits contains NaN or Inf")
        target_logits = torch.nan_to_num(target_logits, nan=0.0, posinf=1e6, neginf=-1e6)

    # 应用温度缩放
    input_logits = input_logits / temperature
    target_logits = target_logits / temperature

    # 使用数值稳定的log_softmax
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # 添加clip避免数值问题
    target_softmax = torch.clamp(target_softmax, min=1e-8, max=1.0)

    # 计算KL散度: KL(target_softmax || input_log_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean', log_target=False)
    print("kl_div", kl_div)

    # 检查结果
    if torch.isnan(kl_div) or torch.isinf(kl_div):
        print(f"KL divergence is NaN/Inf: {kl_div}")
        return torch.tensor(0.0, device=input_logits.device)

    return kl_div


def softmax_kl_loss3(input_logits, target_probs, temperature=3.0):
    """
    修复版本：使用温度缩放和对称KL
    """
    assert input_logits.size() == target_probs.size()

    # 应用温度缩放到输入logits
    input_log_softmax = F.log_softmax(input_logits / temperature, dim=1)
    target_softmax = F.softmax(torch.log(target_probs + 1e-8) / temperature, dim=1)

    # 数值稳定性处理
    target_softmax = torch.clamp(target_softmax, min=1e-8, max=1.0)

    # 对称KL损失 (更稳定)
    kl_forward = F.kl_div(input_log_softmax, target_softmax, reduction='batchmean')
    kl_backward = F.kl_div(torch.log(target_softmax + 1e-8),
                           F.softmax(input_logits / temperature, dim=1),
                           reduction='mean')

    symmetric_kl = (kl_forward + kl_backward) / 2

    return symmetric_kl

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


def weighted_cross_entropy(pred, target, weight):
    """
    pred: [B, C, D, H, W] 网络输出（未softmax）
    target: [B, D, H, W] 或 [B, 1, D, H, W] 标签
    weight: [B, D, H, W] 或 [B, 1, D, H, W] 逐像素权重
    """
    ce_per_pixel = F.cross_entropy(pred, target, reduction='none')
    # 应用权重
    # weighted_loss = ce_per_pixel * weight
    weighted_loss = ce_per_pixel
    # 返回平均损失
    return weighted_loss.mean()
    # 计算标准的交叉熵损失（无reduce）
    # cross_entropy = nn.CrossEntropyLoss(weight)
    # cross_entropy = cross_entropy(pred, target)
    # return cross_entropy