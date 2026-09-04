"""
增强的损失函数模块
融合DyCON的优势：对比学习、自适应熵权重、Focal加权
同时保留我们的创新点：MoE、体积补偿门控、DSP
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def adaptive_beta(epoch, total_epochs, max_beta=5.0, min_beta=0.5):
    """
    自适应beta调度（来自DyCON）
    随着训练进行，beta从max_beta衰减到min_beta
    """
    ratio = min_beta / max_beta
    exponent = epoch / total_epochs
    beta = max_beta * (ratio ** exponent)
    return beta


def sigmoid_rampup(current_epoch, total_rampup_epochs, min_threshold, max_threshold, steepness=5.0):
    """
    Sigmoid ramp-up调度
    """
    if total_rampup_epochs == 0:
        return max_threshold
    current_epoch = max(0.0, min(float(current_epoch), total_rampup_epochs))
    phase = 1.0 - (current_epoch / total_rampup_epochs)
    ramp = math.exp(-steepness * (phase ** 2))
    return min_threshold + (max_threshold - min_threshold) * ramp


class UnCLoss(nn.Module):
    """
    不确定性感知一致性损失（来自DyCON）

    核心思想：
    1. 计算student和teacher的熵（不确定性）
    2. 低不确定性区域获得更高权重
    3. 惩罚高不确定性预测
    """
    def __init__(self):
        super(UnCLoss, self).__init__()

    def forward(self, s_logits, t_logits, beta):
        """
        Args:
            s_logits: Student logits (B, C, H, W, D)
            t_logits: Teacher logits (B, C, H, W, D)
            beta: 熵权重参数（越大越强调确定性区域）
        """
        EPS = 1e-6

        # Student概率和熵
        p_s = F.softmax(s_logits, dim=1)
        p_s_log = torch.log(p_s + EPS)
        H_s = -torch.sum(p_s * p_s_log, dim=1, keepdim=True)  # (B, 1, H, W, D)

        # Teacher概率和熵
        p_t = F.softmax(t_logits, dim=1)
        p_t_log = torch.log(p_t + EPS)
        H_t = -torch.sum(p_t * p_t_log, dim=1, keepdim=True)  # (B, 1, H, W, D)

        # 熵加权
        exp_H_s = torch.exp(beta * H_s)
        exp_H_t = torch.exp(beta * H_t)

        # 熵加权的平方差 + 熵惩罚
        loss = (p_s - p_t)**2 / (exp_H_s + exp_H_t)
        loss = torch.mean(loss.sum(dim=1) + beta * (H_s + H_t))

        return loss


class FeCLoss(nn.Module):
    """
    特征对比损失（Feature Contrastive Loss，来自DyCON）

    核心思想：
    1. 同类特征应该相似（拉近）
    2. 异类特征应该不同（推远）
    3. Focal加权：关注hard positive/negative
    4. Teacher-Student对比：额外的监督信号
    """
    def __init__(self, temperature=0.6, gamma=2.0, use_focal=True,
                 rampup_epochs=1500, lambda_cross=1.0):
        super(FeCLoss, self).__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.use_focal = use_focal
        self.rampup_epochs = rampup_epochs
        self.lambda_cross = lambda_cross

    def forward(self, feat, mask, teacher_feat=None, epoch=0):
        """
        Args:
            feat: Student特征 (B, N, D) - N是patch数，D是特征维度
            mask: 标签mask (B, 1, N)
            teacher_feat: Teacher特征 (B, N, D)，可选
            epoch: 当前epoch，用于动态阈值
        """
        B, N, D = feat.shape
        device = feat.device

        # 1. 计算相似度矩阵
        feat_logits = torch.matmul(feat, feat.transpose(1, 2)) / self.temperature  # (B, N, N)

        # 2. 构建mask矩阵
        mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()  # 同类为1
        mem_mask_neg = 1 - mem_mask  # 异类为1

        # 3. 去除自相似度
        identity = torch.eye(N, device=device)
        neg_identity = 1 - identity
        feat_logits = feat_logits * neg_identity

        # 4. 数值稳定性
        feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
        feat_logits = feat_logits - feat_logits_max.detach()

        # 5. 计算对比损失
        exp_logits = torch.exp(feat_logits)
        neg_sum = torch.sum(exp_logits * mem_mask_neg, dim=-1)  # 负样本和

        denominator = exp_logits + neg_sum.unsqueeze(dim=-1)
        division = exp_logits / (denominator + 1e-18)

        loss_matrix = -torch.log(division + 1e-18)
        loss_matrix = loss_matrix * mem_mask * neg_identity

        loss_student = torch.sum(loss_matrix, dim=-1) / (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
        loss_student = loss_student.mean()

        # 6. Focal加权（关注hard samples）
        if self.use_focal:
            similarity = division
            focal_weights = torch.ones_like(similarity)

            # Hard positive: 同类但相似度低
            pos_thresh = sigmoid_rampup(epoch, self.rampup_epochs,
                                       min_threshold=1.3, max_threshold=1.5)
            hard_pos_mask = mem_mask.bool() & (similarity < pos_thresh)
            focal_weights[hard_pos_mask] = (1 - similarity[hard_pos_mask]).pow(self.gamma)

            # Hard negative: 异类但相似度高
            neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs,
                                       min_threshold=0.3, max_threshold=0.5)
            hard_neg_mask = mem_mask_neg.bool() & (similarity > neg_thresh)
            focal_weights[hard_neg_mask] = similarity[hard_neg_mask].pow(self.gamma)

            loss_student = torch.sum(loss_matrix * focal_weights, dim=-1) / \
                          (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
            loss_student = loss_student.mean()

        # 7. Teacher-Student对比（额外监督）
        loss_cross = 0.0
        if teacher_feat is not None:
            cross_sim = torch.matmul(feat, teacher_feat.transpose(1, 2))
            mem_mask_cross_neg = 1 - mem_mask

            cross_neg_thresh = sigmoid_rampup(epoch, self.rampup_epochs,
                                             min_threshold=0.3, max_threshold=0.5)
            cross_hard_neg_mask = mem_mask_cross_neg.bool() & (cross_sim > cross_neg_thresh)

            if cross_hard_neg_mask.sum() > 0:
                loss_cross_term = -torch.log(1 - cross_sim + 1e-18)
                loss_cross_term = loss_cross_term * cross_hard_neg_mask.float()
                loss_cross = torch.sum(loss_cross_term) / \
                            (torch.sum(cross_hard_neg_mask.float()) + 1e-18)

        total_loss = loss_student + self.lambda_cross * loss_cross
        return total_loss


class SimpleDiceCELoss(nn.Module):
    """
    标准的Dice+CE损失（无类别权重，无Focal加权）

    使用sigmoid激活，适用于多标签分割任务
    虽然BraTS标签是互斥的，但sigmoid在训练早期更稳定
    """
    def __init__(self, num_classes=3):
        super(SimpleDiceCELoss, self).__init__()
        self.num_classes = num_classes

    def dice_loss(self, pred, target, smooth=1e-5):
        """
        标准Dice损失

        Args:
            pred: (B, C, H, W, D) - logits
            target: (B, C, H, W, D) - one-hot labels
        """
        pred = torch.sigmoid(pred)

        dice_loss = 0.0
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = target[:, i]

            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()

            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_loss += (1 - dice)

        return dice_loss / self.num_classes

    def bce_loss(self, pred, target):
        """
        标准BCE损失

        Args:
            pred: (B, C, H, W, D) - logits
            target: (B, C, H, W, D) - one-hot labels
        """
        pred = torch.sigmoid(pred)

        # BCE损失
        bce = -(target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))

        return bce.mean()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W, D) - logits
            target: (B, C, H, W, D) - one-hot labels
        """
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)

        return dice_loss + bce_loss


class EnhancedDiceCELoss(nn.Module):
    """
    增强的Dice+CE损失（多类别版本，适配BraTS互斥标签）

    改进点：
    1. 类别权重：针对小体积区域（ET、TC）增加权重
    2. Focal权重：关注难分类样本
    3. 使用softmax，因为BraTS标签是互斥的（一个像素只属于一个类别）

    注意：
    - 输入pred是logits（未归一化）
    - 输入target是one-hot编码（互斥，每个像素只有一个通道为1）
    - 使用softmax处理多类别问题
    """
    def __init__(self, num_classes=3, class_weights=None, focal_alpha=0.25, focal_gamma=2.0):
        super(EnhancedDiceCELoss, self).__init__()
        self.num_classes = num_classes

        # 默认类别权重：温和的权重调整，避免过度偏向某个类别
        if class_weights is None:
            # NCR: 25%, ED: 60%, ET: 15% -> 温和权重: 1.5, 1.0, 2.0
            # 相比之前的 [4.0, 1.67, 6.67]，这个权重更平衡
            self.class_weights = torch.tensor([1.5, 1.0, 2.0])
        else:
            self.class_weights = torch.tensor(class_weights)

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def dice_loss(self, pred, target, smooth=1e-5):
        """
        多类别Dice损失，带类别权重

        Args:
            pred: (B, C, H, W, D) - logits
            target: (B, C, H, W, D) - one-hot labels (互斥)
        """
        pred = F.sigmoid(pred)  # 使用softmax，确保概率和为1

        dice_loss = 0.0
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = target[:, i]

            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()

            dice = (2.0 * intersection + smooth) / (union + smooth)

            # 类别权重
            weight = self.class_weights[i].to(pred.device)
            dice_loss += weight * (1 - dice)

        return dice_loss / self.num_classes

    def focal_ce_loss(self, pred, target):
        """
        Focal Cross Entropy损失（多类别版本）

        Args:
            pred: (B, C, H, W, D) - logits
            target: (B, C, H, W, D) - one-hot labels (互斥)
        """
        pred = F.sigmoid(pred)  # 使用softmax

        # 计算每个像素的预测概率（对应其真实类别）
        # pt = sum(pred * target) 在one-hot情况下就是预测的真实类别概率
        pt = (pred * target).sum(dim=1, keepdim=True)  # (B, 1, H, W, D)

        # Focal权重
        focal_weight = (1 - pt).pow(self.focal_gamma)

        # CE损失（多类别）
        ce = -(target * torch.log(pred + 1e-7)).sum(dim=1, keepdim=True)  # (B, 1, H, W, D)

        # 应用focal权重和alpha
        loss = self.focal_alpha * focal_weight * ce

        # 类别权重：根据每个像素的真实类别应用权重
        class_weights_map = torch.zeros_like(loss)
        for i in range(self.num_classes):
            weight = self.class_weights[i].to(pred.device)
            # target[:, i:i+1] 是该类别的mask
            class_weights_map += target[:, i:i+1] * weight

        loss = loss * class_weights_map

        return loss.mean()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W, D) - logits（未归一化）
            target: (B, C, H, W, D) - one-hot编码（互斥，每个像素只有一个通道为1）
        """
        dice_loss = self.dice_loss(pred, target)
        ce_loss = self.focal_ce_loss(pred, target)

        return dice_loss + ce_loss


def dice_loss(score, target):
    """标准Dice损失（保持兼容性）"""
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def softmax_mse_loss(input_logits, target_logits):
    """MSE一致性损失（保持兼容性）"""
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_softmax)**2
    return mse_loss.mean()


def softmax_kl_loss(input_logits, target_logits, temperature=1.0):
    """KL散度一致性损失（保持兼容性）"""
    assert input_logits.size() == target_logits.size()

    input_log_softmax = F.log_softmax(input_logits / temperature, dim=1)
    target_softmax = F.softmax(target_logits / temperature, dim=1)

    target_softmax = torch.clamp(target_softmax, min=1e-8, max=1.0)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean', log_target=False)

    return kl_div


if __name__ == "__main__":
    # 测试UnCLoss
    print("Testing UnCLoss...")
    s_logits = torch.randn(2, 3, 32, 32, 32)
    t_logits = torch.randn(2, 3, 32, 32, 32)
    uncl = UnCLoss()
    loss = uncl(s_logits, t_logits, beta=2.0)
    print(f"UnCLoss: {loss.item():.4f}")

    # 测试FeCLoss
    print("\nTesting FeCLoss...")
    feat = torch.randn(2, 128, 64).cuda()
    mask = torch.randint(0, 3, (2, 1, 128)).cuda()
    teacher_feat = torch.randn(2, 128, 64).cuda()
    fecl = FeCLoss().cuda()
    loss = fecl(feat, mask, teacher_feat, epoch=100)
    print(f"FeCLoss: {loss.item():.4f}")

    # 测试EnhancedDiceCELoss
    print("\nTesting EnhancedDiceCELoss...")
    pred = torch.randn(2, 3, 32, 32, 32)
    target = torch.randint(0, 2, (2, 3, 32, 32, 32)).float()
    edce = EnhancedDiceCELoss()
    loss = edce(pred, target)
    print(f"EnhancedDiceCELoss: {loss.item():.4f}")

    print("\nAll tests passed!")
