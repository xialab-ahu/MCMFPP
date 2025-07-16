import numpy as np
import torch
from torch import nn
import random
import math
import os
from torch.nn import functional as F
import pandas as pd
def set_seed(seed):#Fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalDiceLoss(nn.Module):
    """ Multi-label focal-dice loss 是一种结合了Focal Loss和Dice Loss的损失函数，旨在解决图像分割任务中的类别不平衡和边界模糊问题"""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: shape (N, C) - raw outputs
        targets: shape (N, C) - 0/1 labels
        """
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
class BinaryDiceLoss(nn.Module):
    """ Dice loss """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1)

        loss = 1 - (2 * num) / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class AsymmetricLoss(nn.Module):
    """ Asymmetric loss """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,
                 reduction='mean'):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        if self.reduction == 'mean':
            loss = -loss.mean()
        else:
            loss = -loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, loss_type='focal', gamma=2.0):
        """
        samples_per_cls: list or tensor, 每个类别的样本数（长度 = 类别数）
        beta: 平滑参数，接近1时权重差距大；常设为0.99 ~ 0.9999
        loss_type: 'bce', 'focal', or 'sigmoid'
        gamma: focal loss 的聚焦参数
        """
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

        effective_num = 1.0 - torch.pow(torch.tensor(self.beta), torch.tensor(samples_per_cls))
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(samples_per_cls)  # normalize
        self.class_weights = weights.unsqueeze(0)  # shape: (1, C)

    def forward(self, logits, labels):
        """
        logits: (N, C) raw output (no sigmoid)
        labels: (N, C) multi-hot tensor
        """
        weights = self.class_weights.to(logits.device)  # shape: (1, C)
        weights = weights.repeat(logits.size(0), 1)  # shape: (N, C)
        labels = labels.float()

        if self.loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction='none')
        elif self.loss_type == 'focal':
            probs = torch.sigmoid(logits)
            ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            p_t = probs * labels + (1 - probs) * (1 - labels)
            focal_loss = (1 - p_t) ** self.gamma * ce_loss
            loss = weights * focal_loss
        else:
            raise ValueError("Unsupported loss type")

        return loss.mean()

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps
    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase
    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


