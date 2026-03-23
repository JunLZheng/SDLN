# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
from sklearn.metrics import r2_score


class CustomCrossEntropyLoss(nn.Module):      # 添加用于分类的交叉熵损失函数
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'): # ignore_index的使用场景是针对包含填充值或无效标签的情况。例如，在自然语言处理任务中，当使用序列数据进行文本分类时，可能会在输入序列的末尾添加填充标记。这些填充标记在计算损失时应该被忽略，因为它们不包含有用的信息
        super(CustomCrossEntropyLoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, out, label):
        return self.loss_function(out, label.squeeze().long())

'''

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', lambda_reg=0.001):
        super(CustomCrossEntropyLoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction)
        self.lambda_reg = lambda_reg  # 正则化参数

    def forward(self, out, label):
        # 计算交叉熵损失
        ce_loss = self.loss_function(out, label.squeeze().long())
        
        # 计算 L2 正则化项
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)**2  # 使用torch.norm计算 L2 范数并求平方
        l2_reg *= self.lambda_reg
        
        # 将交叉熵损失和正则化项相加作为总损失
        total_loss = ce_loss + l2_reg
        
        return total_loss

class CustomMSELoss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', lambda_reg=0.001):
        super(CustomMSELoss, self).__init__()
        self.loss_function = nn.MSELoss(size_average, reduce, reduction)
        self.lambda_reg = lambda_reg  # 正则化参数

    def forward(self, out, label):
        # 计算均方误差损失
        mse_loss = self.loss_function(out.squeeze(), label.squeeze())
        
        # 计算 L2 正则化项
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)**2  # 使用torch.norm计算 L2 范数并求平方
        l2_reg *= self.lambda_reg
        
        # 将均方误差损失和正则化项相加作为总损失
        total_loss = mse_loss + l2_reg
        
        return total_loss

'''

class CustomMSELoss(nn.Module):               # 添加了用于回归的MSE函数
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(CustomMSELoss, self).__init__()
        self.loss_function = nn.MSELoss(size_average, reduce, reduction)

    def forward(self, out, label):
        label = label.squeeze()
        out = out.squeeze()
        return self.loss_function(out, label)




class SoftMaxwithLoss(Module):
    """
    This function returns cross entropy loss for semantic segmentation 该函数返回语义分割的交叉熵损失
    """

    def __init__(self):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=255)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        loss = self.criterion(self.softmax(out), label)

        return loss


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions 平衡交叉熵损失，可选择性地忽略特定区域
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())
        labels = torch.ge(label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class BinaryCrossEntropyLoss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced. 带有忽略区域的二元交叉熵损失，非平衡。
    """

    def __init__(self, size_average=True, batch_average=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())

        labels = torch.ge(label, 0.5).float()

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = loss_pos + loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used. 深度预测的损失。默认情况下使用L1损失。
    """
    def __init__(self, loss='l1'):
        super(DepthLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError('Loss {} currently not supported in DepthLoss'.format(loss))

    def forward(self, out, label):
        mask = (label != 255)
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(Module):
    """
    L1 loss with ignore labels                           带有忽略标签的L1损失
    normalize: normalization for surface normals         normalize:用于表面法线的归一化。
    """
    def __init__(self, size_average=True, normalize=False, norm=1):
        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = (label != ignore_label)
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
            loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')
        else:
            loss = self.loss_func(torch.masked_select(out, mask), torch.masked_select(label, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss

        return loss
