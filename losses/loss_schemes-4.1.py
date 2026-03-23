#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskLoss(nn.Module):
    def __init__(self, loss_ft, task):
        super(SingleTaskLoss, self).__init__()
        self.loss_ft = loss_ft
        self.task = task

    
    def forward(self, pred, gt):
        out = {self.task: self.loss_ft(pred[self.task], gt[self.task])}
        out['total'] = out[self.task]
        return out


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):                     # 接收三个参数，任务列表，损失函数列表，损失函数权重(各个任务的重要性)
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    
    def forward(self, pred, gt):                                                                     # 接收预测值pred和真实值gt作为输入。在前向传播过程中，它会遍历任务列表，对每个任务使用对应的损失函数计算损失值，并将结果保存在一个字典中。然后，它根据损失权重对各个任务的损失值进行加权求和，并将总损失值添加到输出字典中返回
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
        return out


class PADNetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,
                    loss_weights: dict):
        super(PADNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    
    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            # pred_ = F.interpolate(pred['initial_%s' %(task)], img_size, mode='bilinear')
            pred_ = pred['initial_%s' %(task)]
            gt_ = gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out['deepsup_%s' %(task)] = loss_
            total += self.loss_weights[task] * loss_

        # Losses at output  
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out[task] = loss_
            total += self.loss_weights[task] * loss_

        out['total'] = total

        return out


class MTINetLoss(nn.Module):                                                          # 需要修改******************
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,     # 输入任务名 辅助任务名 根据任务选择损失函数 各个人物之间的权重比值    get_loss是计算单个任务的损失值，这个MTINetLoss则是计算整个模型额损失函数
                    loss_weights: dict):
        super(MTINetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear_layer = nn.Linear(6, 1)
        self.class_7  = nn.Sequential(
            nn.Conv2d(7, 7, 1, bias=False),
            nn.Flatten(),
            nn.Linear(7 * 25 * 25, 7),
            #nn.Linear(500, 6)
            )
        self.regres_1  = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Flatten(),
            nn.Linear(1 * 25 * 25, 1),
            #nn.Linear(500, 1)
            )

    
    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]
        
        # Losses initial task predictions at multiple scales (deepsup) 在多个尺度上损失初始任务预测（深监督）
        for scale in range(4):
            pred_scale = pred['deep_supervision']['scale_%s' %(scale)] # 这是过程特征图，没有经过蒸馏输出
            pred_scale = {t: F.interpolate(pred_scale[t], size=(25, 25), mode='bilinear', align_corners=True) for t in self.auxilary_tasks}
            pred_scale = {t: pred_scale[t] for t in self.auxilary_tasks}  # 创建一个字典[class:特征图,regres:特征图]
            # losses_scale = {t: self.loss_ft[t](self.global_avg_pool(pred_scale[t]).view(pred_scale[t].shape[0],-1), gt[t]) for t in self.auxilary_tasks}
            # losses_scale = {t: self.loss_ft[t](self.global_avg_pool[t](pred_scale[t]), gt[t]) for t in self.auxilary_tasks}
            losses_scale = {}
            for t in self.auxilary_tasks:
                if t == 'class':
                    re = self.class_7(pred_scale[t])
                elif t == 'regres':
                    re = self.regres_1(pred_scale[t])
                else:
                    raise ValueError(f"未知的任务类型: {t}")
                    
                losses_scale[t] = self.loss_ft[t](re, gt[t])

            for k, v in losses_scale.items():
                out['scale_%d_%s' %(scale, k)] = v
                total += self.loss_weights[k] * v

        # Losses at output 计算输出的损失函数
        losses_out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        for k, v in losses_out.items():
            out[k] = v
            total += self.loss_weights[k] * v

        out['total'] = total

        return out


class MATADNLoss(nn.Module):                                                          # 需要修改******************
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,     # 输入任务名 辅助任务名 根据任务选择损失函数 各个人物之间的权重比值    get_loss是计算单个任务的损失值，这个MTINetLoss则是计算整个模型额损失函数
                loss_weights: dict
                    ):
        super(MATADNLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear_layer = nn.Linear(6, 1)
        self.class_6  = nn.Sequential(
            nn.Conv2d(9, 9, 1, bias=False),
            nn.Flatten(),
            nn.Linear(9 * 25 * 25, 7),
            #nn.Linear(500, 6)
            )
        self.regres_1  = nn.Sequential(
            nn.Conv2d(9, 9, 1, bias=False),
            nn.Flatten(),
            nn.Linear(9 * 25 * 25, 1),
            #nn.Linear(500, 1)
            )

    
    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]
        
        # Losses initial task predictions at multiple scales (deepsup) 在多个尺度上损失初始任务预测（深监督）
        for scale in range(2): 
            pred_scale = pred['deep_supervision']['scale_%s' %(scale)] # 这是过程特征图，没有经过蒸馏输出
            pred_scale = {t: F.interpolate(pred_scale[t], size=(25, 25), mode='bilinear', align_corners=True) for t in self.auxilary_tasks}
            pred_scale = {t: pred_scale[t] for t in self.auxilary_tasks}  # 创建一个字典[class:特征图,regres:特征图]
            losses_scale = {}
            for t in self.auxilary_tasks:
                if t == 'class':
                    re = self.class_6(pred_scale[t])
                elif t == 'regres':
                    re = self.regres_1(pred_scale[t])
                else:
                    raise ValueError(f"未知的任务类型: {t}")
                    
                losses_scale[t] = self.loss_ft[t](re, gt[t])

            for k, v in losses_scale.items():
                out['scale_%d_%s' %(scale, k)] = v
                total += self.loss_weights[k] * v

        # Losses at output 计算输出的损失函数
        losses_out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        for k, v in losses_out.items():
            
            out[k] = v
            total += self.loss_weights[k] * v
        out['total'] = total

        return out

class MATADN1Loss(nn.Module):                                                          # 需要修改******************
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,     # 输入任务名 辅助任务名 根据任务选择损失函数 各个人物之间的权重比值    get_loss是计算单个任务的损失值，这个MTINetLoss则是计算整个模型额损失函数
                    loss_weights: dict):
        super(MATADN1Loss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear_layer = nn.Linear(6, 1)
        self.class_6  = nn.Sequential(
            nn.Conv2d(9, 9, 1, bias=False),
            nn.Flatten(),
            nn.Linear(9 * 25 * 25, 7),
            #nn.Linear(500, 6)
            )
        self.regres_1  = nn.Sequential(
            nn.Conv2d(9, 9, 1, bias=False),
            nn.Flatten(),
            nn.Linear(9 * 25 * 25, 1),
            #nn.Linear(500, 1)
            )

    
    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]
        
        # Losses initial task predictions at multiple scales (deepsup) 在多个尺度上损失初始任务预测（深监督）
        for scale in range(2): 
            pred_scale = pred['deep_supervision']['scale_%s' %(scale)] # 这是过程特征图，没有经过蒸馏输出
            pred_scale = {t: F.interpolate(pred_scale[t], size=(25, 25), mode='bilinear', align_corners=True) for t in self.auxilary_tasks}


            pred_scale = {t: pred_scale[t] for t in self.auxilary_tasks}  # 创建一个字典[class:特征图,regres:特征图]
            losses_scale = {}
            for t in self.auxilary_tasks:
                if t == 'class':
                    re = self.class_6(pred_scale[t])
                elif t == 'regres':
                    re = self.regres_1(pred_scale[t])
                else:
                    raise ValueError(f"未知的任务类型: {t}")
                    
                losses_scale[t] = self.loss_ft[t](re, gt[t])

            for k, v in losses_scale.items():
                out['scale_%d_%s' %(scale, k)] = v
                total += self.loss_weights[k] * v

        # Losses at output 计算最终输出的损失函数
        losses_out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        for k, v in losses_out.items():
            out[k] = v
            total += self.loss_weights[k] * v

        out['total'] = total

        return out
    

class MATADN2Loss(nn.Module):                                                          # 需要修改******************
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,     # 输入任务名 辅助任务名 根据任务选择损失函数 各个人物之间的权重比值    get_loss是计算单个任务的损失值，这个MTINetLoss则是计算整个模型额损失函数
                    loss_weights: dict):
        super(MATADN2Loss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear_layer = nn.Linear(6, 1)
        self.class_6  = nn.Sequential(
            nn.Conv2d(9, 9, 1, bias=False),
            nn.Flatten(),
            nn.Linear(9 * 25 * 25, 7),
            #nn.Linear(500, 6)
            )
        self.regres_1  = nn.Sequential(
            nn.Conv2d(9, 9, 1, bias=False),
            nn.Flatten(),
            nn.Linear(9 * 25 * 25, 1),
            #nn.Linear(500, 1)
            )

    
    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]
        
        # Losses initial task predictions at multiple scales (deepsup) 在多个尺度上损失初始任务预测（深监督）
        for scale in range(2): 
            pred_scale = pred['deep_supervision']['scale_%s' %(scale)] # 这是过程特征图，没有经过蒸馏输出
            pred_scale = {t: F.interpolate(pred_scale[t], size=(25, 25), mode='bilinear', align_corners=True) for t in self.auxilary_tasks}
            pred_scale = {t: pred_scale[t] for t in self.auxilary_tasks}  # 创建一个字典[class:特征图,regres:特征图]
            # losses_scale = {t: self.loss_ft[t](self.global_avg_pool(pred_scale[t]).view(pred_scale[t].shape[0],-1), gt[t]) for t in self.auxilary_tasks}
            # losses_scale = {t: self.loss_ft[t](self.global_avg_pool[t](pred_scale[t]), gt[t]) for t in self.auxilary_tasks}
            losses_scale = {}
            for t in self.auxilary_tasks:
                if t == 'class':
                    re = self.class_6(pred_scale[t])
                elif t == 'regres':
                    re = self.regres_1(pred_scale[t])
                else:
                    raise ValueError(f"未知的任务类型: {t}")
                    
                losses_scale[t] = self.loss_ft[t](re, gt[t])

            for k, v in losses_scale.items():
                out['scale_%d_%s' %(scale, k)] = v
                total += self.loss_weights[k] * v

        # Losses at output 计算输出的损失函数
        losses_out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        for k, v in losses_out.items():
            out[k] = v
            total += self.loss_weights[k] * v

        out['total'] = total

        return out