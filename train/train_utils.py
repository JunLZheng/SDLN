#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import AverageMeter, ProgressMeter, get_output
import torch
import torch.nn as nn


def get_loss_meters(p):                                                                                          # 根据输入参数模型来返回一个包含损失函数的字典，用于监督训练过程中的指标
    """ Return dictionary with loss meters to monitor training """ # 返回监督训练的loss函数
    all_tasks = p.ALL_TASKS.NAMES
    tasks = p.TASKS.NAMES


    if p['model'] == 'mti_net':                                                                                  # Extra losses at multiple scales 多尺度的额外损失
        losses = {}
        for scale in range(4):                                                                                   # 针对每一个尺度和任务，都创建一个相应的损失函数，添加到损失函数列表中
            for task in all_tasks:
                losses['scale_%d_%s' %(scale, task)] = AverageMeter('Loss scale-%d %s ' %(scale+1, task), ':.4e')
        for task in tasks:                                                                                       # 针对每一个任务，创建一个相应的损失函数，添加到损失函数列表中
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')

    elif p['model'] == 'matadn':                                                                                  # Extra losses at multiple scales 多尺度的额外损失
        losses = {}
        
        for scale in range(4):                                                                                   # 针对每一个尺度和任务，都创建一个相应的损失函数，添加到损失函数列表中
            for task in all_tasks:
                losses['scale_%d_%s' %(scale, task)] = AverageMeter('Loss scale-%d %s ' %(scale+1, task), ':.4e')
        for task in tasks:                                                                                       # 针对每一个任务，创建一个相应的损失函数，添加到损失函数列表中
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')

    elif p['model'] == 'matadn1':                                                                                  # Extra losses at multiple scales 多尺度的额外损失
        losses = {}
        for scale in range(4):                                                                                   # 针对每一个尺度和任务，都创建一个相应的损失函数，添加到损失函数列表中
            for task in all_tasks:
                losses['scale_%d_%s' %(scale, task)] = AverageMeter('Loss scale-%d %s ' %(scale+1, task), ':.4e')
        for task in tasks:                                                                                       # 针对每一个任务，创建一个相应的损失函数，添加到损失函数列表中
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')
            
    elif p['model'] == 'matadn2':                                                                                  # Extra losses at multiple scales 多尺度的额外损失
        losses = {}
        for scale in range(4):                                                                                   # 针对每一个尺度和任务，都创建一个相应的损失函数，添加到损失函数列表中
            for task in all_tasks:
                losses['scale_%d_%s' %(scale, task)] = AverageMeter('Loss scale-%d %s ' %(scale+1, task), ':.4e')
        for task in tasks:                                                                                       # 针对每一个任务，创建一个相应的损失函数，添加到损失函数列表中
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')



    elif p['model'] == 'pad_net':                                                                                # Extra losses because of deepsupervision 由于深度监督（deepsupervision），额外的损失
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')


    else: # Only losses on the main task.
        losses = {task: AverageMeter('Loss %s' %(task), ':.4e') for task in tasks}


    losses['total'] = AverageMeter('Loss Total', ':.4e')
    return losses

def updateBN(model,args):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1


def train_vanilla(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights 使用固定的损失权重进行普通训练"""
    losses = get_loss_meters(p)                                                               # 获得记载损失函数的列表，类似log
    performance_meter = PerformanceMeter(p)                                                   # 获取每一个任务的损失函数，并分别存放在损失函数列表中，可进行重置更新等操作
    progress = ProgressMeter(len(train_loader),                                               # 显示训练或者测试过程中损失函数的变化
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        output = model(images)
        
        # Measure loss and performance
        loss_dict = criterion(output, targets)                                                # 获取损失函数，根据yaml的信息，选择单任务还是双任务，依次选择损失函数
        for k, v in loss_dict.items():                                                        # 更新损失函数
            losses[k].update(v.item())                                                        # 更新损失函数展示
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()                                                         # 根据loss字典进行反向传播训练

        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results
