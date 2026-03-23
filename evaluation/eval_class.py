# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import os.path
import numpy as np
import glob
import torch
import json
import scipy.io as sio


def eval_class(loader, folder):

    correct = 0 # 用于统计正确预测的数量
    total = 0 # 用于统计总共的样本数量
    n_valid = 0.0

    for i, sample in enumerate(loader):

        # if i % 500 == 0:
            #print('Evaluating class: {} of {} objects'.format(i, len(loader)))

        # Load result 加载结果
        filename = os.path.join(folder, sample['meta']['image'] + '.mat')    # 表明一个数据的文件名
        pred = sio.loadmat(filename)['class'].astype(np.float32)
        pred = torch.tensor(pred)

        label = sample['class']

        # 计算分类准确率
        _, predicted = torch.max(pred, 1)
        predicted = predicted.tolist()
        correct += (predicted == label).sum().item()      
        total += label.size
    
    accuracy = correct / total
    eval_result = dict()
    eval_result['accuracy'] = accuracy

    return eval_result


class ClassMeter(object):
    def __init__(self):
        self.accuracy = 0.0
        # self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.correct = 0 # 用于统计正确预测的数量
        self.total = 0 # 用于统计总共的样本数量

    @torch.no_grad()
    def update(self, pred, gt):                                   # gt(groundtrue)
        pred, gt = pred.squeeze(), gt.squeeze()
        _, predicted = torch.max(pred, 1)
        self.correct = (predicted == gt).sum().item()
        self.total = gt.size(0)
        if self.total != 0:  # 避免除以零的错误
            self.accuracy = self.correct / self.total

    def reset(self):
        # self.rmses = []
        # self.log_rmses = []
        self.accuracy = []
        self.correct = 0
        self.total = 0

    def get_score(self, verbose=True):
        eval_result = dict()
        #eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        #eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)
        if self.total != 0:  # 避免除以零的错误
            self.accuracy = self.correct / self.total
        eval_result['accuracy'] = self.accuracy
        if verbose:                                                                    # 选择是否需要打印出每一个预测结果
            #print('Results for class prediction')
            for x in eval_result:
                spaces = ''
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result
        

def eval_class_predictions(database, save_dir, overfit=False):

    # Dataloaders                                                       # 评价数据导入
    if database == 'NYUD':
        from MTL.MTL.data.nyud_yuan import NYUD_MT 
        gt_set = 'val'
        db = NYUD_MT(split=gt_set, do_class=True, overfit=overfit)
    
    elif database == 'ALGAE':
        from data.nyud import NYUD_MT 
        gt_set = 'val'
        db = NYUD_MT(split=gt_set, do_class=True, overfit=overfit)
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_class'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model 评估模型       
    # print('Evaluate the saved images (class)')
    eval_results = eval_class(db, os.path.join(save_dir, 'class','val'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print results 打印结果
    # print('Results for class Estimation')
    for x in eval_results:
        spaces = ''
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))

    return eval_results

def test_class_predictions(database, save_dir, overfit=False):

    # Dataloaders                                                       # 评价数据导入
    if database == 'NYUD':
        from MTL.MTL.data.nyud_yuan import NYUD_MT 
        gt_set = 'test'
        db = NYUD_MT(split=gt_set, do_class=True, overfit=overfit)
    
    elif database == 'ALGAE':
        from data.nyud import NYUD_MT 
        gt_set = 'test'
        db = NYUD_MT(split=gt_set, do_class=True, overfit=overfit)
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test_ture' + '_class'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model 评估模型       
    # print('Evaluate the saved images (class)')
    eval_results = eval_class(db, os.path.join(save_dir, 'class','test'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print results 打印结果
    # print('Results for class Estimation')
    for x in eval_results:
        spaces = ''
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))

    return eval_results
