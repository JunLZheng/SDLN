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
from sklearn.metrics import r2_score

def eval_regres(loader, folder):
    all_label_val = []
    all_pred_val = []

    for i, sample in enumerate(loader):
        #if i % 500 == 0:
            #print('Evaluating regres: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.mat')
        pred = sio.loadmat(filename)['regres'].astype(np.float32)
        label = sample['regres']

        # 将单个数值的标签和预测值添加到列表中
        all_label_val.append(label)
        all_pred_val.append(pred)



    all_label_val = np.concatenate(all_label_val)
    all_pred_val = np.concatenate(all_pred_val)

    
    r2 = r2_score(all_label_val, all_pred_val)
    mse = np.mean((all_label_val - all_pred_val) ** 2)
    rmse = np.sqrt(np.mean((all_label_val - all_pred_val)**2))

    # 将可能包含 float32 类型的数据转换为 float 类型
    r2 = float(r2)
    mse = float(mse)
    rmse = float(rmse)

    eval_result = dict()
    eval_result['R2'] = r2
    eval_result['mse'] = mse
    eval_result['rmse'] = rmse

    return eval_result




class RegresMeter(object):
    def __init__(self):
        self.r2 = 0.0
        self.mse = 0.0
        self.rmse = 0.0


    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze().cpu().numpy(), gt.squeeze().cpu().numpy()
        self.r2 = r2_score(gt, pred)
        self.mse = np.mean((gt - pred) ** 2)
        self.rmse = np.sqrt(np.mean((gt - pred)**2))

    def reset(self):
        self.r2s = []
        self.mses = []
        self.rmses = []
        
    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['r2'] = self.r2
        eval_result['mse'] = self.mse
        eval_result['rmse'] = self.rmse

        if verbose:
            # print('Results for regres prediction')
            for x in eval_result:
                spaces = ''
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result
        

def eval_regres_predictions(database, save_dir, overfit=False):

    # Dataloaders
    if database == 'NYUD':
        from MTL.MTL.data.nyud_yuan import NYUD_MT 
        gt_set = 'val'
        db = NYUD_MT(split=gt_set, do_regres=True, overfit=overfit)
    
    elif database == 'ALGAE':
        from data.nyud import NYUD_MT 
        gt_set = 'val'
        db = NYUD_MT(split=gt_set, do_regres=True, overfit=overfit)
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_regres'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    # print('Evaluate the saved images (regres)')
    eval_results = eval_regres(db, os.path.join(save_dir, 'regres','val'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print results
    # print('Results for Regres Estimation')
    for x in eval_results:
        spaces = ''
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))

    return eval_results

def test_regres_predictions(database, save_dir, overfit=False):

    # Dataloaders
    if database == 'NYUD':
        from MTL.MTL.data.nyud_yuan import NYUD_MT 
        gt_set = 'test'
        db = NYUD_MT(split=gt_set, do_regres=True, overfit=overfit)
    
    elif database == 'ALGAE':
        from data.nyud import NYUD_MT 
        gt_set = 'test'
        db = NYUD_MT(split=gt_set, do_regres=True, overfit=overfit)
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test_ture' + '_regres'
    fname = os.path.join(save_dir, base_name + '.json')

    # test the model
    # print('Test the saved images (regres)')
    test_results = eval_regres(db, os.path.join(save_dir, 'regres','test'))
    with open(fname, 'w') as f:
        json.dump(test_results, f)

    # Print results
    # print('Results for Regres Estimation')
    for x in test_results:
        spaces = ''
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, test_results[x]))

    return test_results
