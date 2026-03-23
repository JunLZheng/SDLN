#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from utils.utils import get_output, mkdir_if_missing
import matplotlib.pyplot as plt
import torch.nn.functional as F


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks 一个通用的性能测量器，可以显示一个或多个任务的性能。 """
    def __init__(self, p):
        self.database = p['train_db_name']
        self.tasks = p.TASKS.NAMES
        self.meters = {t: get_single_task_meter(p, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict


def calculate_multi_task_performance(eval_dict, single_task_dict):                    # 计算多任务学习（MTL）相对于单任务学习（STL）的性能提升
    assert(set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]

        if task == 'depth': # rmse lower is better
            mtl_performance -= (mtl['rmse'] - stl['rmse'])/stl['rmse']

        elif task in ['semseg', 'sal', 'human_parts']: # mIoU higher is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU'])/stl['mIoU']

        elif task == 'normals': # mean error lower is better
            mtl_performance -= (mtl['mean'] - stl['mean'])/stl['mean']

        elif task == 'edge': # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF'])/stl['odsF']

        elif task == 'class': # accuracy higher is better
            mtl_performance += (mtl['accuracy'] - stl['accuracy'])/(stl['accuracy']+1e-9)              # 反之除数为0,如果没有进行单任务训练，那么Multi-task learning performance on test set会很离谱

        elif task == 'regres': # rmse lower is better
            mtl_performance += (mtl['rmse'] - stl['rmse'])/stl['rmse']

        else:
            raise NotImplementedError

    return mtl_performance / num_tasks



def get_single_task_meter(p, database, task):
    """ Retrieve a meter to measure the single-task performance 各种单任务模型的指标，包括图像分割，人体姿态估计等 """
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(database)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter()

    elif task == 'sal':
        from evaluation.eval_sal import SaliencyMeter
        return SaliencyMeter()

    elif task == 'depth':
        from MTL.MTL.evaluation.eval_regres import DepthMeter
        return DepthMeter()

    elif task == 'edge': # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=p['edge_w'])

    elif task == 'class':
        from evaluation.eval_class import ClassMeter
        return ClassMeter()

    elif task == 'regres':
        from evaluation.eval_regres import RegresMeter
        return RegresMeter()

    else:
        raise NotImplementedError


def validate_results(p, current, reference):
    """
        Compare the results between the current eval dict and a reference eval dict. 比较当前评估字典和参考评估字典之间的结果
        Returns a tuple (boolean, eval_dict).
        The boolean is true if the current eval dict has higher performance compared 如果当前评估字典的性能比参考评估字典更好,则布尔值为True。
        to the reference eval dict.
        The returned eval dict is the one with the highest performance.              返回的评估字典是性能最佳的那个。
    """
    tasks = p.TASKS.NAMES

    if len(tasks) == 1: # Single-task performance
        task = tasks[0]
        if task == 'semseg': # Semantic segmentation (mIoU)
            if current['semseg']['mIoU'] > reference['semseg']['mIoU']:
                print('New best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = True
            else:
                print('No new best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = False

        elif task == 'human_parts': # Human parts segmentation (mIoU)
            if current['human_parts']['mIoU'] > reference['human_parts']['mIoU']:
                print('New best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = True
            else:
                print('No new best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = False

        elif task == 'sal': # Saliency estimation (mIoU)
            if current['sal']['mIoU'] > reference['sal']['mIoU']:
                print('New best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = True
            else:
                print('No new best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = False

        elif task == 'depth': # Depth estimation (rmse)
            if current['depth']['rmse'] < reference['depth']['rmse']:
                print('New best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = True
            else:
                print('No new best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = False

        elif task == 'normals': # Surface normals (mean error)
            if current['normals']['mean'] < reference['normals']['mean']:
                print('New best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = True
            else:
                print('No new best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = False

        elif task == 'edge': # Validation happens based on odsF
            if current['edge']['odsF'] > reference['edge']['odsF']:
                print('New best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = True

            else:
                print('No new best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = False

        elif task == 'class': # Validation happens based on accuracy
            if current['class']['accuracy'] > reference['class']['accuracy']:
                print('New best classification  model %.3f -> %.3f' %(reference['class']['accuracy'], current['class']['accuracy']))
                improvement = True

            else:
                print('No new best classification model %.3f -> %.3f' %(reference['class']['accuracy'], current['class']['accuracy']))
                improvement = False

        elif task == 'regres': # Validation happens based on rmse
            if current['regres']['r2'] > reference['regres']['r2']:
                print('New best regression model %.3f -> %.3f' %(reference['regres']['r2'], current['regres']['r2']))
                improvement = True

            else:
                print('No new best regression model %.3f -> %.3f' %(reference['regres']['r2'], current['regres']['r2']))
                improvement = False


    else: # Multi-task performance
        if current['multi_task_performance'] > reference['multi_task_performance']:
            print('New best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = True

        else:
            print('No new best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = False

    if improvement: # Return result
        return True, current

    else:
        return False, reference


@torch.no_grad()
def eval_model(p, val_loader, model):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = p.TASKS.NAMES
    performance_meter = PerformanceMeter(p)

    model.eval()

    for i, batch in enumerate(val_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}
        output = model(images)

        # Measure performance
        performance_meter.update({t: get_output(output[t], t) for t in tasks}, targets)

    eval_results = performance_meter.get_score(verbose = True)
    return eval_results

#
# @torch.no_grad()
# def save_model_predictions(p, val_loader, model):                                 # 输入三个参数：参数字典p，验证数据加载器，模型  # 将预测结果存在result列表中
#     """ Save model predictions for all tasks 保存所有任务的模型预测 """
#
#     #print('Save model predictions to {}'.format(p['save_dir']))
#     model.eval()                                                                  # 进行模型预测
#     tasks = p.TASKS.NAMES
#     save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks}
#     for save_dir in save_dirs.values():
#         mkdir_if_missing(save_dir)
#
#     for ii, sample in enumerate(val_loader):
#         inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
#         img_size = (inputs.size(2), inputs.size(3))                               # 返回空间尺寸 100*100
#         output = model(inputs)
#
#         for task in p.TASKS.NAMES:
#             output_task = get_output(output[task], task).cpu().data.numpy()
#
#             for jj in range(int(inputs.size()[0])):                               # 对每一个输出做处理，inputs.size()[0]为样本数
#                 #if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == 255:
#                     #continue
#                 fname = meta['image'][jj]
#                 # result = cv2.resize(output_task[jj], dsize=(meta['im_size'][1][jj], meta['im_size'][0][jj]), interpolation=p.TASKS.INFER_FLAGVALS[task])  # 根据不同任务选择不同的插值方法
#                 result = output_task
#                 if task == 'depth':
#                     sio.savemat(os.path.join(save_dirs[task], 'val', fname + '.mat'), {'depth': result})
#                 elif task == 'class':
#                     sio.savemat(os.path.join(save_dirs[task], 'val', fname + '.mat'), {'class': result})
#                 elif task == 'regres':
#                     sio.savemat(os.path.join(save_dirs[task], 'val', fname + '.mat'), {'regres': result})
#                 else:
#                     imageio.imwrite(os.path.join(save_dirs[task], 'val', fname + '.png'), result.astype(np.uint8))

import time  # 加在文件开头

@torch.no_grad()
def save_model_predictions(p, val_loader, model):
    """ Save model predictions for all tasks 保存所有任务的模型预测 """

    model.eval()
    tasks = p.TASKS.NAMES
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks}
    for save_dir in save_dirs.values():
        mkdir_if_missing(os.path.join(save_dir, 'val'))  # 确保 val 子目录也存在

    total_time = 0.0
    total_frames = 0
    fps_list = []
    for ii, sample in enumerate(val_loader):
        start_time = time.time()  # ⏱️ 开始计时

        inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
        img_size = (inputs.size(2), inputs.size(3))
        output = model(inputs)

        batch_size = inputs.size(0)
        total_frames += batch_size  # 累加处理的帧数

        for task in p.TASKS.NAMES:
            output_task = get_output(output[task], task).cpu().data.numpy()

            for jj in range(batch_size):
                fname = meta['image'][jj]
                result = output_task

                if task == 'depth':
                    sio.savemat(os.path.join(save_dirs[task], 'val', fname + '.mat'), {'depth': result})
                elif task == 'class':
                    sio.savemat(os.path.join(save_dirs[task], 'val', fname + '.mat'), {'class': result})
                elif task == 'regres':
                    sio.savemat(os.path.join(save_dirs[task], 'val', fname + '.mat'), {'regres': result})
                else:
                    imageio.imwrite(os.path.join(save_dirs[task], 'val', fname + '.png'), result.astype(np.uint8))

        end_time = time.time()  # ⏱️ 结束计时
        elapsed = end_time - start_time
        total_time += elapsed

        # 可选：打印每个 batch 的 FPS
        fps = batch_size / elapsed
        fps_list.append(fps)
        print(f"[{ii+1}/{len(val_loader)}] Batch FPS: {fps:.2f}")

    # 计算平均 FPS
    avg_fps = total_frames / total_time
    avg_fps2 = sum(fps_list) / len(fps_list)
    print(f"\n[INFO] Finished prediction. Average FPS: {avg_fps:.2f} ({total_frames} frames in {total_time:.2f} seconds)")
    print(f"\n[INFO] Finished prediction. Average FPS: {avg_fps2:.2f}")


# --- 提取特征图的字典
feature_maps = {}

# --- 注册 Hook 函数
#   x0 --> primary_class: module.scale_2.refinement.class   primary_regress: module.scale_2.refinement.regres   auxi   module.distillation_scale_2.self_attention.regres.class.attention     module.distillation_scale_2.self_attention.class.regres.attention
#   x1 --> primary_class: module.scale_0.refinement.class   primary_regress: module.scale_0.refinement.regres   auxi   module.distillation_scale_0.self_attention.regres.class.attention     module.distillation_scale_0.self_attention.class.regres.attention
# def register_feature_hook(model, layer_name='module.scale_2.refinement.class'):
def register_feature_hook(model, layer_name=''):
    def hook_fn(module, input, output):
        feature_maps['target'] = output.detach().cpu()

    # 获取目标层
    for name, module in model.named_modules():
        if name == layer_name:
            return module.register_forward_hook(hook_fn)
    raise ValueError(f"Layer {layer_name} not found in model.")

# --- 特征图可视化函数
def save_feature_map(feature_map, save_path_prefix):
    os.makedirs(save_path_prefix, exist_ok=True)
    feature_map = F.interpolate(feature_map, size=(100, 100), mode='bilinear', align_corners=False)
    feature_map = feature_map.squeeze(0)  # 去掉 batch 维度，形状 [C, H, W]
    for i in range(feature_map.shape[0]):
        channel = feature_map[i]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-5)  # 归一化
        plt.imsave(os.path.join(save_path_prefix, f"channel_{i}.png"), channel.numpy(), cmap='viridis')


@torch.no_grad()
def save_model_test(p, test_loader, model):
    """ Save model predictions for all tasks 保存所有任务的模型预测 """

    model.eval()
    # for name, module in model.named_modules():
    #    print(name, module)

    # 注册 Hook
    hook = register_feature_hook(model, layer_name='module.scale_2.refinement.regres')  # 修改为你感兴趣的层  module.scale_2.refinement.regres module.ffn1

    tasks = p.TASKS.NAMES
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)

    for ii, sample in enumerate(test_loader):
        inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
        img_size = (inputs.size(2), inputs.size(3))
        output = model(inputs)

        # 特征图保存（仅保存第一个样本）
        if 'target' in feature_maps:
           feature_save_dir = os.path.join(p['save_dir'], 'feature_maps', f"sample_{ii}")
           save_feature_map(feature_maps['target'], feature_save_dir)

        for task in p.TASKS.NAMES:
            output_task = get_output(output[task], task).cpu().data.numpy()

            for jj in range(int(inputs.size()[0])):
                fname = meta['image'][jj]
                result = output_task
                if task == 'depth':
                    sio.savemat(os.path.join(save_dirs[task], 'test', fname + '.mat'), {'depth': result})
                elif task == 'class':
                    sio.savemat(os.path.join(save_dirs[task], 'test', fname + '.mat'), {'class': result})
                elif task == 'regres':
                    sio.savemat(os.path.join(save_dirs[task], 'test', fname + '.mat'), {'regres': result})
                else:
                    imageio.imwrite(os.path.join(save_dirs[task], 'test', fname + '.png'), result.astype(np.uint8))

    #hook.remove()




def eval_all_results(p):
    """ Evaluate results for every task by reading the predictions from the save dir 通过读取保存目录中的预测结果，对每个任务进行评估 """
    save_dir = p['save_dir']
    results = {}

    if 'edge' in p.TASKS.NAMES:
        from evaluation.eval_edge import eval_edge_predictions
        results['edge'] = eval_edge_predictions(p, database=p['val_db_name'],
                             save_dir=save_dir)

    if 'semseg' in p.TASKS.NAMES:
        from evaluation.eval_semseg import eval_semseg_predictions
        results['semseg'] = eval_semseg_predictions(database=p['val_db_name'],
                              save_dir=save_dir, overfit=p.overfit)

    if 'human_parts' in p.TASKS.NAMES:
        from evaluation.eval_human_parts import eval_human_parts_predictions
        results['human_parts'] = eval_human_parts_predictions(database=p['val_db_name'],
                                   save_dir=save_dir, overfit=p.overfit)

    if 'normals' in p.TASKS.NAMES:
        from evaluation.eval_normals import eval_normals_predictions
        results['normals'] = eval_normals_predictions(database=p['val_db_name'],
                               save_dir=save_dir, overfit=p.overfit)

    if 'sal' in p.TASKS.NAMES:
        from evaluation.eval_sal import eval_sal_predictions
        results['sal'] = eval_sal_predictions(database=p['val_db_name'],
                           save_dir=save_dir, overfit=p.overfit)

    if 'depth' in p.TASKS.NAMES:
        from MTL.MTL.evaluation.eval_depth import eval_depth_predictions
        results['depth'] = eval_depth_predictions(database=p['val_db_name'],
                             save_dir=save_dir, overfit=p.overfit)

    if 'class' in p.TASKS.NAMES:
        from evaluation.eval_class import eval_class_predictions
        results['class'] = eval_class_predictions(database=p['val_db_name'],
                             save_dir=save_dir, overfit=p.overfit)

    if 'regres' in p.TASKS.NAMES:
        from evaluation.eval_regres import eval_regres_predictions
        results['regres'] = eval_regres_predictions(database=p['val_db_name'],
                             save_dir=save_dir, overfit=p.overfit)

    if p['setup'] == 'multi_task': # Perform the multi-task performance evaluation
        single_task_test_dict = {}
        for task, val_dict in p.TASKS.SINGLE_TASK_TEST_DICT.items():
            with open(val_dict, 'r') as f_:                                   # test_dict->val_dict
                 single_task_test_dict[task] = json.load(f_)
        results['multi_task_performance'] = calculate_multi_task_performance(results, single_task_test_dict)
        #print('Multi-task learning performance on test set is %.2f' %(100*results['multi_task_performance']))

    return results

def test_all_results(p):
    """ Evaluate results for every task by reading the predictions from the save dir 通过读取保存目录中的预测结果，对每个任务进行评估 """
    save_dir = p['save_dir']
    results = {}

    if 'edge' in p.TASKS.NAMES:
        from evaluation.eval_edge import eval_edge_predictions
        results['edge'] = eval_edge_predictions(p, database=p['test_db_name'],
                             save_dir=save_dir)

    if 'semseg' in p.TASKS.NAMES:
        from evaluation.eval_semseg import eval_semseg_predictions
        results['semseg'] = eval_semseg_predictions(database=p['test_db_name'],
                              save_dir=save_dir, overfit=p.overfit)

    if 'human_parts' in p.TASKS.NAMES:
        from evaluation.eval_human_parts import eval_human_parts_predictions
        results['human_parts'] = eval_human_parts_predictions(database=p['test_db_name'],
                                   save_dir=save_dir, overfit=p.overfit)

    if 'normals' in p.TASKS.NAMES:
        from evaluation.eval_normals import eval_normals_predictions
        results['normals'] = eval_normals_predictions(database=p['test_db_name'],
                               save_dir=save_dir, overfit=p.overfit)

    if 'sal' in p.TASKS.NAMES:
        from evaluation.eval_sal import eval_sal_predictions
        results['sal'] = eval_sal_predictions(database=p['test_db_name'],
                           save_dir=save_dir, overfit=p.overfit)

    if 'depth' in p.TASKS.NAMES:
        from MTL.MTL.evaluation.eval_depth import eval_depth_predictions
        results['depth'] = eval_depth_predictions(database=p['test_db_name'],
                             save_dir=save_dir, overfit=p.overfit)

    if 'class' in p.TASKS.NAMES:
        from evaluation.eval_class import test_class_predictions
        results['class'] = test_class_predictions(database=p['test_db_name'],
                             save_dir=save_dir, overfit=p.overfit)

    if 'regres' in p.TASKS.NAMES:
        from evaluation.eval_regres import test_regres_predictions
        results['regres'] = test_regres_predictions(database=p['test_db_name'],
                             save_dir=save_dir, overfit=p.overfit)

    if p['setup'] == 'multi_task': # Perform the multi-task performance evaluation
        single_task_test_dict = {}
        for task, test_dict in p.TASKS.SINGLE_TASK_TEST_DICT.items():
            with open(test_dict, 'r') as f_:
                 single_task_test_dict[task] = json.load(f_)
        results['multi_task_performance'] = calculate_multi_task_performance(results, single_task_test_dict)
        #print('Multi-task learning performance on test set is %.2f' %(100*results['multi_task_performance']))

    return results
