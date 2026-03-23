#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch
import torch.optim as optim
import datetime
from utils.utils import time2file_name, initialize_logger, save_checkpoint

from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion, get_test_dataset, get_test_dataloader
from utils.logger import Logger
from train.train_utils import train_vanilla
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results, save_model_test, test_all_results
from termcolor import colored


parser = argparse.ArgumentParser(description='Vanilla Training')   # Vanilla Training使用最基本的训练方法进行模型训练的过程
parser.add_argument('--config_env',default=r"configs/env.yml",
                    help='Config file for the environment')        # 环境配置文件
parser.add_argument('--config_exp', default=r"configs/nyud/hrnet18/matadn1.yml",
                    help='Config file for the experiment')
parser.add_argument("--gpu_id", type=str, default='0,1,3', help='path log files')
parser.add_argument('--use_checkpoint', action='store_true', help='Whether to use checkpoint',default=None)
parser.add_argument('--pretrained_model_path', type=str,default=r"'
parser.add_argument('--model_test', type=str, default=True)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def main():
    # 获取配置文件
    cv2.setNumThreads(0)                                           # 设置多线程处理，这里指使用所有可用CPU
    p = create_config(args.config_env, args.config_exp)            # 配置任务信息等重要信息
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))

    # Get model
    model = get_model(p)
    model = torch.nn.DataParallel(model)                           # 可以将模型包装在该类中，并指定要使用的GPU设备。然后，该类将自动将输入数据划分为多个小批次，在每个GPU上并行地运行模型的前向传播和反向传播，并在多个GPU之间同步梯度更新。
    model = model.cuda()
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    # Get criterion
    criterion = get_criterion(p)
    criterion.cuda()

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Optimizer
    optimizer = get_optimizer(p, model)

    # Dataset
    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
    test_dataset = get_test_dataset(p, val_transforms)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    test_dataloader = get_test_dataloader(p, test_dataset)
    print('Train samples %d - Val samples %d - Test samples %d' %(len(train_dataset), len(val_dataset), len(test_dataset)))
    print('Train transformations:')
    print(train_transforms)
    
    # Resume from checkpoint
    use_checkpoint = args.pretrained_model_path
    if use_checkpoint is not None:
        if os.path.isfile(use_checkpoint):
            print(colored('Restart from checkpoint {}'.format(use_checkpoint), 'blue'))
            checkpoint = torch.load(use_checkpoint, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            torch.save(model, 'ISDLN_slim.pt') 
    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0

    if args.model_test is not None:
       print("Only test!")
       save_model_predictions(p, val_dataloader, model)                           # 对模型进行预测，并保存所有预测图片
       curr_result = eval_all_results(p)
       save_model_test(p, test_dataloader, model)
       test_stats = test_all_results(p)
       sys.exit()

    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr 调整学习率
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.8f}'.format(lr))

        # Train 训练
        print('Train ...')
        eval_train = train_vanilla(p, train_dataloader, model, criterion, optimizer, epoch)             # 训练文件，包括了导入数据、训练、损失函数反向传播
        
        # Evaluate 评估
            # Check if need to perform eval first 检查是否要先进行评估
        if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']: # 为了加速 -> 避免每个时期都进行评估，只在最后10个时期进行测试。
            if epoch + 1 > p['epochs'] - 10:
                eval_bool = True
            else:
                eval_bool = False
        else:
            eval_bool = True

        # Perform evaluation                                                           # 对模型进行评估
        if eval_bool:
            print('Evaluate ...')
            save_model_predictions(p, val_dataloader, model)                           # 对模型进行预测，并保存所有预测图片
            curr_result = eval_all_results(p)  
            print('test ...')
            save_model_test(p, test_dataloader, model)
            test_stats = test_all_results(p)
        # Checkpoint 保存断点
        # print('Checkpoint ...')
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join(p['checkpoint'], 'net_%depoch.pt' % epoch))
        torch.save(model.state_dict(), R'xxxxx')
        # torch.save(model, R'D:\MTL\MTL\model.pt')
        # torch.save(model,)    



    

if __name__ == "__main__":
    main()
