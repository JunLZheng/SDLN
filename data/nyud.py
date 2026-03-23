# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import sys
import tarfile
import cv2

from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib

from utils.mypath import MyPath
from utils.utils import mkdir_if_missing
from data.google_drive import download_file_from_google_drive
import h5py
class NYUD_MT(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes semantic segmentation and depth prediction.

    Data can also be found at:
    https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view?usp=sharing

    """

    GOOGLE_DRIVE_ID = '14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw'
    FILE = 'NYUD_MT.tgz'

    def __init__(self,
                 root=MyPath.db_root_dir('NYUD_MT'),
                 download=False,                                       # 是否需要在谷歌驱动下载数据集
                 split='test',
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_edge=False,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 do_class=False,
                 do_regres=False
                 ):

        self.root = root

        if download:
            self._download()

        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.retname = retname

        # Original Images                                              # 数据特征，输入数据
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, 'images')                      # 获取数据存放地址
        
        '''
        # Edge Detection
        self.do_edge = do_edge
        self.edges = []
        _edge_gt_dir = os.path.join(root, 'edge')

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(root, 'segmentation')

        # Surface Normals
        self.do_normals = do_normals
        self.normals = []
        _normal_gt_dir = os.path.join(root, 'normals')

        # Depth
        self.do_depth = do_depth
        self.depths = []
        _depth_gt_dir = os.path.join(root, 'depth')
        '''

        # classification
        self.do_class = do_class
        self.classs = []
        _class_gt_dir = os.path.join(root, 'class')

        # regression
        self.do_regres = do_regres
        self.regress = []
        _regres_gt_dir = os.path.join(root, 'regres')

        # train/val/test splits are pre-cut                           # 训练集、验证集、测试集分割
        _splits_dir = os.path.join(root, 'gt_sets')

        # print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), 'r') as f:          # splt中有train和val两个变量
                lines = f.read().splitlines()                                                     # 读取列表中的内容，并将其每一个元素放置在不同的行

            for ii, line in enumerate(lines):

                # Images
                _image = os.path.join(_image_dir, line + '.mat')                                  # 原输入是图片，需要修改成mat
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))                                             # 删除字符串末尾换行符后添加至im_ids中
                
                '''
                # Edges
                _edge = os.path.join(self.root, _edge_gt_dir, line + '.npy')
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = os.path.join(self.root, _semseg_gt_dir, line + '.png')
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                # Surface Normals
                _normal = os.path.join(self.root, _normal_gt_dir, line + '.npy')
                assert os.path.isfile(_normal)
                self.normals.append(_normal)

                # Depth Prediction
                _depth = os.path.join(self.root, _depth_gt_dir, line + '.npy')
                assert os.path.isfile(_depth)
                self.depths.append(_depth)
                '''

                # classification
                _class = os.path.join(self.root, _class_gt_dir, line + '.mat')
                assert os.path.isfile(_class)
                self.classs.append(_class)

                # regression
                _regres = os.path.join(self.root, _regres_gt_dir, line + '.mat')
                assert os.path.isfile(_regres)
                self.regress.append(_regres)
        '''
        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))
        '''
        if self.do_class:
            assert (len(self.images) == len(self.classs))
        if self.do_regres:
            assert (len(self.images) == len(self.regress))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        # print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):   # 一个sample里面包含一个image、一个class、一个regres
        sample = {}
        _img = self._load_img(index)
        sample['image'] = _img
        '''
        if self.do_edge:
            _edge = self._load_edge(index)
            if _edge.shape != _img.shape[:2]:                                                       # 如果边缘检测任务的标签和输入空间尺寸不一样，则会进行插值
                _edge = cv2.resize(_edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)    # [::-1]用于颠倒列表中元素的顺序，这里是将高度和宽度信息颠倒，使得宽度在前面
            sample['edge'] = _edge

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                print('RESHAPE SEMSEG')
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            if _normals.shape[:2] != _img.shape[:2]:
                _normals = cv2.resize(_normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            if _depth.shape[:2] != _img.shape[:2]:
                print('RESHAPE DEPTH')
                _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth
        '''
        '''
        if self.do_class:                                                                          ###################################### 类别标签数据需要重新更改
            _class = self._load_class(index)
            if isinstance(_class['label'], str):
                print('One-Hot Encoding')
                unique_labels = np.unique(_class['label'])
                _class = cv2.resize(_class, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['do_class'] = _class
        '''
        if self.do_class:  # 检查是否需要处理类别标签
            #one_hot_labels = []
            _class = self._load_class(index)  # 从数据集中加载类别标签
            sample['class'] = _class
            # print(sample['class'],'nnnnnnnnnnnnnnnnnnnnnnnnnn')
        '''
               if isinstance(_class, str):  # 检查类别标签是否为字符串类型
               print('One-Hot Encoding')  # 输出信息表明要进行独热编码操作
               # 获取唯一的类别标签
               unique_labels = np.unique(_class)
               # 将字符串标签转换为数值型
               label_to_index = {label: index for index, label in enumerate(unique_labels)}
               numeric_label = label_to_index[_class]

               # 进行独热编码
               num_labels = len(unique_labels)
               one_hot_label = np.zeros(num_labels)
               one_hot_label[numeric_label] = 1

               # 将独热编码后的标签存储在列表中（这里可能需要更合适的数据结构）
               # 例如，可以将独热编码后的标签添加到一个列表中，每次处理一个样本的标签，以便最后存储所有标签的独热编码
               # 这里的代码需要根据实际情况调整，以适应你的数据结构和存储需求
               # 以下示例将独热编码后的标签存储在一个名为 'one_hot_labels' 的列表中
               # one_hot_labels.append(one_hot_label)
               print(one_hot_label,'1111111')
        '''
            


        if self.do_regres:                                                                           ###################################### 浓度标签数据需要重新更改
            # all_regres = []
            _regres = self._load_regres(index)
            sample['regres'] = _regres
            # print(sample['regres'],'ttttttttttttttttttttttttttttt')
            '''
            while True:
                _regres = self._load_regres(index)
                if _regres is None:
                    break
            all_regres.append(_regres)
            print(all_regres,'a00000000')
            if all_regres :
                all_regres = np.array(all_regres)
                normalized_regres = (all_regres - all_regres.min()) / (all_regres.max() - all_regres.min())
                sample['do_regres'] = normalized_regres
            else:
                print("No data to process.")
            '''

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),                                       # meta用于存储图像序号，以及对应尺寸 
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)
        '''
        if 'image' in sample:
            value_shape = None
            value = sample['image']
    
           # 检查值的类型是否是列表或者 NumPy 数组，如果是，则打印其形状
            if isinstance(value, (list, np.ndarray)):
               value_shape = np.array(value).shape
               print(f"The shape of 'class' value is: {value_shape}")
            else:
               print("The value corresponding to 'class' is not a list or NumPy array.")
        else:
            print("'class' is not present in the sample dictionary.")
        '''
        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):                                                                                # 打开图像将其转为RGB格式后，在转为32位的数组形式
        with h5py.File(self.images[index], 'r') as mat:                          # h5py.File(hyper_path, 'r')将MATLAB文件中mat结构体写入，并将其储存在mat文件中
            _img =np.float32(np.array(mat['hsis']))                                  # hsis  multis
        #_img = np.load(self.images[index],allow_pickle=True).astype(np.float32)
        #_img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_edge(self, index):
        _edge = np.load(self.edges[index]).astype(np.float32)
        return _edge

    def _load_semseg(self, index):
        # Note: We ignore the background class as other related works.                                         # 注意：我们忽略了背景类别，就像其他相关的工作一样。
        _semseg = np.array(Image.open(self.semsegs[index])).astype(np.float32)
        _semseg[_semseg == 0] = 256                                                                            # 可能是为了将像素值转换到从 0 开始的索引，例如，原本值为 1 的像素变为 0
        _semseg = _semseg - 1
        return _semseg

    def _load_depth(self, index):
        _depth = np.load(self.depths[index])
        return _depth

    def _load_normals(self, index):
        _normals = np.load(self.normals[index])
        return _normals
    
    def _load_class(self, index):
        with h5py.File(self.classs[index], 'r') as mat:                          # h5py.File(hyper_path, 'r')将MATLAB文件中mat结构体写入，并将其储存在mat文件中
            _class =np.float32(np.array(mat['label']))
        return _class
    
    def _load_regres(self, index):
        with h5py.File(self.regress[index], 'r') as mat:                          # h5py.File(hyper_path, 'r')将MATLAB文件中mat结构体写入，并将其储存在mat文件中
            _regres =np.float32(np.array(mat['conc']))
        return _regres

    def _download(self):
        _fpath = os.path.join(MyPath.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(MyPath.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def __str__(self):
        return 'NYUD Multitask (split=' + str(self.split) + ')'


def test_mt():                                                      # 主程序
    import torch
    import data.custom_transforms as tr
    import  matplotlib.pyplot as plt 
    from torchvision import transforms
    '''
    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.ScaleNRotate(rots=(-2, 2), scales=(.75, 1.25),
                                                    flagvals={'image': cv2.INTER_CUBIC,
                                                              'edge': cv2.INTER_NEAREST,
                                                              'semseg': cv2.INTER_NEAREST,
                                                              'normals': cv2.INTER_LINEAR,
                                                              'depth': cv2.INTER_LINEAR,}),
                                    tr.FixedResize(resolutions={'image': (512, 512),
                                                                'edge': (512, 512),
                                                                'semseg': (512, 512),
                                                                'normals': (512, 512),
                                                                'depth': (512, 512),},
                                                   flagvals={'image': cv2.INTER_CUBIC,
                                                             'edge': cv2.INTER_NEAREST,
                                                             'semseg': cv2.INTER_NEAREST,
                                                             'normals': cv2.INTER_LINEAR,
                                                             'depth': cv2.INTER_LINEAR,}),
                                    tr.AddIgnoreRegions(),
                                    tr.ToTensor()])
                                    '''
    dataset = NYUD_MT(split='train', transform=None, retname=True,
                      do_edge=False,
                      do_semseg=False,
                      do_normals=False,
                      do_depth=False,
                      do_class=True,
                      do_regres=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=5)

    for i, sample in enumerate(dataloader):                                                              # 去展示加载的数据 # 未修改
        print(i)
        for j in range(sample['image'].shape[0]):
            f, ax_arr = plt.subplots(5)
            for k in range(len(ax_arr)):
                ax_arr[k].cla()                                                                          # 清楚单个子图
            ax_arr[0].imshow(np.transpose(sample['image'][j], (1,2,0)))
            ax_arr[1].imshow(sample['edge'][j,0])
            ax_arr[2].imshow(sample['semseg'][j,0]/40)
            ax_arr[3].imshow(np.transpose(sample['normals'][j], (1,2,0)))
            max_depth = torch.max(sample['depth'][j,0][sample['depth'][j,0] != 255]).item()
            ax_arr[4].imshow(sample['depth'][j,0]/max_depth) # Not ideal. Better is to show inverse depth.

            plt.show()
        break


if __name__ == '__main__':
    test_mt()                                 # 在训练过程中也没有调用test_mt()
