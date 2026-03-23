#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

""" 
    MTI-Net implementation based on HRNet backbone 
    https://arxiv.org/pdf/2001.06902.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import BasicBlock
# from models.layers import SEBlock,SABlock
# from models.padnet import MultiTaskDistillationModule
from einops import rearrange
from functools import partial

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        #self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn
    

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1 当stride != 1时，self.conv1和self.downsample层都对输入进行下采样(不应该说的是下采样，而是卷积残差连接)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.GELU()               # nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.groups = 6

    def forward(self, x):
        identity = x
        

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)                               # 去除



        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = identity

        return out 

class InitialTaskPredictionModule(nn.Module):
    """ Module to make the inital task predictions """
    def __init__(self, p, auxilary_tasks, input_channels, task_channels):
        super(InitialTaskPredictionModule, self).__init__()        
        self.auxilary_tasks = auxilary_tasks

        # Per task feature refinement + decoding 每个任务的特征细化和解码
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.ModuleDict({task: nn.Sequential(BasicBlock(channels, channels), BasicBlock(channels, channels)) for task in self.auxilary_tasks})
        
        else:                                                                           # 从解码的第二个阶段开始，将cat之后的张量进行下采样(通道数变少，跟编码阶段的通道数一样)
            refinement = {}
            for t in auxilary_tasks:
                downsample = nn.Sequential(nn.Conv2d(input_channels, task_channels, 1, bias=False), 
                                nn.BatchNorm2d(task_channels))
                refinement[t] = nn.Sequential(BasicBlock(input_channels, task_channels, downsample=downsample),
                                              BasicBlock(task_channels, task_channels)
                                                )
            self.refinement = nn.ModuleDict(refinement)

        #self.decoders = nn.ModuleDict({task: nn.Conv2d(task_channels, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1) for task in self.auxilary_tasks})
        self.decoders = nn.ModuleDict()

        for task in self.auxilary_tasks:
            self.decoders[task] = nn.Sequential(nn.Conv2d(task_channels, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1), # p.AUXILARY_TASKS.NUM_OUTPUT[task]
                                                 )



    def forward(self, features_curr_scale, features_prev_scale=None):
        '''
        if features_prev_scale is not None: # Concat features that were propagated from previous scale 连接来自先前尺度的传播特征(如果有FPM模块传入的特征，则进行该操作)
            x = {t: torch.cat((features_curr_scale, F.interpolate(features_prev_scale[t], features_curr_scale.shape[-2:], mode='bilinear', align_corners=True)), 1) for t in self.auxilary_tasks}
            # x = {t: torch.cat((features_curr_scale, F.interpolate(features_prev_scale[t], scale_factor=2, mode='bilinear')), 1) for t in self.auxilary_tasks}   # 将前阶段和现阶段的特征图进行融合 # 这边去缩放倍数为2，只针对于32 16 8 4 之前俺的任务有用，我的输入是100*100，插值取整会出现不一样的结果


        else:
            x = {t: features_curr_scale for t in self.auxilary_tasks}
        '''

        if features_prev_scale is not None: 
           x = {}
           for t in self.auxilary_tasks:
               interpolated_feature = F.interpolate(features_prev_scale[t], features_curr_scale.shape[-2:], mode='bilinear', align_corners=True)
               concatenated_features = torch.cat((features_curr_scale, interpolated_feature), 1)
               x[t] = concatenated_features
        else:
           x = {t: features_curr_scale for t in self.auxilary_tasks}


        # Refinement + Decoding
        out = {}
        for t in self.auxilary_tasks:
            out['features_%s' %(t)] = self.refinement[t](x[t])                       # features_class.features_regre属于out字典的其中两个键名
            
            out[t] = self.decoders[t](out['features_%s' %(t)])                       # 分别将其输入到decoder中

        return out
    
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    
class TIPM(nn.Module):
    def __init__(self, auxilary_tasks, per_task_channels,
    ):
        super().__init__()
        self.auxilary_tasks = auxilary_tasks
        self.per_task_channels = per_task_channels
        self.conv_split0 = nn.Conv2d(per_task_channels*2, per_task_channels, 1, bias=False)
        self.conv_split1 = nn.Conv2d(per_task_channels, per_task_channels, kernel_size=3, stride=1, padding=1, groups=per_task_channels, bias=False)
        self.to_q = nn.Linear(per_task_channels, per_task_channels, bias=False) # 创建了一个线性层，其中输入的维度为dim，输出的维度为dim_head * heads，且不包含偏置项，该处只用到了一头，dim_head=dim;heads=dim_stage//dim=dim//dim
        self.to_k = nn.Linear(per_task_channels, per_task_channels, bias=False) # Linear中的shape是nn.Linear(in_features,out_features)，输入为[batch_size, n, in_features]，输出为[batch_size, n, out_features]
        self.to_v = nn.Linear(per_task_channels, per_task_channels, bias=False)
        self.proj = nn.Linear(per_task_channels, per_task_channels, bias=True)
        self.dim = per_task_channels
        self.conv_split = nn.Conv2d(per_task_channels, per_task_channels, 1, bias=False)
        #self.se = nn.ModuleDict({task: nn.Conv2d(per_task_channels*2, per_task_channels, 1, bias=False) for task in self.auxilary_tasks})
        self.se = nn.ModuleDict({task: SEBlock(self.per_task_channels) for task in self.auxilary_tasks})

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]                b:Batch,批处理大小,表示一个batch中的图像数量
        return out: [b,h,w,c]
        """
        """
        x_in: [b,c,h,w]                b:Batch,批处理大小,表示一个batch中的图像数量
        return out: [b,c,h,w]
        """
        # b, h, w, c = x_in.shape
        
        x = torch.cat([x_in['features_%s' %(task)] for task in self.auxilary_tasks], 1)  
        x = self.conv_split0(x)
        x = self.conv_split1(x)
        x = x.permute(0, 2, 3, 1)
        b, h, w, c = x.shape
        x = x.reshape(b,h*w,c)                                                                 # 以10张100*100*19为例，（10，100*100，19）
        q = self.to_q(x)                                                                      #（10，100*100，19），输入为19个神经元，每个神经元输入10000*1的空间信息向量，输出为19个神经元，得到的是10000*1的QKV           
        k = self.to_k(x)
        v = self.to_v(x)
        q = q.transpose(-2, -1)                                                                    # 表示将 q 的最后两个维度进行坐标轴互换（b,heads,c,hw）
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)                                                            # 对q的最后一个维度除以对应的范数（p=2为2范数）
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q；@表示矩阵乘法                               # (b,heads,hw,c)
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 2, 1)    # Transpose
        x = x.reshape(b, h * w, self.per_task_channels)
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)
        out_c = self.conv_split(out_c)
        out = {}
        for task in self.auxilary_tasks:
            out[task] = self.se[task](out_c) + x_in['features_%s' %(task)]  
        

        return out

class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r 
        self.Pool2d = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Sequential(nn.Conv2d(channels, channels//self.r, 1, bias=False),          # nn.Linear(channels, channels//self.r)
                                     nn.ReLU(),
                                     nn.Conv2d(channels//self.r, channels, 1, bias=False),          # nn.Linear(channels//self.r, channels)
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(self.Pool2d(x))
        # squeeze = self.squeeze(torch.mean(x, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)

class SABlock(nn.Module):
    """ Spatial self-attention block """                                # 空间自注意力模块
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())                   # nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.groups = 3

    def forward(self, x_in):
        attention_mask = self.attention(x_in)
        features = self.conv(x_in)
        return torch.mul(features, attention_mask)   
   
class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation                                                 # 将蒸馏之后，辅助任务的特征用了起来
        We apply an attention mask to features from other tasks and                     # 我们对来自其他任务的特征应用注意力掩码，并将结果添加为残差
        add the result as a residual.
    """
    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks                                                              # 主要任务
        self.auxilary_tasks = auxilary_tasks                                            # 辅助任务
        self.self_attention = {}                                                        # 储存注意力机制
        
        for t in self.tasks:                                                            # 为每个主任务创建注意力模块
            other_tasks = [a for a in self.auxilary_tasks if a != t]                    # 选择除了当前主任务以外的其他任务作为辅助任务
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})                                  # 为当前主任务创建注意力模块，并存储在 self_attention 中
        self.self_attention = nn.ModuleDict(self.self_attention)


    def forward(self, x):
        adapters = {}
        for t in self.tasks:
            adapters[t] = {}
            for a in self.auxilary_tasks:
                if a != t:
                   adapters[t][a] = self.self_attention[t][a](x['features_%s' %(a)])

        # adapters = {t: {a: self.self_attention[t][a](x['features_%s' %(a)]) for a in self.auxilary_tasks if a!= t} for t in self.tasks}
        #out = {t: x['features_%s' %(t)] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0) for t in self.tasks}         # 将所有辅助任务的输出特征进行stack(不是cat)
        out = {}
        for t in self.tasks:
            feature_t = x['features_%s' %(t)]
            adapter_values = [v for v in adapters[t].values()]
            stacked_adapters = torch.stack(adapter_values)
            summed_adapters = torch.sum(stacked_adapters, dim=0)
            out[t] = feature_t + summed_adapters

        return out
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[1, 3, 5, 7], stride=1, conv_groups=[1, 3, 6, 9]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, inplans//2, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=inplans//2)
        self.conv_2 = conv(inplans, inplans//2, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=inplans//2)
        self.conv_3 = conv(inplans, inplans//2, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=inplans//2)


        self.se = SEWeightModule(inplans//2)
        self.split_channel = inplans//2
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Conv2d((inplans//2)*3, planes, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)

        feats = torch.cat((x1, x2, x3), dim=1)
        feats = feats.view(batch_size, 3, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)

        x_se = torch.cat((x1_se, x2_se, x3_se), dim=1)
        attention_vectors = x_se.view(batch_size, 3, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(3):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        
        out = self.fc(out)

        return out
class FeedForward(nn.Module):
    def __init__(self, dim,out_dim,
    ):
        super().__init__()
        self.per_task_channels = dim
        self.conv_split0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_q = nn.Linear(dim, dim, bias=False) # 创建了一个线性层，其中输入的维度为dim，输出的维度为dim_head * heads，且不包含偏置项，该处只用到了一头，dim_head=dim;heads=dim_stage//dim=dim//dim
        self.to_k = nn.Linear(dim, dim, bias=False) # Linear中的shape是nn.Linear(in_features,out_features)，输入为[batch_size, n, in_features]，输出为[batch_size, n, out_features]
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dim = dim
        self.project_out = nn.Conv2d(dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]                b:Batch,批处理大小,表示一个batch中的图像数量
        return out: [b,h,w,c]
        """
        """
        x_in: [b,c,h,w]                b:Batch,批处理大小,表示一个batch中的图像数量
        return out: [b,c,h,w]
        """
        # b, h, w, c = x_in.shape
        
        x = self.conv_split0(x_in)
        x = x.permute(0, 2, 3, 1)
        b, h, w, c = x.shape
        x = x.reshape(b,h*w,c)                                                                 # 以10张100*100*19为例，（10，100*100，19）
        q = self.to_q(x)                                                                      #（10，100*100，19），输入为19个神经元，每个神经元输入10000*1的空间信息向量，输出为19个神经元，得到的是10000*1的QKV           
        k = self.to_k(x)
        v = self.to_v(x)
        q = q.transpose(-2, -1)                                                                    # 表示将 q 的最后两个维度进行坐标轴互换（b,heads,c,hw）
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)                                                            # 对q的最后一个维度除以对应的范数（p=2为2范数）
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q；@表示矩阵乘法                               # (b,heads,hw,c)
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 2, 1)    # Transpose
        x = x.reshape(b, h * w, self.per_task_channels)
        out = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)
        out = self.project_out(out)
        return out

    
class MATADN1(nn.Module):
    """ 
        MTI-Net implementation based on HRNet backbone 
        https://arxiv.org/pdf/2001.06902.pdf
    """
    def __init__(self, p, backbone, backbone_channels, heads):
        super(MATADN1, self).__init__()
        # General
        self.tasks = p.TASKS.NAMES
        self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.num_scales = len(backbone_channels)
        self.channels = backbone_channels        

        # Backbone
        self.backbone = backbone

        self.ffn1 = PSAModule(self.channels[0]+ self.channels[1],(self.channels[0]+ self.channels[1])//2 )
        self.ffn2 = PSAModule(self.channels[2] + self.channels[3],(self.channels[2] + self.channels[3])//2)
        
        # Feature Propagation Module 特征传播模块
        self.fpm_scale_2 = TIPM(self.auxilary_tasks,(self.channels[2]+self.channels[3])//2)

        # Initial task predictions at multiple scales 多尺度下的初始任务预测
        self.scale_2 = InitialTaskPredictionModule(p, self.auxilary_tasks, ((self.channels[2] + self.channels[3])//2), ((self.channels[2] + self.channels[3])//2))
        self.scale_0 = InitialTaskPredictionModule(p, self.auxilary_tasks, ((self.channels[0] + self.channels[1] + self.channels[2] + self.channels[3])//2), ((self.channels[0]+ self.channels[1])//2))
        

        # Distillation at multiple scales 多尺度蒸馏
        self.distillation_scale_0 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, ((self.channels[0]+ self.channels[1])//2))
        self.distillation_scale_2 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, ((self.channels[2] + self.channels[3])//2))
        
        # Feature aggregation through HRNet heads 通过HRNet头进行特征聚合
        self.heads = heads 
        

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # Backbone
        x = self.backbone(x)                                                  # x0[1, 18, 25, 25] [1, 36, 13, 13] [1, 72, 7, 7] [1, 144, 4, 4]

        feature1_0 = F.interpolate(x[1], x[0].shape[-2:], mode='bilinear', align_corners=True)
        features_10cat = torch.cat((x[0], feature1_0), 1)   # [1, 54, 25, 25]
        features_10cat = self.ffn1(features_10cat)
        feature3_2 = F.interpolate(x[3], x[2].shape[-2:], mode='bilinear', align_corners=True)
        feature_32cat = torch.cat((x[2], feature3_2), 1) # [1, 216, 7, 7]
        feature_32cat = self.ffn2(feature_32cat)

        
        # Predictions at multiple scales 多尺度预测
        x_2 = self.scale_2(feature_32cat)   #[1, 216, 7, 7]
        #print(x_2['features_class'].shape,'11111111')                                 
        x_2_fpm = self.fpm_scale_2(x_2)     # [1, 216, 7, 7] 
        #print(x_2_fpm['class'].shape,'22222222')                                
        x_0 = self.scale_0(features_10cat, x_2_fpm)   # [1, 18, 25, 25]   
        #print(x_0['features_class'].shape,'3333333')                               
        
        out['deep_supervision'] = {'scale_0': x_0, 'scale_1': x_2}         

        # Distillation + Output 蒸馏输出
        features_0 = self.distillation_scale_0(x_0)  # [1, 18, 25, 25]
        #print(features_0['class'].shape,'44444444444')                          
        features_2 = self.distillation_scale_2(x_2) # [1, 216, 7, 7]
        #print(features_2['class'].shape,'5555555555555')                            
        multi_scale_features = {t: [features_0[t],  features_2[t]] for t in self.tasks}

        # Feature aggregation 特征聚合
        for t in self.tasks:
            out[t] = self.heads[t](multi_scale_features[t])
                       
        return out
