# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os

# PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]
PROJECT_ROOT_DIR = r"D:\MTL\MTL"

class MyPath(object):
    """
    User-specific path configuration.               # 用户特定路径配置
    """
    @staticmethod
    def db_root_dir(database=''):
        db_root = r'D:/MTL/MTL/dataset/'             # /path/to/databases/
        db_names = {'PASCAL_MT', 'NYUD_MT'}

        if database in db_names:
            return os.path.join(db_root, database)
        
        elif not database:
            return db_root
        
        else:
            raise NotImplementedError

    @staticmethod
    def seism_root():                                    # 边缘检测所需要的数据路径
        return '/path/to/seism/'
