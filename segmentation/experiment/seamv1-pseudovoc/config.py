# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

# EXP_NAME 支持环境变量 SEED 区分多种子实验:
#   SEED=1 → seamv1-pseudovoc-DRSA  (默认, 高质量 CE only)
#   SEED=2 → seamv1-pseudovoc-DRSA-s2
#   SEED=3 → seamv1-pseudovoc-DRSA-s3
# 注意: Windows cmd 中 set SEED=2 && python 会给值附带尾随空格, 用 strip() 防御
_base_exp = os.environ.get('EXP_NAME', 'seamv1-pseudovoc-DRSA').strip()
_seed = os.environ.get('SEED', '1').strip()
if _seed != '1':
    _base_exp = f'{_base_exp}-s{_seed}'

config_dict = {
				'EXP_NAME': _base_exp,
			'GPUS': 1,

		'DATA_NAME': 'VOCDataset',
		'DATA_YEAR': 2012,
		'DATA_AUG': True,
			'DATA_WORKERS': 0,
		'DATA_MEAN': [0.485, 0.456, 0.406],
		'DATA_STD': [0.229, 0.224, 0.225],
		'DATA_RANDOMCROP': 448,
		'DATA_RANDOMSCALE': [0.5, 1.5],
		'DATA_RANDOM_H': 10,
		'DATA_RANDOM_S': 10,
		'DATA_RANDOM_V': 10,
		'DATA_RANDOMFLIP': 0.5,
			'DATA_PSEUDO_GT': 'F:/ImageSegmentation/SEAM-REV-V1/pseudo_train_drs',
			
			'MODEL_NAME': 'deeplabv1',
			'MODEL_BACKBONE': 'resnet38',
			'MODEL_BACKBONE_PRETRAIN': True,
			'MODEL_BACKBONE_DILATED': False,
			'MODEL_BACKBONE_MULTIGRID': False,
			'MODEL_BACKBONE_DEEPBASE': False,
			'MODEL_NUM_CLASSES': 21,
			'USE_SPFR': False,           # True=B组(SPFR), False=A组(CE only)

		'TRAIN_LR': 0.001,
		'TRAIN_MOMENTUM': 0.9,
		'TRAIN_WEIGHT_DECAY': 0.0005,
		'TRAIN_BN_MOM': 0.0003,
		'TRAIN_POWER': 0.9,
			'TRAIN_BATCHES': 2,
		'TRAIN_SHUFFLE': True,
		'TRAIN_MINEPOCH': 0,
			'TRAIN_ITERATION': 20000,
		'TRAIN_TBLOG': True,

		'TEST_MULTISCALE': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
		'TEST_FLIP': True,
			'TEST_CRF': False,
		'TEST_BATCHES': 1,		
}

config_dict['ROOT_DIR'] = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
config_dict['MODEL_SAVE_DIR'] = os.path.join(config_dict['ROOT_DIR'],'model',config_dict['EXP_NAME'])
config_dict['TRAIN_CKPT'] = None
config_dict['LOG_DIR'] = os.path.join(config_dict['ROOT_DIR'],'log',config_dict['EXP_NAME'])
config_dict['TEST_CKPT'] = os.path.join(config_dict['MODEL_SAVE_DIR'],'deeplabv1_resnet38_VOCDataset_itr20000_all.pth')

sys.path.insert(0, os.path.join(config_dict['ROOT_DIR'], 'lib'))
