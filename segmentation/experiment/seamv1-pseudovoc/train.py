# ----------------------------------------
# SEAM + SPFR — Segmentation Network Training
# USE_SPFR=False → A组(CE only), USE_SPFR=True → B组(CE+SPFR)
# ----------------------------------------

import torch
import numpy as np
import random
import os
seed = int(os.environ.get('SEED', '1').strip())
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
print(f'Random seed: {seed}')
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import sys
import time

from config import config_dict
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.sync_batchnorm.replicate import patch_replication_callback
from utils.configuration import Configuration
from utils.finalprocess import writelog
from utils.imutils import img_denorm
from net.sync_batchnorm import SynchronizedBatchNorm2d
from utils.visualization import generate_vis, max_norm
from tqdm import tqdm
sys.path.insert(0, 'F:/ImageSegmentation/SEAM-REV-V1')
from spfr import generate_superpixels, resize_superpixels

cfg = Configuration(config_dict)

def spfr_loss_seg(features, pseudo_label, segments, margin=1.0):
	"""L_inter: adjacent superpixels same class→pull, different→push"""
	from spfr.pooling import superpixel_pool, build_adjacency
	C, H, W = features.shape
	device = features.device
	segments = segments.to(device)
	pseudo_label = pseudo_label.to(device)

	prototypes, counts, valid = superpixel_pool(features, segments, min_pixels=10)
	K = prototypes.shape[0]
	if valid.sum() < 2:
		return torch.tensor(0.0, device=device, requires_grad=True)

	sp_labels = torch.full((K,), -1, dtype=torch.long, device=device)
	for k in range(K):
		mask = (segments == k)
		pl_k = pseudo_label[mask]
		pl_k = pl_k[pl_k != 255]
		if len(pl_k) == 0: continue
		unique, cnt = torch.unique(pl_k, return_counts=True)
		sp_labels[k] = unique[cnt.argmax()]

	adj = build_adjacency(segments, valid)
	if adj.shape[0] == 0:
		return torch.tensor(0.0, device=device, requires_grad=True)

	proto_norm = F.normalize(prototypes, p=2, dim=1)
	i_idx, j_idx = adj[:, 0], adj[:, 1]
	dist = torch.norm(proto_norm[i_idx] - proto_norm[j_idx], p=2, dim=1)
	same = (sp_labels[i_idx] == sp_labels[j_idx]) & (sp_labels[i_idx] >= 0)
	diff = (sp_labels[i_idx] != sp_labels[j_idx]) & (sp_labels[i_idx] >= 0) & (sp_labels[j_idx] >= 0)

	loss = torch.tensor(0.0, device=device)
	if same.any(): loss += dist[same].pow(2).mean()
	if diff.any(): loss += torch.clamp(margin - dist[diff], min=0).pow(2).mean()
	return loss / max(1, same.sum() + diff.sum())

def train_net():
	period = 'train'
	transform = 'weak'
	dataset = generate_dataset(cfg, period=period, transform=transform)
	dataloader = DataLoader(dataset, batch_size=cfg.TRAIN_BATCHES, shuffle=cfg.TRAIN_SHUFFLE,
		num_workers=cfg.DATA_WORKERS, pin_memory=True, drop_last=True)

	net = generate_net(cfg, batchnorm=nn.BatchNorm2d)
	if cfg.TRAIN_CKPT:
		net.load_state_dict(torch.load(cfg.TRAIN_CKPT), strict=True)
		print('load pretrained model')
	if cfg.TRAIN_TBLOG:
		tblogger = SummaryWriter(cfg.LOG_DIR)

	print('Use %d GPU' % cfg.GPUS)
	device = torch.device(0)
	if cfg.GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
		parameter_source = net.module
	else:
		parameter_source = net
	net.to(device)
	parameter_groups = parameter_source.get_parameter_groups()
	optimizer = optim.SGD(
		params=[
			{'params': parameter_groups[0], 'lr': cfg.TRAIN_LR, 'weight_decay': cfg.TRAIN_WEIGHT_DECAY},
			{'params': parameter_groups[1], 'lr': 2*cfg.TRAIN_LR, 'weight_decay': 0},
			{'params': parameter_groups[2], 'lr': 10*cfg.TRAIN_LR, 'weight_decay': cfg.TRAIN_WEIGHT_DECAY},
			{'params': parameter_groups[3], 'lr': 20*cfg.TRAIN_LR, 'weight_decay': 0},
		],
		momentum=cfg.TRAIN_MOMENTUM, weight_decay=cfg.TRAIN_WEIGHT_DECAY)
	itr = cfg.TRAIN_MINEPOCH * len(dataset) // cfg.TRAIN_BATCHES
	max_itr = cfg.TRAIN_ITERATION
	max_epoch = max_itr * cfg.TRAIN_BATCHES // len(dataset) + 1
	tblogger = SummaryWriter(cfg.LOG_DIR)
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	with tqdm(total=max_itr) as pbar:
		for epoch in range(cfg.TRAIN_MINEPOCH, max_epoch):
			for i_batch, sample in enumerate(dataloader):
				now_lr = adjust_lr(optimizer, itr, max_itr, cfg.TRAIN_LR, cfg.TRAIN_POWER)
				optimizer.zero_grad()

				inputs, seg_label = sample['image'], sample['segmentation']
				n, c, h, w = inputs.size()

				if cfg.USE_SPFR:
					pred1, feat1 = net(inputs.to(0), return_feat=True)
					loss_ce = criterion(pred1, seg_label.to(0))
					loss_spfr = torch.tensor(0.0, device=inputs.device)
					for b in range(n):
						segments, _ = generate_superpixels(inputs[b].cpu(), n_segments=200, region_size=25, fast=True)
						fh, fw = feat1.shape[2], feat1.shape[3]
						seg_ds = resize_superpixels(segments, fh, fw).to(inputs.device)
						seg_lbl_ds = F.interpolate(seg_label[b].unsqueeze(0).unsqueeze(0).float(),
							size=(fh, fw), mode='nearest').squeeze().long().to(inputs.device)
						ls = spfr_loss_seg(feat1[b], seg_lbl_ds, seg_ds)
						loss_spfr += ls.to(inputs.device) if isinstance(ls, torch.Tensor) else ls
					loss_spfr = loss_spfr / max(n, 1)
					loss = loss_ce + 0.1 * loss_spfr
				else:
					pred1 = net(inputs.to(0))
					loss = criterion(pred1, seg_label.to(0))

				loss.backward()
				optimizer.step()

				pbar.set_description("loss=%g " % (loss.item()))
				pbar.update(1)
				time.sleep(0.001)
				if cfg.TRAIN_TBLOG and itr % 100 == 0:
					inputs1 = img_denorm(inputs[-1].cpu().numpy()).astype(np.uint8)
					label1 = sample['segmentation'][-1].cpu().numpy()
					label_color1 = dataset.label2colormap(label1).transpose((2, 0, 1))
					n, c, h, w = inputs.size()
					seg_vis1 = torch.argmax(pred1[-1], dim=0).detach().cpu().numpy()
					seg_color1 = dataset.label2colormap(seg_vis1).transpose((2, 0, 1))
					tblogger.add_scalar('loss', loss.item(), itr)
					tblogger.add_scalar('lr', now_lr, itr)
					tblogger.add_image('Input', inputs1, itr)
					tblogger.add_image('Label', label_color1, itr)
					tblogger.add_image('SEG1', seg_color1, itr)
				itr += 1
				if itr >= max_itr:
					break
			save_path = os.path.join(cfg.MODEL_SAVE_DIR, '%s_%s_%s_epoch%d.pth' % (
				cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, epoch))
			torch.save(parameter_source.state_dict(), save_path)
			print('%s has been saved' % save_path)
			remove_path = os.path.join(cfg.MODEL_SAVE_DIR, '%s_%s_%s_epoch%d.pth' % (
				cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, epoch - 1))
			if os.path.exists(remove_path):
				os.remove(remove_path)

	save_path = os.path.join(cfg.MODEL_SAVE_DIR, '%s_%s_%s_itr%d_all.pth' % (
		cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, cfg.TRAIN_ITERATION))
	torch.save(parameter_source.state_dict(), save_path)
	if cfg.TRAIN_TBLOG:
		tblogger.close()
	print('%s has been saved' % save_path)
	writelog(cfg, period)

def adjust_lr(optimizer, itr, max_itr, lr_init, power):
	now_lr = lr_init * (1 - itr / (max_itr + 1)) ** power
	optimizer.param_groups[0]['lr'] = now_lr
	optimizer.param_groups[1]['lr'] = 2 * now_lr
	optimizer.param_groups[2]['lr'] = 10 * now_lr
	optimizer.param_groups[3]['lr'] = 20 * now_lr
	return now_lr

if __name__ == '__main__':
	train_net()
