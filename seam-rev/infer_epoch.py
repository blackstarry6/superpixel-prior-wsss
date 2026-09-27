"""
Batch CAM inference for a given checkpoint.
Usage: python infer_epoch.py --weights <checkpoint.pth> --out_dir <output_dir>
"""
import sys, os
sys.path.insert(0, '.')
import torch, numpy as np, cv2
from PIL import Image
import torch.nn.functional as F
from network.resnet38_SEAM import Net
from collections import OrderedDict
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--voc12_root', default='VOC2012')
parser.add_argument('--val_list', default='voc12/val.txt')
parser.add_argument('--scales', default='0.5,1.0,1.5')
args = parser.parse_args()

device = 'cuda'
scales = [float(s) for s in args.scales.split(',')]
os.makedirs(args.out_dir, exist_ok=True)

# Load model
model = Net()
sd = torch.load(args.weights, map_location='cpu')
nsd = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
model.load_state_dict(nsd, strict=True)
model = model.to(device).eval()
print(f'Model: {args.weights}')

# Load image list + labels
img_list = open(args.val_list).read().strip().split('\n')
cls_labels = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

# Track progress
total = len(img_list); done = 0; skip = 0
for line in tqdm(img_list, desc=os.path.basename(args.out_dir)):
    img_name = line.split(' ')[0][-15:-4]
    out_path = os.path.join(args.out_dir, img_name + '.npy')
    if os.path.exists(out_path):
        skip += 1; continue
    
    label = cls_labels[img_name]
    img_path = os.path.join(args.voc12_root, 'JPEGImages', img_name + '.jpg')
    orig_img = np.asarray(Image.open(img_path))
    H, W = orig_img.shape[:2]

    # Multi-scale inference
    cam_list = []
    for s in scales:
        hh, ww = int(round(H * s)), int(round(W * s))
        img_s = cv2.resize(orig_img, (ww, hh), interpolation=cv2.INTER_CUBIC)
        img_norm = img_s.astype(np.float32).copy()
        img_norm[..., 0] = (img_norm[..., 0] / 255. - 0.485) / 0.229
        img_norm[..., 1] = (img_norm[..., 1] / 255. - 0.456) / 0.224
        img_norm[..., 2] = (img_norm[..., 2] / 255. - 0.406) / 0.225
        t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            _, cam = model(t)
            cam_fg = cam[:, 1:]
            cam_up = F.interpolate(cam_fg, (H, W), mode='bilinear', align_corners=False)[0].cpu().numpy()
            cam_list.append(cam_up)

            # Horizontal flip
            t_f = torch.flip(t, dims=[-1])
            _, cam_f = model(t_f)
            cam_f = F.interpolate(cam_f[:, 1:], (H, W), mode='bilinear', align_corners=False)[0].cpu().numpy()
            cam_list.append(np.flip(cam_f, axis=-1))

    # Aggregate
    sum_cam = np.sum(cam_list, axis=0)
    sum_cam[sum_cam < 0] = 0
    cam_max = np.max(sum_cam, (1, 2), keepdims=True)
    cam_min = np.min(sum_cam, (1, 2), keepdims=True)
    norm_cam = (sum_cam - cam_min + 1e-5) / (cam_max - cam_min + 1e-5)

    cam_dict = {c: norm_cam[c] for c in range(20) if label[c] > 0}
    np.save(out_path, cam_dict)
    done += 1

print(f'Done: {done} new, {skip} skipped, {total} total')
