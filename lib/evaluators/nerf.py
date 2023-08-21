import sys
sys.path.append('/home/zhuhe/codes/learning_nerf')

from typing import Any
import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch
import lpips
import imageio
from lib.utils import img_utils
import cv2
import json

class Evaluator:
    def __init__(self, net):
        self.net = net
        self.img_id = 0
        # os.system('mkdir -p ' + cfg.result_dir)
        # os.system('mkdir -p ' + cfg.result_dir + '/vis')

    def evaluate(self, output, batch):
        # assert image number = 1
        H, W = batch['meta']['H'].item(), batch['meta']['W'].item()
        pred_rgb = (output['rgb_map']*255.).reshape(H, W, 3).detach().cpu().numpy().astype(np.uint8)
        gt_rgb = (batch['target']*255.).reshape(H, W, 3).detach().cpu().numpy().astype(np.uint8)
        psnr_item = psnr(gt_rgb, pred_rgb, data_range=1.)
        # self.psnrs.append(psnr_item)

        # save predicted rgb
        save_root = os.path.join(cfg.result_dir, 'vis')
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, 'res_{}.jpg'.format(self.img_id))
        imageio.imwrite(save_path, img_utils.horizon_concate(gt_rgb, pred_rgb))

        self.img_id += 1
        
        return psnr_item

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        print(ret)
        self.psnrs = []
        print('Save visualization results to {}'.format(cfg.result_dir))
        json.dump(ret, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w'))
        return ret
    
    def __call__(self, batch):
        output = self.net(batch['rays'])
        return self.evaluate(output, batch)
