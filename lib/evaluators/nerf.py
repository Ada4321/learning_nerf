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
        self.psnrs = []
        os.system('mkdir -p ' + cfg.result_dir)
        # os.system('mkdir -p ' + cfg.result_dir + '/vis')

    def evaluate(self, output, batch):
        # assert image number = 1
        H, W = batch['meta']['H'].item(), batch['meta']['W'].item()
        pred_rgb = output['rgb_map'].reshape(H, W, 3).detach().cpu().numpy()
        gt_rgb = batch['target'].reshape(H, W, 3).detach().cpu().numpy()
        psnr_item = psnr(gt_rgb, pred_rgb, data_range=1.)
        self.psnrs.append(psnr_item)

        save_path = os.path.join(cfg.result_dir, 'vis/res_{}.jpg'.format(self.img_id))
        imageio.imwrite(save_path, img_utils.horizon_concate(gt_rgb, pred_rgb))
        self.img_id += 1

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        print(ret)
        self.psnrs = []
        print('Save visualization results to {}'.format(cfg.result_dir))
        json.dump(ret, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w'))
        return ret
    
    def __call__(self, batch):
        rays, target = batch['rays'], batch['target']
        output = self.net(rays)

        self.evaluate(output, batch)
        return self.summarize()