import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision import transforms as T
import imageio
import json
import cv2

import sys
sys.path.append('/home/zhuhe/codes/learning_nerf')
sys.path.append('/home/zhuhe/codes/learning_nerf/lib')
from lib.utils import data_utils
from lib.config import cfg
from utils.nerf.nerf_utils import get_rays, get_rays_np


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()

        # dataset configurations
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        self.input_ratio = kwargs['input_ratio']
        self.batching_strategy = None
        if 'batching_strategy' in kwargs:
            self.batching_strategy = kwargs['batching_strategy']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_rays
        assert kwargs['input_ratio'] == 0.5 or kwargs['input_ratio'] == 1.
        self.half_res = kwargs['input_ratio'] == 0.5
        self.white_bkgd = cfg.task_arg.white_bkgd

        # load json information
        self.json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(self.split))))
        #self.json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))
        self.frames = self.json_info['frames']
        if self.split != 'train':
            self.frames = self.frames[::10]

        # load camera rays over the whole dataset
        if self.batching_strategy == 'cross':
            self.load_all_frames()


    def __len__(self):
        if self.split != 'train':
            return len(self.frames)
        if self.batching_strategy == 'cross':
            assert self.rays_rgb is not None
            return len(self.rays_rgb)
        return cfg.ep_iter
        

    def __getitem__(self, idx):
        if self.split != "train":
            # load target image and pose
            img_path = os.path.join(self.data_root, self.frames[idx]['file_path'][2:]+'.png')  
            target_img = (np.array(imageio.v2.imread(img_path)) / 255.).astype(np.float32)
            pose = np.array(self.frames[idx]['transform_matrix']).astype(np.float32)
            
            # deal with white background
            if self.white_bkgd:
                target_img = target_img[...,:3] * target_img[...,-1:] + (1 - target_img[...,-1:])
            else:
                target_img = target_img[...,:3]

            # deal with resolution
            H, W, C = target_img.shape     # (H,W,3)
            camera_angle_x = float(self.json_info['camera_angle_x'])
            focal = .5 * W / np.tan(.5 * camera_angle_x)
            
            if self.half_res:
                H = H // 2  # resolution should be integers
                W = W // 2
                focal = focal / 2.
                target_img = cv2.resize(target_img, (W, H), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA is suitable for image downsamp

            # intrinsics
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

            # get camera rays
            rays_d, rays_o = get_rays_np(H, W, K, pose[:3,:4])
            rays_d, rays_o = rays_d.reshape(-1,3), rays_o.reshape(-1,3)

            # return batch
            rays = np.stack([rays_d, rays_o], axis=1)
            if self.split == 'test':
                return {
                    'rays': rays,           # (H*W,2,3)
                    'target': target_img,   # (H,W,3)
                    'meta': {
                        'H': H, 'W': W
                    }
                }
            return {'rays': rays,           # (H*W,2,3)
                    'target': target_img}   # (H,W,3)
        
        if self.batching_strategy == 'cross':
            item = self.rays_rgb[idx]         # (3,3)
            rays, target = item[:2], item[2]  # rays:(2,3) target(3,)
            return {'rays': rays, 'target': target}
        
        elif self.batching_strategy == 'within':
            # randomly sample one image from the training set
            img_idx = np.random.choice(len(self.frames))

            # load target image and pose
            img_path = os.path.join(self.data_root, self.frames[img_idx]['file_path'][2:]+'.png')  
            target_img = (np.array(imageio.v2.imread(img_path)) / 255.).astype(np.float32)
            pose = np.array(self.frames[img_idx]['transform_matrix']).astype(np.float32)
            
            # deal with white background
            if self.white_bkgd:
                target_img = target_img[...,:3] * target_img[...,-1:] + (1 - target_img[...,-1:])
            else:
                target_img = target_img[...,:3]

            # deal with resolution
            H, W, C = target_img.shape     # (H,W,4)
            camera_angle_x = float(self.json_info['camera_angle_x'])
            focal = .5 * W / np.tan(.5 * camera_angle_x)
            
            if self.half_res:
                H = H // 2  # resolution should be integers
                W = W // 2
                focal = focal / 2.
                target_img = cv2.resize(target_img, (W, H), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA is suitable for image downsamp

            # intrinsics
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

            # get camera rays
            rays_d, rays_o = get_rays_np(H, W, K, pose[:3,:4])

            # sample a batch of rays
            select_inds = np.random.choice(H*W, size=self.batch_size, replace=False)
            rays_d = rays_d.reshape(-1,3)[select_inds]
            rays_o = rays_o.reshape(-1,3)[select_inds]
            target = target_img.reshape(-1,C)[select_inds]

            # returm batch
            rays = np.stack([rays_d, rays_o], axis=1)
            return {'rays': rays,       # (B,2,3)
                    'target': target}   # (B,3)

            

    def pose_spherical(self, theta, phi, radius):
        c2w = self.trans_t(radius)
        c2w = self.rot_phi(phi/180. * np.pi) @ c2w
        c2w = self.rot_theta(theta/180. * np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w  # camera to world

    def trans_t(self, t):
        return torch.Tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,t],
            [0,0,0,1]]).float()

    def rot_phi(self, phi):
        return torch.Tensor([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1]]).float()

    def rot_theta(self, th):
        return torch.Tensor([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1]]).float()

    def load_all_frames(self):
        self.imgs = []
        self.cam_poses = []

        for frame in self.frames:
            fname = os.path.join(self.data_root, frame['file_path'][2:]+'.png')
            self.imgs.append(imageio.v2.imread(fname))
            self.cam_poses.append(frame['transform_matrix'])  # c2w tranformation matrix for each image(view)

        self.imgs = (np.array(self.imgs) / 255.).astype(np.float32)   # normalization, keep all 4 channels RGBA (N,H,W,4)
        self.cam_poses = np.array(self.cam_poses).astype(np.float32)  # (N,4,4)
        # white_bkgd
        if self.white_bkgd:
            """
                images -- RGBA
                A is Alpha channel representing pixel transparency
                A = 0 -- totally transparent
                A = 255 -- opaque
                self.imgs[...,:3] -- RGB, self.imgs[...,-1] -- A
            ---------------------------------------------------------
                self.imgs = self.imgs[...,:3]*self.imgs[...,-1:] + 1.*(1-self.imgs[...,-1:])
                A weighted sum of RGB values and the color white(with 1 denoting the color white)
            """
            self.imgs = self.imgs[...,:3]*self.imgs[...,-1:] + (1-self.imgs[...,-1:])

        # focal
        self.H, self.W = self.imgs[0][:2]
        self.camera_angle_x = float(self.json_info['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)

        # half res
        if self.half_res:
            self.H = self.H // 2  # resolution should be integers
            self.W = self.W // 2
            self.focal = self.focal / 2.
            for img in self.imgs:
                # cv2.resize方法先指定W后指定H
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA is suitable for image downsamp
        
        # camera intrinsics
        self.K = np.array([
            [self.focal, 0, 0.5*self.W],
            [0, self.focal, 0.5*self.H],
            [0, 0, 1]
        ])

        # render_poses (40,4,4)  TODO: 作用是什么？
        self.render_poses = torch.stack([self.pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], dim=0)

        # get rays_rgb
        self.load_all_rays()

    def load_all_rays(self):
        """ load rays in minibatch, sampling rays cross all images
        """
        # gather all rays across all views
        rays = np.stack([get_rays_np(self.H, self.W, self.K, p[:3,:4]) for p in self.cam_poses], axis=0)  # (N_img,1+1,H,W,3)

        # concat
        rays_rgb = np.concatenate([rays, self.imgs[:,None]], axis=1)  # (N_img,1+1+1,H,W,3) rd+ro+rgb
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4])                # (N_img,H,W,1+1+1,3)
        rays_rgb = rays_rgb.reshape(-1, rays_rgb.shape[-2], rays_rgb.shape[-1]).astype(np.float32)  #(N_rays=N_img*H*W,1+1+1,3)

        self.rays_rgb = rays_rgb  # rd+ro+rgb