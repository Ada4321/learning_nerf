import numpy as np
import torch


def get_rays(H, W, K, c2w):
    """
        Getting origins and directions of camera rays(torch implementation)
        Args:
            H: image height
            W: image weight
            K: camera intrinsic
            c2w: camera to world matrix

        Returns:
            rays_o: ray origins
            rays_d: ray directions

    """
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], dim=-1)

    rays_d = torch.matmul(c2w[:3,:3], dirs[:,:,:,None]).reshape(dirs.shape)
    rays_o = torch.broadcast_to(c2w[:3,-1], rays_d)

    return rays_d, rays_o

def get_rays_np(H, W, K, c2w):
    """
    Getting origins and directions of camera rays(numpy implementation)
    Args:
        H: image height
        W: image weight
        K: camera intrinsic
        c2w: camera to world matrix

    Returns:
        rays_o: ray origins
        rays_d: ray directions

    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    """
    Getting coordinates for each pixel in the OpenCV system
        i: np.array([
            [0,1,2,...,W-1],
            [0,1,2,...,W-1],
            ...
            [0,1,2,...,W-1]
        ])
        j: np.array([
            [0,0,0,...,0],
            [1,1,1,...,1],
            ...
            [H-1,H-1,H-1,...,H-1]
        ])
    """
    # ray directions in camera coordinate system
    x = (i - K[0][2]) / K[0][0]   # x coords for each ray vec (H,W)
    y = -(j - K[1][2]) / K[1][1]  # y coords for each ray vec (H,W)
    z = -np.ones_like(i)          # z coords for each ray vec (H,W)
    dirs = np.stack([x,y,z], axis=-1)  # ray direction for each pixel in camera frame (H,W,3)

    # rotate ray directions from camera frame's origin to the world frame
    # only the directions of the vectors matter, so we only have to do the rotation
    rays_d = np.matmul(c2w[:3, :3], dirs[:,:,:,None]).reshape(dirs.shape)  # (H,W,3)

    # compute the origin of all rays in the world frame
    # for every pixel in the same image, the origin of their corresponding rays are the same
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d)  # (3,) => (H,W,3)

    return rays_d, rays_o