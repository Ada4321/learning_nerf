import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/home/zhuhe/codes/learning_nerf')
sys.path.append('/home/zhuhe/codes/learning_nerf/lib')
sys.path.append('/home/zhuhe/codes/learning_nerf/lib/networks/nerf/')
from lib.config import cfg
from nerf import NeRF
from utils.nerf.nerf_utils import get_rays

class Network(nn.Module):
    """Wrapping NeRF model and volume rendering blocks
    """
    def __init__(self):
        super(Network, self).__init__()
        # configurations ============================================
        task_arg = cfg.task_arg
        # chunkify
        self.chunk_size = task_arg.chunk_size
        self.chunk_size_ray = task_arg.chunk_size_ray
        self.cascade_sampling = len(task_arg.cascade_samples) > 1
        self.white_bkgd = task_arg.white_bkgd
        # cascade sampling
        self.N_samples = task_arg.cascade_samples[0]
        if self.cascade_sampling:
            self.N_samples_fine = task_arg.cascade_samples[1]
        # scene bounds
        self.bds_dict = task_arg.scene_bounds
        # stratified sampling
        self.stratified = task_arg.stratified

        # init nerf models(coarse and fine) =========================
        self.model = NeRF()
        self.model_fine = NeRF() if self.cascade_sampling else None
        self.model_dict = {
            'coarse': self.model,
            'fine': self.model_fine
        }

    def batchify(self, fn, chunk):
        if chunk is None:
            return fn
        return lambda inputs: torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)

    def query_from_nerf(self, inputs, viewdirs, sampling_state):
        """
        Prepares inputs and applies nerf querying.
            inputs: input point coordinates xyz (N_rays, N_points, coor_dim)
            viewdirs: input viewpoints (N_rays, view_dim)
            sampling state: for cascade sampling, coarse or fine
            netchunk: chunk size for batchification
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # (N_allpoints, coor_dim)
        input_dirs = viewdirs[:, None].expand(inputs.shape)          # (N_rays, 1, view_dim) => (N_rays, N_points, view_dim)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])  # (N_allpoints, view_dim)
        inputs_flat = torch.cat([inputs_flat, input_dirs_flat], dim=-1)  # (N_allpoints, coor_dim+view_dim)

        outputs_flat = self.batchify(self.model_dict[sampling_state], self.chunk_size)(inputs_flat)       # (N_allpoints, 4)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])  # (N_rays, N_points, 4)
        return outputs

    def forward(self, rays):
        """Render rays
        Args:
            rays: input rays(parametrized with direction+origin)

        Returns:
            all_res:
                rgb_map:
                disp_map:
                acc_map:
                extras:
        """
        # create ray batch
        # rays (B,2,3)
        if rays.ndim == 4:
            rays = rays.squeeze(0)
        rays_d, rays_o = rays.float().transpose(0,1)  # rays_d (B,3)  rays_o (B,3)
        # get viewdirs
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  #(B,3)
        # get bounds
        near, far = self.bds_dict['near'], self.bds_dict['far']
        near, far = near*torch.ones_like(rays_d[...,:1]), far*torch.ones_like(rays_d[...,:1])  #(B,1)
        rays = torch.cat([rays_d, rays_o, near, far, viewdirs], dim=-1)  #(B,11)

        # render
        all_ret = self.batchify_rays(rays)

        for k in all_ret:
            k_sh = list(rays_d.shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        return all_ret

    def batchify_rays(self, rays_flat):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, rays_flat.shape[0], self.chunk_size_ray):
            ret = self.render_rays(rays_flat[i:i+self.chunk_size_ray])  # result dict
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k:torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret  # result dict of all rays

    def render_rays(self, rays):
        """Volume rendering one minibatch of rays
        Args:
            rays: one minibatch of rays

        Returns:

        """
        # parsing ray components
        N_rays = rays.shape[0]
        rays_d, rays_o = rays[:,:3], rays[:,3:6]  # (B,3)
        near, far = rays[:, 6], rays[:, 7]        # (B,)
        viewdirs = rays[:,-3:]                    # (B,3)

        # sampling raw points along the rays
        # get bins along z-axis
        t_vals = torch.linspace(0, 1, self.N_samples, device=rays.device).reshape(1,-1)              # (1,N_samples)
        z_vals = near.reshape(-1,1) * (1. - t_vals) + far.reshape(-1,1) * t_vals  # (B,N_samples)
        # stratified sampling
        if self.stratified:
            t_rand = torch.rand(z_vals.shape, device=rays.device)                     # (B, N_samples)
            mid = 0.5 * (z_vals[...,:-1] + z_vals[...,1:])                # (B, N_samples-1)
            lower = torch.cat([z_vals[...,:1], mid], dim=-1)      # (B, N_samples)
            upper = torch.cat([mid, z_vals[...,-1:]], dim=-1)     # (B, N_samples)
            z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[:,None] + z_vals[...,None] * rays_d[:,None]  # (B, N_samples, 3)

        # prediction of the raw network
        
        raw = self.query_from_nerf(inputs=pts, viewdirs=viewdirs, sampling_state='coarse')
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d)

        # prediction of the fine network
        if self.cascade_sampling:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            # finer sampling
            z_vals_mid = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = self.sample_pdf(z_vals_mid, weights[...,1:-1], self.N_samples_fine).detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

            # finer results
            fine = self.query_from_nerf(inputs=pts, viewdirs=viewdirs, sampling_state='fine')
            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(fine, z_vals, rays_d)

        # return results
        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if self.cascade_sampling:
            ret.update({'rgb_map_0': rgb_map_0, 'disp_map_0': disp_map_0, 'acc_map_0': acc_map_0})
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

        for k in ret:
            if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def raw2outputs(self, raws, z_vals, rays_d):
        """Transforms density and rgb to output maps(by performing volume rendering)
        Args:
            raws: [num_rays, num_samples, 4]
            z_vals: [num_rays, num_samples]
            rays_d: [num_rays, 3]

        Returns:
            rgb_map: [num_rays, 2], estimated RGB color of a ray
            disp_map:
            weights: [num_rays, num_samples], weights assigned to each sampled point(indicating the likelyhood of being a surface point)
            acc_map: [num_rays], sum of weights along each ray, accumulated likelyhood of occupation
            depth_map: [num_rays], estimated distance to object
        """
        density = raws[...,-1]  # [num_rays, num_samples]
        density_slice = density[density.shape[0]-1]
        rgb = raws[...,:-1]     # [num_rays, num_samples, 3]

        # convert density to alpha
        raw2alpha = lambda densiy, dists: 1. - torch.exp(-densiy*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.tensor(1e8).expand(dists[...,:1].shape).to(dists.device)], dim=-1)
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [num_rays, num_samples]

        alpha = raw2alpha(density, dists)  # [num_rays, num_samples]

        # weights
        weights = alpha * torch.cumprod(torch.cat([torch.ones(alpha[...,:1].shape).to(alpha.device), 1.-alpha+1e-10], dim=-1), dim=-1)[...,:-1]

        # output maps
        rgb_map = torch.sum(weights[...,None] * rgb, dim=-2)
        depth_map = torch.sum(weights * z_vals, dim=-1)
        acc_map = torch.sum(weights, dim=-1)
        disp_map = 1. / torch.max(1e-8 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)+1e-8) # the larger the depth, the smaller the disparity

        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map.reshape(-1,1))

        return rgb_map, disp_map, acc_map, weights, depth_map

    def sample_pdf(self, bins, weights, N_samples):
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        # Invert CDF
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
        u = u.contiguous().to(bins.device)
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples