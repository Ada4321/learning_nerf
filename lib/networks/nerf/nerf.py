import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg


act_dict = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    '': None
}

class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()

        # configurations
        nerf_cfg = cfg.network.nerf
        xyz_cfg = cfg.network.xyz_encoder
        dir_cfg = cfg.network.dir_encoder
        self.W = nerf_cfg.W
        self.D = nerf_cfg.D
        self.V_D = nerf_cfg.V_D
        self.input_dim_xyz = xyz_cfg.input_dim
        self.input_dim_dir = dir_cfg.input_dim
        self.freq_xyz = xyz_cfg.freq
        self.freq_dir = dir_cfg.freq
        self.skips_xyz = xyz_cfg.skips
        self.skips_dir = dir_cfg.skips
        self.act_xyz = xyz_cfg.act
        self.act_dir = dir_cfg.act
        self.pe_dim_xyz = self.input_dim_xyz * self.freq_xyz * 2
        self.pe_dim_dir = self.input_dim_dir * self.freq_dir * 2

        # positional encoding
        self.pe_xyz = PositionEncoding(self.freq_xyz)
        self.pe_dir = PositionEncoding(self.freq_dir)

        # encoders
        self.xyz_encoder = self.make_encoder(in_dim=self.pe_dim_xyz,
                                             out_dim=self.W+1,
                                             h_dim=self.W,
                                             H=self.D,
                                             skips=self.skips_xyz,
                                             )
        self.dir_encoder = self.make_encoder(in_dim=self.pe_dim_dir+self.W,
                                             out_dim=3,
                                             h_dim=self.W//2,
                                             H=self.V_D,
                                             skips=self.skips_dir,
                                             )

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_dim_xyz, self.input_dim_dir], dim=-1)
        # position encoding
        input_pts = self.pe_xyz(input_pts)
        input_views = self.pe_dir(input_views)

        # predict density
        h = self.encode(encoder=self.xyz_encoder,
                        inputs=input_pts,
                        skips=self.skips_xyz,
                        act=self.act_xyz)
        feature, density = torch.split(h, [h.size()[-1]-1, 1], dim=-1)

        # predict rgb
        input_views = torch.cat([input_views, feature], dim=-1)
        rgb = self.encode(encoder=self.dir_encoder,
                          inputs=input_views,
                          skips=self.skips_dir,
                          act=self.act_dir)

        return torch.cat([rgb, F.relu(density)], dim=-1)  # (N_points_batch, 4)

    def make_encoder(self, in_dim, out_dim, h_dim, H, skips):
        assert H >= 0
        if H == 0:
            return nn.ModuleList([nn.Linear(in_dim, out_dim)])
        return nn.ModuleList([nn.Linear(in_dim, h_dim)] + \
                             [nn.Linear(h_dim, h_dim) if i not in skips else nn.Linear(in_dim+h_dim, h_dim) for i in range(H-1)] + \
                             [nn.Linear(h_dim, out_dim)])

    def encode(self, encoder, inputs, skips, act):
        h = inputs
        for i, linear in enumerate(encoder):
            h = linear(h)  # linear
            if act_dict[act[i]] is not None:  # activation
                h = act_dict[act[i]](h)
            if i in skips:                    # skip connection
                h = torch.cat([h, inputs], dim=-1)
        return h


class PositionEncoding(nn.Module):
    def __init__(self, L):
        super(PositionEncoding, self).__init__()
        self.L = L

    def forward(self, p):
        pe_matrix = torch.zeros(*p.shape, 2*self.L, device=p.device)
        powers = torch.pow(2, torch.linspace(0, self.L-1, self.L)).reshape(1, -1).to(p.device)
        """
        p: (..., input_dim) unsqueeze=> (..., input_dim, 1)
        powers: (1, L)
        ========================>
        pe: (..., input_dim, L)
        """
        #pe = torch.pi * torch.matmul(p.unsqueeze(-1), powers)
        pe = torch.matmul(p.unsqueeze(-1), powers)
        pe_matrix[..., 0::2] = torch.sin(pe)
        pe_matrix[..., 1::2] = torch.cos(pe)

        return pe_matrix.transpose(1,2).reshape(pe_matrix.shape[0], -1)  # pe_matrix: (..., input_dim*2L)
