import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_criterion = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def forward(self, batch):
        rays, target = batch['rays'], batch['target']
        output = self.net(rays)
        rgb_map = output['rgb_map']
        rgb_map_0 = output['rgb_map_0'] if 'rgb_map_0' in output.keys() else None
        scalar_stats = {}

        loss = self.color_criterion(rgb_map, target)
        psnr = -10. * torch.log(loss.detach()) / torch.log(torch.Tensor([10.]).to(loss.device))
        scalar_stats.update({'loss_fine': loss})
        scalar_stats.update({'psnr_fine': psnr})

        if rgb_map_0 is not None:
            loss_0 = self.color_criterion(rgb_map_0, target)
            psnr_0 = -10. * torch.log(loss_0.detach()) / torch.log(torch.Tensor([10.]).to(loss_0.device))

            scalar_stats.update({'loss_coarse': loss_0})
            scalar_stats.update({'psnr_coarse': psnr_0})

            loss += loss_0

        scalar_stats.update({'loss': loss})

        return loss, scalar_stats