import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from lib.config import cfg, args
from lib.utils.base_utils import load_object
from os.path import join
import wandb
import numpy as np
import json
from argparse import ArgumentParser


class plwrapper(pl.LightningModule):
    def __init__(self, cfg, mode='train'):
        # load_object <=> import
        super().__init__()
        self.cfg = cfg
        self.network = load_object(cfg.network_module, {})
        if mode == 'train':
            self.train_dataset = load_object(cfg.train_dataset_module, cfg.train_dataset)
            self.val_dataset = load_object(cfg.val_dataset_module, cfg.val_dataset)
        self.network_wrapper = load_object(cfg.loss_module, {'net': self.network})  # loss module
        self.evaluator = load_object(cfg.evaluator_module, {'net': self.network})
        #self.visualizer = load_object(cfg.visualizer_module, {})
        self.vals = []
        self.predictions = []

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        #__import__('ipdb').set_trace()
        self.network.train()
        #batch['step'] = self.trainer.global_step
        #batch['meta']['step'] = self.trainer.global_step
        output = self.evaluator(batch)
        #self.visualizer(output, batch)
        
        return output

    def training_step(self, batch, batch_idx):
        #__import__('ipdb').set_trace()
        batch['step'] = self.trainer.global_step
        #print(batch['step'])
        #batch['meta']['step'] = self.trainer.global_step
        # training_step defines the train loop. It is independent of forward
        loss, loss_stats = self.network_wrapper(batch)
        # if self.trainer.global_step % cfg.log_interval == 0:
        #     self.logger.log_metrics(loss_stats, step=batch['step'])
        for key, val in loss_stats.items():
            self.log(key, val, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        #__import__('ipdb').set_trace()
        batch['step'] = self.trainer.global_step
        #batch['meta']['step'] = self.trainer.global_step
        loss, loss_stats = self.network_wrapper(batch)
        for key in list(loss_stats.keys()):
            loss_stats['val_'+key] = loss_stats.pop(key)
        for key, val in loss_stats.items():
            self.log(key, val, sync_dist=True)
        #self.vals.append(loss_stats)
        #self.logger.log_metrics(loss_stats, step=batch['step'], )

        return loss
    
    # def on_validation_epoch_end(self) -> None:
    #     loss_stats = {}
    #     for stat_dict in self.vals:
    #         for k in stat_dict.keys():
    #             if k not in loss_stats:
    #                 loss_stats[k] = stat_dict[k]
    #             else:
    #                 loss_stats[k] += stat_dict[k]
    #     loss_stats = {k:v/len(self.vals) for k,v in loss_stats.items()}
    #     self.logger.log_metrics(loss_stats, step=self.trainer.global_step)

    #     return super().on_validation_epoch_end()
    
    def predict_step(self, batch, batch_idx):
        #__import__('ipdb').set_trace()
        pred = self(batch)
        self.predictions.append(pred)

        return pred
    
    def on_predict_end(self) -> None:
        return {'psnr': np.mean(self.predictions)}

    def train_dataloader(self):
        from lib.datasets.make_dataset import make_data_sampler, make_batch_data_sampler, make_collator, worker_init_fn
        from torch.utils.data import DataLoader
        batch_size = cfg.train.batch_size
        shuffle = cfg.train.shuffle
        drop_last = False

        sampler = make_data_sampler(self.train_dataset, shuffle, cfg.distributed)
        batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter=cfg.ep_iter, is_train=True)

        num_workers = cfg.train.num_workers
        collator = make_collator(cfg, True)

        data_loader = DataLoader(self.train_dataset,
                            batch_sampler=batch_sampler,
                            num_workers=num_workers,
                            collate_fn=collator,
                            worker_init_fn=worker_init_fn)
        return data_loader

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    def configure_optimizers(self):
        from lib.train.optimizer import make_optimizer
        from lib.train.scheduler import make_lr_scheduler, set_lr_scheduler
        optimizer = make_optimizer(cfg, self.network)
        scheduler = make_lr_scheduler(cfg, optimizer)
        return [optimizer], [scheduler]


def train(cfg):
    model = plwrapper(cfg)

    if cfg.resume and os.path.exists(join(cfg.trained_model_dir, 'last.ckpt')):
        resume_from_checkpoint = join(cfg.trained_model_dir, 'last.ckpt')
        #resume_from_checkpoint = None
    else:
        resume_from_checkpoint = None
        if os.path.exists(cfg.record_dir):
            pass
        os.makedirs(cfg.record_dir, exist_ok=True)
        print(cfg, file=open(join(cfg.record_dir, 'exp.yml'), 'w'))

    #logger = TensorBoardLogger(save_dir=cfg.record_dir, name=cfg.exp_name)
    logger = WandbLogger(project=cfg.project_name, save_dir=cfg.record_dir, name=cfg.exp_name)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        dirpath=cfg.trained_model_dir,
        every_n_epochs=20,
        save_last=True,
        save_top_k=-1,
        monitor='loss',
        filename="{epoch}")
    
    # Log true learning rate, serves as LR-Scheduler callback
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    extra_args = {
        # 'num_nodes': len(cfg.gpus),
        'accelerator': 'gpu',
    }

    if len(cfg.gpus) > 0:
        extra_args['strategy'] = 'ddp'
        extra_args['use_distributed_sampler'] = False
        #extra_args['replace_sampler_ddp'] = False

    trainer = pl.Trainer(
        #gpus=len(cfg.gpus),
        num_nodes=len(cfg.gpus),
        logger=logger,
        #resume_from_checkpoint=resume_from_checkpoint,
        callbacks=[ckpt_callback, lr_monitor],
        max_epochs=cfg.train.epoch,
        check_val_every_n_epoch=cfg.eval_ep,
        log_every_n_steps=cfg.log_interval,
        # profiler='simple',
        **extra_args
    )
    trainer.fit(model, ckpt_path=resume_from_checkpoint)

def load_ckpt(model, ckpt_path, model_name='network'):
    print('Load from {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    checkpoint_ = {}
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in []:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    model.load_state_dict(checkpoint_, strict=False)
    return epoch


def test(cfg):
    from glob import glob
    from os.path import join
    from tqdm import tqdm
    if False:
        network = load_object(cfg.network_module, cfg.network_args)
        ckptpath = sorted(glob(join(cfg.trained_model_dir, '*ckpt*')))[-1]
        epoch = load_ckpt(network, ckptpath)
        # cfg.visualizer_args.out += '_{}'.format(epoch)
        # device = torch.device('cuda')
        # network = network.to(device)
        # network.train()
        renderer = load_object(cfg.renderer_module, cfg.renderer_args, net=network)
    model = plwrapper(cfg, mode=cfg.split)
    ckptpath = join(cfg.trained_model_dir, 'last.ckpt')
    #ckptpath = join('/home/zhuhe/codes/learning_nerf/trained_models/nerf/lego/learning_nerf_no_pi', 'last.ckpt')
    assert os.path.exists(ckptpath)
    epoch = load_ckpt(model.network, ckptpath)
    
    model.step = epoch * cfg.ep_iter
    if len(cfg.gpus) == 1:
        pass
        # for key, val in model.network.models.items():print(key, sum([v.numel() for v in val.parameters()]))
        # import ipdb;ipdb.set_trace()

    # model.visualizer.data_dir += '_{}'.format(epoch)

    if cfg.split == 'test' or cfg.split == 'eval':
        dataset = load_object(cfg.test_dataset_module, cfg.test_dataset)
    elif cfg.split in ['demo', 'canonical', 'novelposes']:
        dataset = load_object(cfg['data_{}_module'.format(cfg.split)], cfg['data_{}_args'.format(cfg.split)])
    elif cfg.split == 'trainvis':
        dataset = model.train_dataset
        dataset.sample_args.nrays *= 16

    # ranges = cfg.get('visranges', [0, -1, 1])
    # if ranges[1] == -1:
    #     ranges[1] = len(dataset)
    #sampler = FrameSampler(dataset, ranges)

    dataloader = torch.utils.data.DataLoader(dataset,
        # sampler=sampler,
        # batch_sampler=batch_sampler,
        batch_size=1, num_workers=cfg.test.num_workers)

    extra_args = {
        # 'num_nodes': len(cfg.gpus),
        'accelerator': 'gpu',
    }
    if len(cfg.gpus) > 0:
        extra_args['strategy'] = 'ddp'
        # extra_args['replace_sampler_ddp'] = False
    mode = 1
    if mode == 1:
        trainer = pl.Trainer(
            num_nodes=len(cfg.gpus),
            # resume_from_checkpoint=resume_from_checkpoint,
            # ckpt_path=resume_from_checkpoint,
            max_epochs=cfg.train.epoch,
            # profiler="simple",
            **extra_args
        )
        preds = trainer.predict(model, dataloader)
        json.dump(preds, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w'))
    elif mode == 2:
        device = torch.device('cuda')
        model = model.to(device)
        for batch in dataloader:
            for k in batch:
                if k != 'meta' and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            # print(batch['meta'])
            # if data['meta']['index'] < 150:
            #     print('[info] skip data {}'.format(data['meta']['index']))
            #     continue
            pred = model(batch)
            print(pred)
            
    # device = torch.device('cuda')
    # from easymocap.mytools import Timer
    # import numpy as np
    # for i in range(len(model.val_dataset)):
    #     with Timer('get data'):
    #         batch = model.val_dataset[i]
    #     with Timer('to gpu'):
    #         for k in batch:
    #             if k != 'meta' and isinstance(batch[k], np.ndarray):
    #                 batch[k] = torch.Tensor(batch[k]).to(device)
    #     with Timer('forward'):
    #         with torch.no_grad():
    #             model(batch)



if __name__ == "__main__":
    wandb.init(project=cfg.project_name, name=cfg.exp_name)

    if cfg.fix_random:
        seed_everything(666)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.test:
       test(cfg)
    else:
       train(cfg)
    #test(cfg)
