import torch
import os
import tqdm
import numpy as np
from easydict import EasyDict as edict

from misc import utils
from misc.utils import log
from trainer import base


class Trainer(base.Trainer):

    def __init__(self, opt):
        super().__init__(opt)

    @torch.no_grad()
    def evaluate(self, opt, ep=None, training=False):
        self.model.eval()
        loader = tqdm.tqdm(self.test_loader, desc="evaluating", leave=False)
        avg_latent = np.zeros(opt.latent_dim).astype('float')
        for it, batch in enumerate(loader):
            var = edict(batch)
            var = self.evaluate_batch(opt, var, ep, it)
            avg_latent = avg_latent + var.object_latent.sum(dim=0).data.cpu().numpy()

        avg_latent = avg_latent / len(self.test_data)
        self.dump_results(opt, avg_latent)
        log.info('Canonical latent dumped! Next Step --> Pretrain Deformnet')

    def evaluate_batch(self, opt, var, ep=None, it=None):
        var = utils.move_to_device(var, opt.device)
        var = self.model.forward(opt, var, training=True)
        return var

    def dump_results(self, opt, latent):
        os.makedirs("{}/dump/".format(opt.output_path), exist_ok=True)
        fname = "{}/dump/{}.npy".format(opt.output_path, 'mean_shape_latent')
        np.save(fname, latent)
