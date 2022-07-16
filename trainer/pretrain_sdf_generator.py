import torch
import tqdm
import time
from easydict import EasyDict as edict

from trainer import base
from misc import utils, eval_3D
import misc.visualization_utils as vis_utils
from misc.utils import log


class Trainer(base.Trainer):

    def __init__(self, opt):
        super().__init__(opt)

    def load_dataset(self, opt): return

    def train(self, opt):
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.ep = 0
        self.it = self.iter_start

        self.model.train()
        loader = tqdm.trange(opt.max_iter, desc="training", leave=False)
        for it in loader:
            var = edict()
            _ = self.train_iteration(opt, var, loader)

        self.save_checkpoint(opt, ep=1, it=self.it)
        if opt.tb:
            self.tb.close()
        if opt.visdom:
            self.vis.close()
        log.title("TRAINING DONE")

    @torch.no_grad()
    def evaluate(self, opt, ep=None, training=False): return

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"):
        if opt.eval and opt.visdom:
            dist_acc, dist_cov = eval_3D.compute_chamfer_dist(opt, var)
            vis_utils.vis_pointcloud(opt, self.vis, step, split, pred=var.dpc_pred, GT=var.dpc.points)

    def save_checkpoint(self, opt, ep=0, it=0, latest=False):
        utils.save_checkpoint(opt, self, ep=ep, it=it, children=("sdf_generator",))
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group, opt.name, ep, it))
