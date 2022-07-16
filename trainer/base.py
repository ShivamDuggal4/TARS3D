import os
import time
import tqdm
import visdom
import importlib
import torch
from easydict import EasyDict as edict

from misc import utils
from misc.utils import log


class Trainer():

    def __init__(self, opt):
        super().__init__()
        os.makedirs(opt.output_path, exist_ok=True)
        self.optimizer = getattr(torch.optim, opt.optim.algo)
        if opt.optim.sched:
            self.scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)

    def load_dataset(self, opt, eval_split="val"):
        data = importlib.import_module("dataloader.{}".format(opt.data.dataset))
        log.info("loading training data...")
        self.train_data = data.Dataset(opt, split="train", subset=opt.data.train_sub)
        self.train_loader = self.train_data.setup_loader(opt, shuffle=True)

        log.info("loading test data...")
        self.test_data = data.Dataset(opt, split=eval_split, subset=opt.data.val_sub)
        self.test_loader = self.test_data.setup_loader(opt, shuffle=False, drop_last=False)

    def build_networks(self, opt):
        model = importlib.import_module("models.{}".format(opt.model))
        log.info("building networks...")
        self.model = model.Model(opt).to(opt.device)

    def setup_optimizer(self, opt):
        log.info("setting up optimizers...")
        optim_list = [dict(params=self.model.parameters(), lr=opt.optim.lr), ]
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad, "check 1")
        self.optim = self.optimizer(optim_list)
        if opt.optim.sched:
            self.setup_optimizer_scheduler(opt)

    def setup_optimizer_scheduler(self, opt):
        kwargs = {k: v for k, v in opt.optim.sched.items() if k != "type"}
        self.sched = self.scheduler(self.optim, **kwargs)

    def restore_checkpoint(self, opt):
        epoch_start, iter_start = None, None
        if opt.resume:
            log.info("resuming from previous checkpoint...")
            # in stage two training, opt.mean_latent should always be true and can be set to anything when --resume is True.
            if opt.mean_latent is not None:
                epoch_start, iter_start, meannet_latent = \
                    utils.restore_checkpoint(opt, self, resume=opt.resume, return_meannet_latent=True)
                self.model.meannet_latent = torch.nn.Parameter(meannet_latent)
            else:
                epoch_start, iter_start = utils.restore_checkpoint(opt, self, resume=opt.resume)

        elif opt.load is not None:
            log.info("loading weights from checkpoint {}...".format(opt.load))
            if opt.evaluation:
                log.info("loading canonical latent code from checkpoint {}...".format(opt.load))
                epoch_start, iter_start, meannet_latent = \
                    utils.restore_checkpoint(opt, self, load_name=opt.load, return_meannet_latent=True)
                self.model.meannet_latent = torch.nn.Parameter(meannet_latent)

            else:
                epoch_start, iter_start = utils.restore_checkpoint(opt, self, load_name=opt.load)
                if opt.load_deformnet is not None:
                    log.info("loading deformnet weights from checkpoint {}...".format(opt.load_deformnet))
                    _, _ = utils.restore_checkpoint(opt, self, load_name=opt.load_deformnet)
        else:
            log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def setup_visualizer(self, opt):
        log.info("setting up visualizers...")
        if opt.tb:
            self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path, flush_secs=10)
        if opt.visdom:
            # check if visdom server is runninng
            is_open = utils.check_socket_open(opt.visdom.server, opt.visdom.port)
            retry = None
            while not is_open:
                retry = input("visdom port ({}) not open, retry? (y/n) ".format(opt.visdom.port))
                if retry not in ["y", "n"]:
                    continue
                if retry == "y":
                    is_open = utils.check_socket_open(opt.visdom.server, opt.visdom.port)
                else:
                    break
            self.vis = visdom.Visdom(server=opt.visdom.server, port=opt.visdom.port, env=opt.group)

    def train(self, opt):
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.iter_skip = self.iter_start % len(self.train_loader)
        self.it = self.iter_start

        for self.ep in range(self.epoch_start, opt.max_epoch):
            self.train_epoch(opt)

        if (self.ep+1) % opt.freq.ckpt != 0:
            self.save_checkpoint(opt, ep=self.ep+1, it=self.it)
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom:
            self.vis.close()
        log.title("TRAINING DONE")

    def train_epoch(self, opt):
        self.model.train()
        loader = tqdm.tqdm(self.train_loader, desc="training epoch {}".format(self.ep+1), leave=False)

        for batch in loader:
            if self.iter_skip > 0:
                loader.set_description("(fast-forwarding...)")
                self.iter_skip -= 1
                if self.iter_skip == 0:
                    loader.set_description("training epoch {}".format(self.ep+1))
                continue
            var = edict(batch)
            var = utils.move_to_device(var, opt.device)
            loss = self.train_iteration(opt, var, loader)

        lr = self.sched.get_last_lr()[0] if opt.optim.sched else opt.optim.lr
        log.loss_train(opt, self.ep+1, lr, loss, self.timer)
        if opt.optim.sched:
            self.sched.step()
        if (self.ep+1) % opt.freq.eval == 0:
            self.evaluate(opt, ep=self.ep+1, training=True)
        if (self.ep+1) % opt.freq.ckpt == 0:
            self.save_checkpoint(opt, ep=self.ep+1, it=self.it)

    def train_iteration(self, opt, var, loader):
        self.timer.it_start = time.time()

        self.optim.zero_grad()
        var = self.model.forward(opt, var, training=True)
        loss = self.model.compute_loss(opt, var, training=True)
        loss = self.summarize_loss(opt, var, loss)
        loss.all.backward()
        self.optim.step()

        if (self.it+1) % opt.freq.scalar == 0:
            self.log_scalars(opt, var, loss, step=self.it+1, split="train")
        if (self.it+1) % opt.freq.vis == 0:
            self.visualize(opt, var, step=self.it+1, split="train")
        if (self.it+1) % opt.freq.ckpt_latest == 0:
            self.save_checkpoint(opt, ep=self.ep, it=self.it+1,  latest=True)

        self.it += 1
        loader.set_postfix(it=self.it, loss="{:.3f}".format(loss.all))
        self.timer.it_end = time.time()
        utils.update_timer(opt, self.timer, self.ep, len(loader))
        return loss

    def summarize_loss(self, opt, var, loss):
        loss_all = 0.
        assert("all" not in loss)

        for key in loss:
            assert(key in opt.loss_weight)
            assert(loss[key].shape == ())
            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key]), "loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]), "loss {} is NaN".format(key)
                loss_all += float(opt.loss_weight[key])*loss[key]

        loss.update(all=loss_all)
        return loss

    @torch.no_grad()
    def evaluate(self, opt, ep=None, training=False):
        self.model.eval()
        loss_eval = edict()
        loader = tqdm.tqdm(self.test_loader, desc="evaluating", leave=False)

        for it, batch in enumerate(loader):
            var = edict(batch)
            var = utils.move_to_device(var, opt.device)
            var = self.model.forward(opt, var, training=False)
            loss = self.model.compute_loss(opt, var, training=False)
            loss = self.summarize_loss(opt, var, loss)
            for key in loss:
                loss_eval.setdefault(key, 0.)
                loss_eval[key] += loss[key]*len(var.idx)
            loader.set_postfix(loss="{:.3f}".format(loss.all))
            if it == 0 and training:
                self.visualize(opt, var, step=ep, split="eval")

        for key in loss_eval:
            loss_eval[key] /= len(self.test_data)
        log.loss_eval(opt, loss_eval)

        if training:
            self.log_scalars(opt, var, loss_eval, step=ep, split="eval")
        else:
            self.dump_results(opt, var)

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        for key, value in loss.items():
            if key == "all":
                continue
            if opt.loss_weight[key] is not None:
                self.tb.add_scalar("{0}/loss_{1}".format(split, key), value, step)
        if metric is not None:
            for key, value in metric.items():
                self.tb.add_scalar("{0}/{1}".format(split, key), value, step)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"):
        raise NotImplementedError

    def save_checkpoint(self, opt, ep=0, it=0, latest=False):
        utils.save_checkpoint(opt, self, ep=ep, it=it, latest=latest)
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group, opt.name, ep, it))
