import torch
import torch.nn.functional as torch_F
import trimesh
import numpy as np
from easydict import EasyDict as edict

from misc import utils
from misc.utils import log
from models import base
from modules.renderer import Renderer
from modules.sdf_generator import SDFGenerator
from modules.deformnet import DeformNet


class Model(base.Model):
    """
        Model class to pretrain the DefomNet module post-stage 1.
        Using the learned SDFGenerator and the canonical shape embedding (dumped canonical shape) from stage 1,
            we try to map a unit sphere to the learned canonical shape.
            We use a unit sphere because we know the SDF of every 3D point w.r.t the unit sphere analytically.
        The Deformnet module is then later tuned in Stage 2 (as mentioned in models/reconstruction.py)
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.meannet_latent = torch.nn.Parameter(torch.from_numpy(np.load(opt.mean_latent)).float())
        log.info('canonical latent code loaded from: {}'.format(opt.mean_latent))
        self.deformnet = DeformNet(opt)
        self.sdf_generator = SDFGenerator(opt)
        self.renderer = Renderer(opt)

        utils.flip_gradients(self.sdf_generator, switch=False)
        utils.flip_gradients(self.renderer, switch=False)
        self.meannet_latent.requires_grad = False
        

    def get_impl_estimates(self, var, pts_obj_space):
        pts_deformations, pts_features = var.deform_impl_func(self.opt, pts_obj_space, get_feat=False)
        pts_canonical_space = pts_deformations + pts_obj_space
        sdf = var.sdf_impl_func(self.opt, pts_canonical_space, get_feat=False)
        return sdf

    def forward(self, opt, var, training=False):
        batch_size = opt.batch_size
        shape_reconstruction_latent_enc = self.meannet_latent.unsqueeze(0).repeat(batch_size, 1)
        deformnet_latent_enc = torch.randn(opt.batch_size, opt.latent_dim, device=opt.device) * opt.latent_std
        var.deform_impl_func = self.deformnet(opt, deformnet_latent_enc)
        var.sdf_impl_func = self.sdf_generator.forward(opt, shape_reconstruction_latent_enc).to(opt.device)
        var.impl_func = self.get_impl_estimates
        

        var.dpc = {}
        var.idx = [1]
        sphere = trimesh.primitives.Sphere(radius=0.5, center=np.asarray([0., 0., 0.]))
        var.dpc.points = torch.from_numpy(
            np.asarray(sphere.sample(count=10000))).to(self.opt.device).float().unsqueeze(0)

        return var

    def compute_loss(self, opt, var, epoch_num=None, iter_num=None, training=False):
        loss = edict()
        sdf_pred, sdf_gt, normal_pred, normal_gt = self.get_sphere_sdf_GT(opt, var)
        loss.sphere = self.MSE_loss(sdf_pred, sdf_gt)
        loss.sphere_normal = self.L1_loss(
            torch_F.cosine_similarity(normal_pred, normal_gt, dim=-1)[..., None], 1., mask=(sdf_gt == 0.))
        return loss

    def get_sphere_sdf_GT(self, opt, var, N=10000):
        lower, upper = opt.impl.sdf_range
        points_3D = torch.rand(opt.batch_size, N, 3, device=opt.device)
        points_3D = points_3D*(upper-lower)+lower
        sdf_gt = points_3D.norm(dim=-1, keepdim=True)-opt.impl.pretrain_radius
        normal_gt = points_3D / points_3D.norm(dim=-1, keepdim=True)

        with torch.enable_grad():
            points_3D.requires_grad_(True)
            sdf_pred = var.impl_func(var, points_3D)

        normal_pred = torch.autograd.grad(sdf_pred.sum(), points_3D, create_graph=True)[0]
        normal_pred = normal_pred / normal_pred.norm(dim=-1, keepdim=True)

        return sdf_pred, sdf_gt, normal_pred, normal_gt
