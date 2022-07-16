import torch
import trimesh
import numpy as np
from easydict import EasyDict as edict

from models import base
from modules.sdf_generator import SDFGenerator


class Model(base.Model):
    """
        Model class to pretrain the SDFGenerator pre-Stage 1.
        For each 3D point, we predict its SDF w.r.t the surface of a unit sphere.
            We use a unit sphere because we know the SDF of every 3D point w.r.t the unit sphere analytically.
        The SDFGenerator module is then later tuned in Stage 1 and Stage 2 (as mentioned in models/reconstruction.py)
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.sdf_generator = SDFGenerator(opt)

    def get_impl_estimates(self, var, pts_obj_space, get_feat=False, get_rgb=False, get_deformations=False):
        if self.opt.is_stage_two_active:
            pts_deformations, pts_features, obj_feat = var.deform_impl_func(self.opt, pts_obj_space, get_feat=True)
            pts_canonical_space = pts_deformations + pts_obj_space
            sdf = var.sdf_impl_func(self.opt, pts_canonical_space, pts_features, get_feat=False)
            rgb = var.deform_impl_func.rgb(obj_feat).tanh_()
        else:
            sdf, obj_feat = var.sdf_impl_func(self.opt, pts_obj_space, get_feat=True)
            rgb = var.sdf_impl_func.rgb(obj_feat).tanh_()

        if self.opt.is_stage_two_active and get_deformations:
            return (sdf, pts_deformations)

        if get_rgb:
            return (sdf, rgb)

        return (sdf, obj_feat) if get_feat else sdf

    def forward(self, opt, var, training=False):
        var.latent = torch.randn(opt.batch_size, opt.latent_dim, device=opt.device)*opt.latent_std
        var.sdf_impl_func = self.sdf_generator.forward(opt, var.latent).to(opt.device)
        var.impl_func = self.get_impl_estimates

        var.dpc = {}
        var.idx = [1]
        sphere = trimesh.primitives.Sphere(radius=opt.impl.pretrain_radius, center=np.asarray([0., 0., 0.]))
        var.dpc.points = torch.from_numpy(np.asarray(sphere.sample(count=10000))).to(opt.device).float().unsqueeze(0)

        return var

    def compute_loss(self, opt, var, training=False):
        loss = edict()
        level, sdf_gt = self.get_sphere_sdf_GT(opt, var)
        loss.sphere = self.MSE_loss(level, sdf_gt)
        return loss

    def get_sphere_sdf_GT(self, opt, var, N=10000):
        lower, upper = opt.impl.sdf_range
        points_3D = torch.rand(opt.batch_size, N, 3, device=opt.device)
        points_3D = points_3D*(upper-lower)+lower
        level = var.impl_func(var, points_3D)
        sdf_gt = points_3D.norm(dim=-1, keepdim=True)-opt.impl.pretrain_radius
        return level, sdf_gt
