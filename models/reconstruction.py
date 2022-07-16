import torchvision
import torch
import torch.nn.functional as torch_F
import numpy as np
from easydict import EasyDict as edict

from misc.utils import log
from misc import camera
from modules.renderer import Renderer
from modules.sdf_generator import SDFGenerator
from models import base
from misc import utils


class Model(base.Model):
    """
       Model class for different modules in Stage 1 and Stage 2.
       In Stage 1, we train the image encoder, renderer and the SDF Generator.
       In Stage 2, we tune the image encoder, renderer, SDFGenerator, DeformNet and the canonical latent code.
    """
    def __init__(self, opt):
        super().__init__(opt)
        network = getattr(torchvision.models, opt.arch.enc_network)
        self.opt = opt
        self.encoder = network(pretrained=opt.arch.enc_pretrained)
        self.encoder.fc = torch.nn.Linear(self.encoder.fc.in_features, opt.latent_dim)

        self.sdf_generator = SDFGenerator(opt)
        self.renderer = Renderer(opt)

        if opt.is_stage_two_active:
            from modules.deformnet import DeformNet
            if opt.mean_latent is not None and opt.evaluation is None and opt.resume is None:
                self.meannet_latent = torch.nn.Parameter(torch.from_numpy(np.load(opt.mean_latent)).float())
                log.info('canonical latent code loaded from: {}'.format(opt.mean_latent))
            self.deformnet = DeformNet(opt)

    def get_impl_estimates(self, var, pts_obj_space, get_feat=False, get_rgb=False, get_deformations=False):
        if self.opt.is_stage_two_active:
            pts_deformations, pts_features, obj_feat = var.deform_impl_func(self.opt, pts_obj_space, get_feat=True)
            pts_canonical_space = pts_deformations + pts_obj_space
            # during stage 2 training, we turn off point features for half of the batch to ensure
            # that both deformations and point features are learned well!
            if self.opt.evaluation:
                sdf = var.sdf_impl_func(self.opt, pts_canonical_space, pts_features, get_feat=False)
            else:
                sdf = var.sdf_impl_func(
                    self.opt, pts_canonical_space, pts_features, turn_off_half_point_features=True, get_feat=False)
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
        batch_size = len(var.idx)
        image_latent = self.encoder(var.rgb_input_map)

        if opt.is_stage_two_active:
            shape_reconstruction_latent_enc = self.meannet_latent.unsqueeze(0).repeat(var.rgb_input_map.shape[0], 1)
            deformnet_latent_enc = image_latent
            var.deform_impl_func = self.deformnet(opt, deformnet_latent_enc)
        else:
            shape_reconstruction_latent_enc = image_latent

        var.object_latent = image_latent
        var.sdf_impl_func = self.sdf_generator.forward(opt, shape_reconstruction_latent_enc).to(opt.device)
        var.impl_func = self.get_impl_estimates

        if opt.impl.rand_sample and training:
            var.rgb_recon, var.depth, var.level, var.mask, var.level_all = self.renderer.forward(
                opt, var, var.pose, intr=var.intr, ray_idx=var.ray_idx)  # [B,HW,3]
        else:
            var.rgb_recon, var.depth, var.level, var.mask, var.level_all = self.renderer.forward(
                opt, var, var.pose, intr=var.intr)  # [B,HW,3]
            var.rgb_recon_map = var.rgb_recon.view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
            var.depth_map = var.depth.view(batch_size, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
            var.level_map = var.level.view(batch_size, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
            var.mask_map = var.mask.view(batch_size, opt.H, opt.W, 1).permute(0, 3, 1, 2)  # [B,1,H,W]
        return var

    def compute_loss(self, opt, var, training=False):
        loss = edict()
        if opt.loss_weight.render is not None:
            loss.render = self.MSE_loss(var.rgb_recon, var.rgb_input)
        if opt.loss_weight.shape_silh is not None:
            loss.shape_silh = self.shape_from_silhouette_loss(opt, var)
        if opt.loss_weight.ray_intsc is not None:
            loss.ray_intsc = self.ray_intersection_loss(opt, var)
        if opt.loss_weight.ray_free is not None:
            loss.ray_free = self.ray_freespace_loss(opt, var)
        if opt.loss_weight.eikonal is not None:
            if opt.is_stage_two_active and opt.loss_weight.pts_deformations is not None:
                # we never used symmetry loss, except for Pix3D training (since dataset is too small and challenging)
                if opt.loss_weight.symmetry_loss is not None:
                    var.grad_sdf_norm, var.grad_deform_norm, symmetry_loss = self.sdf_gradient_norm(
                        opt, var, batch_size=len(var.idx), return_deformations=True)
                    loss.symmetry_loss = symmetry_loss
                else:
                    var.grad_sdf_norm, var.grad_deform_norm = self.sdf_gradient_norm(
                        opt, var, batch_size=len(var.idx), return_deformations=True)
                loss.pts_deformations = self.MSE_loss(var.grad_deform_norm, 0)
                
            else:
                if opt.loss_weight.symmetry_loss is not None:
                    var.grad_sdf_norm, symmetry_loss = self.sdf_gradient_norm(
                        opt, var, batch_size=len(var.idx))
                    loss.symmetry_loss = symmetry_loss
                else:
                    var.grad_sdf_norm = self.sdf_gradient_norm(
                        opt, var, batch_size=len(var.idx))
            loss.eikonal = self.MSE_loss(var.grad_sdf_norm, 1)
        return loss

    def ray_intersection_loss(self, opt, var, level_eps=0.01):
        level_in = var.level_all[..., -1:]  # [B,HW,1]
        weight = 1/(var.dt_input+1e-8) if opt.impl.importance else None
        loss = self.L1_loss((level_in+level_eps).relu_(), weight=weight, mask=var.mask_input.bool()) \
            + self.L1_loss((-level_in+level_eps).relu_(), weight=weight, mask=~var.mask_input.bool())
        return loss

    def ray_freespace_loss(self, opt, var, level_eps=0.01):
        level_out = var.level_all[..., :-1]  # [B,HW,N-1]
        loss = self.L1_loss((-level_out+level_eps).relu_())
        return loss

    def shape_from_silhouette_loss(self, opt, var):  # [B,N,H,W]
        batch_size = len(var.idx)
        mask_bg = var.mask_input.long() == 0
        weight = 1/(var.dt_input+1e-8) if opt.impl.importance else None
        # randomly sample depth along ray
        depth_min, depth_max = opt.impl.depth_range
        num_rays = var.ray_idx.shape[1] if "ray_idx" in var else opt.H*opt.W
        depth_samples = torch.rand(batch_size, num_rays, opt.impl.sdf_samples,
                                   1, device=opt.device)*(depth_max-depth_min) + depth_min  # [B,HW,N,1]
        center, ray = camera.get_center_and_ray(opt, var.pose, intr=var.intr)
        if "ray_idx" in var:
            gather_idx = var.ray_idx[..., None].repeat(1, 1, 3)
            ray = ray.gather(dim=1, index=gather_idx)
            if opt.camera.model == "orthographic":
                center = center.gather(dim=1, index=gather_idx)
        points_3D_samples = camera.get_3D_points_from_depth(
            opt, center, ray, depth_samples, multi_samples=True)  # [B,HW,N,3]
        level_samples = var.impl_func(
            var, points_3D_samples)[..., 0]  # [B,HW,N]

        # compute lower bound
        if opt.camera.model == "perspective":
            _, grid_3D = camera.get_camera_grid(opt, batch_size, intr=var.intr)  # [B,HW,3]
            offset = torch_F.normalize(grid_3D[..., :2], dim=-1)*var.dt_input_map.view(batch_size, -1, 1)  # [B,HW,2]
            _, ray0 = camera.get_center_and_ray(opt, var.pose, intr=var.intr, offset=offset)  # [B,HW,3]
            if "ray_idx" in var:
                gather_idx = var.ray_idx[..., None].repeat(1, 1, 3)
                ray0 = ray0.gather(dim=1, index=gather_idx)
            ortho_dist = \
                (ray - ray0*(ray*ray0).sum(dim=-1, keepdim=True) / (ray0*ray0).sum(dim=-1, keepdim=True)).norm(
                    dim=-1, keepdim=True)  # [B,HW,1]
            min_dist = depth_samples[..., 0] * ortho_dist  # [B,HW,N]

        elif opt.camera.model == "orthographic":
            min_dist = var.dt_input

        loss = self.L1_loss((min_dist-level_samples).relu_(), weight=weight, mask=mask_bg)
        return loss

    def sdf_gradient_norm(self, opt, var, batch_size, return_deformations=False, N=10000):
        lower, upper = opt.impl.sdf_range
        points_3D = torch.rand(batch_size, N, 3, device=opt.device)
        points_3D = points_3D*(upper-lower)+lower

        if opt.loss_weight.symmetry_loss is not None:
            points_3D_mirror = points_3D.clone()
            points_3D_mirror[:,:,0] *= -1
            points_3D = torch.cat([points_3D, points_3D_mirror], dim=1)
        
        with torch.enable_grad():
            points_3D.requires_grad_(True)
            if not return_deformations:
                pts_sdf = var.impl_func(var, points_3D)
                grad_sdf = torch.autograd.grad(pts_sdf.sum(), points_3D, create_graph=True)
                grad_sdf_norm = grad_sdf[0].norm(dim=-1, keepdim=True)
                if opt.loss_weight.symmetry_loss is not None:
                    symmetry_loss = self.L1_loss(pts_sdf[:,:N], pts_sdf[:,N:])
                    return grad_sdf_norm, symmetry_loss
                return grad_sdf_norm

            pts_sdf, pts_deformations = var.impl_func(var, points_3D, get_deformations=True)
            if opt.loss_weight.symmetry_loss is not None:
                symmetry_loss = self.L1_loss(pts_sdf[:,:N], pts_sdf[:,N:])
            grad_sdf = torch.autograd.grad(pts_sdf.sum(), points_3D, create_graph=True)
            grad_sdf_norm = grad_sdf[0].norm(dim=-1, keepdim=True)

            u = pts_deformations[:, :, 0]
            v = pts_deformations[:, :, 1]
            w = pts_deformations[:, :, 2]

            grad_outputs = torch.ones_like(u)
            grad_u = torch.autograd.grad(u, [points_3D], grad_outputs=grad_outputs, create_graph=True)[0]
            grad_v = torch.autograd.grad(v, [points_3D], grad_outputs=grad_outputs, create_graph=True)[0]
            grad_w = torch.autograd.grad(w, [points_3D], grad_outputs=grad_outputs, create_graph=True)[0]
            grad_deform = torch.stack([grad_u, grad_v, grad_w], dim=2)  # gradient of deformation wrt. input position

        grad_deform_norm = grad_deform.norm(dim=-1)
        if opt.loss_weight.symmetry_loss is not None:
            return grad_sdf_norm, grad_deform_norm, symmetry_loss
        return grad_sdf_norm, grad_deform_norm
