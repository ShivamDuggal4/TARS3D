import torchvision
import torch
import numpy as np
from models import base
from modules.renderer import Renderer
from modules.sdf_generator import SDFGenerator


class Model(base.Model):
    """
        Model class to dump the mean shape post-stage 1.
        The mean shape is encoded in a latent embedding (used by the SDFGenerator hyper-network).
        This latent code is then optimized as canonical latent code (see Figure 2 right side of the paper) in Stage 2.
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
            self.meannet_latent = torch.nn.Parameter(torch.from_numpy(np.load(opt.mean_latent)).float())
            self.deformnet = DeformNet(opt)

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
        return var
