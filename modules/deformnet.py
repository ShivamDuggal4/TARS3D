import torch
from easydict import EasyDict as edict

from misc import utils
from modules.base import HyperNet, ImplicitFunction


class DeformNet(HyperNet):
    """
        HyperNet module for learning higher-dimensional deformations.
    """
    def __init__(self, opt):
        super().__init__(opt)

    def define_network(self, opt):
        point_dim = 3+(opt.impl.posenc_L*6 if opt.impl.posenc_L else 0)
        feat_dim = opt.arch.deformnet.layers_impl[-1]
        hypernet_layers = utils.get_layer_dims(opt.arch.deformnet.layers_hyper)
        self.hyper_impl = self.get_module_params(
            opt, opt.arch.deformnet.layers_impl, hypernet_layers, k0=point_dim,
            interm_coord=opt.arch.deformnet.interm_coord)
        self.hyper_pts_deformations = self.get_module_params(
            opt, opt.arch.deformnet.layers_pts_deformations, hypernet_layers, k0=feat_dim)
        self.hyper_pts_feat = self.get_module_params(
            opt, opt.arch.deformnet.layers_pts_feat, hypernet_layers, k0=feat_dim)
        self.hyper_rgb = self.get_module_params(opt, opt.arch.deformnet.layers_rgb, hypernet_layers, k0=feat_dim)

    def forward(self, opt, latent):
        point_dim = 3+(opt.impl.posenc_L*6 if opt.impl.posenc_L else 0)
        feat_dim = opt.arch.deformnet.layers_impl[-1]
        impl_layers = edict()
        impl_layers.impl = self.hyperlayer_forward(
            opt, latent, self.hyper_impl, opt.arch.deformnet.layers_impl, k0=point_dim,
            interm_coord=opt.arch.deformnet.interm_coord)
        impl_layers.pts_deformations = self.hyperlayer_forward(
            opt, latent, self.hyper_pts_deformations, opt.arch.deformnet.layers_pts_deformations, k0=feat_dim)
        impl_layers.pts_feat = self.hyperlayer_forward(
            opt, latent, self.hyper_pts_feat, opt.arch.deformnet.layers_pts_feat, k0=feat_dim)
        impl_layers.rgb = self.hyperlayer_forward(
            opt, latent, self.hyper_rgb, opt.arch.deformnet.layers_rgb, k0=feat_dim)
        impl_func = DeformNetImplicitFunction(opt, impl_layers)
        return impl_func


class DeformNetImplicitFunction(ImplicitFunction):
    """
        Implicit Function for predicting higher-dimensional deformations.
        Takes as input HyperNet predicted layers (in __init__ function).
    """
    def __init__(self, opt, impl_layers):
        super().__init__(opt, impl_layers)

    def define_network(self, opt, impl_layers):
        self.impl = torch.nn.ModuleList()
        for linear in impl_layers.impl:
            layer = torch.nn.Sequential(
                linear,
                torch.nn.LayerNorm(linear.bias.shape[-1], elementwise_affine=False),
                torch.nn.ReLU(inplace=False),
            )
            self.impl.append(layer)

        self.pts_deformations = self.define_heads(opt, impl_layers.pts_deformations)
        self.pts_feat = self.define_heads(opt, impl_layers.pts_feat)
        self.rgb = self.define_heads(opt, impl_layers.rgb)

    def forward(self, opt, points_3D, get_feat=False):  # [B,...,3]
        if opt.impl.posenc_L:
            points_enc = self.positional_encoding(opt, points_3D)  # [B,...,6L]
            points_enc = torch.cat([points_enc, points_3D], dim=-1)  # [B,...,6L+3]
        else:
            points_enc = points_3D
        feat = points_enc

        for li, layer in enumerate(self.impl):
            if opt.arch.deformnet.interm_coord and li > 0:
                feat = torch.cat([feat, points_enc], dim=-1)
            feat = layer(feat)
        pts_feat = self.pts_feat(feat)
        pts_deformations = self.pts_deformations(feat)
        return (pts_deformations, pts_feat, feat) if get_feat else (pts_deformations, pts_feat)
