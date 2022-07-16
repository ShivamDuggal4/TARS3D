import torch
from easydict import EasyDict as edict

from misc import utils
from modules.base import HyperNet, ImplicitFunction


class SDFGenerator(HyperNet):
    """
        HyperNet module for learning signed distance field (SDF).
        Used in object space in stage 1, and in canonical space in stage 2.
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.point_feat_predictor = torch.nn.Sequential(
            torch.nn.Linear(33, 32),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(32, opt.pts_feat_dim)
        )
        print('point feat network crated only once')

    def define_network(self, opt):
        point_dim = 3 + (opt.impl.posenc_L*6 if opt.impl.posenc_L else 0) + \
            opt.pts_feat_dim * opt.pts_feat_pos_enc_dim * 2
        feat_dim = opt.arch.sdf_generator.layers_impl[-1]
        hypernet_layers = utils.get_layer_dims(opt.arch.sdf_generator.layers_hyper)
        self.hyper_impl = self.get_module_params(
            opt, opt.arch.sdf_generator.layers_impl, hypernet_layers, k0=point_dim,
            interm_coord=opt.arch.sdf_generator.interm_coord)
        self.hyper_level = self.get_module_params(
            opt, opt.arch.sdf_generator.layers_level, hypernet_layers, k0=feat_dim)
        self.hyper_rgb = self.get_module_params(opt, opt.arch.sdf_generator.layers_rgb, hypernet_layers, k0=feat_dim)

    def forward(self, opt, latent):
        point_dim = 3 + (opt.impl.posenc_L*6 if opt.impl.posenc_L else 0) + opt.pts_feat_dim*opt.pts_feat_pos_enc_dim*2
        feat_dim = opt.arch.sdf_generator.layers_impl[-1]
        impl_layers = edict()
        impl_layers.impl = self.hyperlayer_forward(
            opt, latent, self.hyper_impl, opt.arch.sdf_generator.layers_impl, k0=point_dim,
            interm_coord=opt.arch.sdf_generator.interm_coord)
        impl_layers.level = self.hyperlayer_forward(
            opt, latent, self.hyper_level, opt.arch.sdf_generator.layers_level, k0=feat_dim)
        impl_layers.rgb = self.hyperlayer_forward(
            opt, latent, self.hyper_rgb, opt.arch.sdf_generator.layers_rgb, k0=feat_dim)
        impl_layers.point_feat_predictor = self.point_feat_predictor
        impl_func = SDFImplicitFunction(opt, impl_layers)
        return impl_func


class SDFImplicitFunction(ImplicitFunction):
    """
        Implicit Function for predicting signed-distance field (SDF).
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
        self.level = self.define_heads(opt, impl_layers.level)
        self.rgb = self.define_heads(opt, impl_layers.rgb)
        self.point_feat_predictor = impl_layers.point_feat_predictor

    def forward(self, opt, points_3D, point_feat=None, turn_off_half_point_features=False, get_feat=False):
        if opt.impl.posenc_L:
            points_enc = self.positional_encoding(opt, points_3D)  # [B,...,6L]
            points_enc = torch.cat([points_enc, points_3D], dim=-1)  # [B,...,6L+3]
        else:
            points_enc = points_3D

        if point_feat is None:
            point_feat = self.point_feat_predictor(points_enc)

        # if point_feat is not None and turn_off_point_features:
        #     point_feat[:point_feat.shape[0]//2] = point_feat[:point_feat.shape[0]//2] * 0
        if point_feat is not None and turn_off_half_point_features:
            point_feat[:point_feat.shape[0]//2] = self.point_feat_predictor(points_enc)[:point_feat.shape[0]//2]

        point_feat = self.positional_encoding(opt, point_feat, posenc_L=opt.pts_feat_pos_enc_dim)
        feat = torch.cat([point_feat, points_enc], dim=-1)
        for li, layer in enumerate(self.impl):
            if opt.arch.sdf_generator.interm_coord and li > 0:
                feat = torch.cat([feat, points_enc], dim=-1)
            feat = layer(feat)
        level = self.level(feat)
        return (level, feat) if get_feat else level
