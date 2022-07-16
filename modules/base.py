import numpy as np
import torch
import torch.utils.tensorboard
from misc import utils


class HyperNet(torch.nn.Module):
    """
        Base HyperNet function. Parent class for DeformNet and SDFGenerator
    """
    def __init__(self, opt):
        super().__init__()
        self.define_network(opt)

    def get_module_params(self, opt, layers, hypernet_L, k0, interm_coord=False):
        impl_params = torch.nn.ModuleList()
        L = utils.get_layer_dims(layers)
        for li, (k_in, k_out) in enumerate(L):
            if li == 0:
                k_in = k0
            if interm_coord and li > 0:
                k_in += 3+(opt.impl.posenc_L*6 if opt.impl.posenc_L else 0)
            params = self.define_hyperlayer(opt, hypernet_L, dim_in=k_in, dim_out=k_out)
            impl_params.append(params)
        return impl_params

    def define_hyperlayer(self, opt, L, dim_in, dim_out):
        hyperlayer = []
        for li, (k_in, k_out) in enumerate(L):
            if li == 0:
                k_in = opt.latent_dim
            if li == len(L)-1:
                k_out = (dim_in+1) * dim_out  # weight and bias
            hyperlayer.append(torch.nn.Linear(k_in, k_out))
            if li != len(L)-1:
                hyperlayer.append(torch.nn.ReLU(inplace=False))
        hyperlayer = torch.nn.Sequential(*hyperlayer)
        return hyperlayer

    def hyperlayer_forward(self, opt, latent, module, layers, k0, interm_coord=False):
        batch_size = len(latent)
        impl_layers = []
        L = utils.get_layer_dims(layers)
        for li, (k_in, k_out) in enumerate(L):
            if li == 0:
                k_in = k0
            if interm_coord and li > 0:
                k_in += 3+(opt.impl.posenc_L*6 if opt.impl.posenc_L else 0)
            hyperlayer = module[li]
            out = hyperlayer.forward(latent).view(batch_size, k_in+1, k_out)
            impl_layers.append(utils.BatchLinear(weight=out[:, 1:], bias=out[:, :1]))
        return impl_layers


class ImplicitFunction(torch.nn.Module):
    """
        Base Implicit function. Parent class for DeformNetImplicitFunction and SDFImplicitFunction
    """
    def __init__(self, opt, impl_layers):
        super().__init__()
        self.opt = opt
        self.impl_layers = impl_layers
        self.define_network(opt, impl_layers)

    def define_heads(self, opt, impl_layers):
        layers = []
        for li, linear in enumerate(impl_layers):
            layers.append(linear)
            if li != len(impl_layers)-1:
                layers.append(torch.nn.LayerNorm(linear.bias.shape[-1], elementwise_affine=False))
                layers.append(torch.nn.ReLU(inplace=False))  # avoid backprop issues with higher-order gradients
        return torch.nn.Sequential(*layers)

    def positional_encoding(self, opt, points_3D, posenc_L=None):  # [B,...,3]
        shape = points_3D.shape
        last_dim = points_3D.shape[-1]
        points_enc = []
        if posenc_L is None:
            posenc_L = opt.impl.posenc_L
        if posenc_L:
            freq = 2**torch.arange(posenc_L, dtype=torch.float32, device=opt.device)*np.pi  # [L]
            spectrum = points_3D[..., None] * freq  # [B,...,3,L]
            sin, cos = spectrum.sin(), spectrum.cos()
            points_enc_L = torch.cat([sin, cos], dim=-1).view(*shape[:-1], 2*last_dim*posenc_L)  # [B,...,6L]
            points_enc.append(points_enc_L)
        points_enc = torch.cat(points_enc, dim=-1)  # [B,...,X]
        return points_enc
