import torch
from misc import camera


class Renderer(torch.nn.Module):
    """
        LSTM differentiable renderer adopted from SRN and SDF-SRN.
    """
    def __init__(self, opt):
        super().__init__()
        self.define_ray_LSTM(opt)

    def define_ray_LSTM(self, opt):
        feat_dim = opt.arch.sdf_generator.layers_impl[-1]
        self.ray_lstm = torch.nn.LSTMCell(input_size=feat_dim, hidden_size=opt.arch.renderer.lstm_dim)
        self.lstm_pred = torch.nn.Linear(opt.arch.renderer.lstm_dim, 1)
        # initialize LSTM
        for name, param in self.ray_lstm.named_parameters():
            if "bias" not in name:
                continue
            n = param.shape[0]
            param.data[n//4:n//2].fill_(1.)

    def forward(self, opt, var, pose, intr=None, ray_idx=None):
        batch_size = len(pose)
        center, ray = camera.get_center_and_ray(opt, pose, intr=intr)  # [B,HW,3]

        if ray_idx is not None:
            gather_idx = ray_idx[..., None].repeat(1, 1, 3)
            ray = ray.gather(dim=1, index=gather_idx)
            if opt.camera.model == "orthographic":
                center = center.gather(dim=1, index=gather_idx)

        num_rays = ray_idx.shape[1] if ray_idx is not None else opt.H*opt.W
        depth = torch.empty(batch_size, num_rays, 1, device=opt.device).fill_(opt.impl.init_depth)  # [B,HW,1]
        level_all = []
        state = None

        for s in range(opt.impl.srn_steps):
            points_3D = camera.get_3D_points_from_depth(opt, center, ray, depth)  # [B,HW,3]
            level, feat = var.impl_func(var, points_3D, get_feat=True)  # [B,HW,K]
            level_all.append(level)
            state = self.ray_lstm(feat.view(batch_size*num_rays, -1), state)
            delta = self.lstm_pred(state[0]).view(batch_size, num_rays, 1).abs_()  # [B,HW,1]
            depth = depth+delta

        # final endpoint (supposedly crossing the zero-isosurface)
        points_3D_2ndlast, level_2ndlast = points_3D, level
        points_3D = camera.get_3D_points_from_depth(opt, center, ray, depth)  # [B,HW,3]
        level = var.impl_func(var, points_3D)  # [B,HW,1]
        mask = (level <= 0).float()
        level_all.append(level)
        level_all = torch.cat(level_all, dim=-1)  # [B,HW,N]

        # get isosurface=0 intersection
        def func(x): return var.impl_func(var, x)
        points_3D_iso0 = self.bisection(x0=points_3D_2ndlast, x1=points_3D, y0=level_2ndlast, y1=level,
                                        func=func, num_iter=opt.impl.bisection_steps)  # [B,HW,3]

        level, rgb = var.impl_func(var, points_3D_iso0, get_rgb=True)  # [B,HW,K]
        depth = camera.get_depth_from_3D_points(opt, center, ray, points_3D_iso0)  # [B,HW,1]

        return rgb, depth, level, mask, level_all

    def bisection(self, x0, x1, y0, y1, func, num_iter):
        for s in range(num_iter):
            x2 = (x0+x1)/2
            y2 = func(x2)

            # update x0 if side else update x1
            side = ((y0 < 0) ^ (y2 > 0)).float()
            x0, x1 = x2*side+x0*(1-side), x1*side+x2*(1-side)
            y0, y1 = y2*side+y0*(1-side), y1*side+y2*(1-side)
        x2 = (x0+x1)/2
        return x2
