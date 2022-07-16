import os
import tqdm
import torch
import torch.nn.functional as torch_F
from easydict import EasyDict as edict

from trainer import base
from misc import utils, camera, eval_3D, dump_3D_and_deformations
import misc.visualization_utils as vis_utils
from misc.utils import log


class Trainer(base.Trainer):
    """
        - Main model class responsible for the core reconstruction operation.
        - Both Stage 1 and Stage 2 reconstructions are performed via this class.
        - Adopts form the base class defined in pipeline/base.py
    """
    def __init__(self, opt):
        super().__init__(opt)

    @torch.no_grad()
    def evaluate(self, opt, ep=None, training=False):

        if not training:
            os.makedirs("{}/dump/".format(opt.output_path), exist_ok=True)
        self.model.eval()
        loss_eval = edict()
        metric_eval = dict(dist_acc=0., dist_cov=0.)
        loader = tqdm.tqdm(self.test_loader, desc="evaluating", leave=False)

        for it, batch in enumerate(loader):
            var = edict(batch)
            var, loss = self.evaluate_batch(opt, var, ep, it)
            for key in loss:
                loss_eval.setdefault(key, 0.)
                loss_eval[key] += loss[key]*len(var.idx)

            if not training and opt.is_stage_two_active:
                dump_3D_and_deformations.create_all_meshes(opt, var)

            dist_acc, dist_cov = eval_3D.compute_chamfer_dist(opt, var)
            metric_eval["dist_acc"] += dist_acc*len(var.idx)
            metric_eval["dist_cov"] += dist_cov*len(var.idx)
            loader.set_postfix(loss="{:.3f}".format(loss.all))
            if it == 0 and training:
                self.visualize(opt, var, step=ep, split="eval")
            if not training:
                self.dump_results(opt, var, write_new=(it == 0))

        for key in loss_eval:
            loss_eval[key] /= len(self.test_data)
        for key in metric_eval:
            metric_eval[key] /= len(self.test_data)
        log.loss_eval(opt, loss_eval, chamfer=(metric_eval["dist_acc"], metric_eval["dist_cov"]))
        if training:
            # log/visualize results to tb/vis
            self.log_scalars(opt, var, loss_eval, metric=metric_eval, step=ep, split="eval")

    def evaluate_batch(self, opt, var, ep=None, it=None):
        var = utils.move_to_device(var, opt.device)
        var = self.model.forward(opt, var, training=False)
        loss = self.model.compute_loss(opt, var, training=False)
        loss = self.summarize_loss(opt, var, loss)
        return var, loss

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):

        if split == "train":
            dist_acc, dist_cov = eval_3D.compute_chamfer_dist(opt, var)
            metric = dict(dist_acc=dist_acc, dist_cov=dist_cov)
        super().log_scalars(opt, var, loss, metric=metric, step=step, split=split)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"):
        vis_utils.tb_image(
            opt, self.tb, step, split, "image_input", var.rgb_input_map, masks=var.mask_input_map,
            from_range=(-1, 1), poses=var.pose)
        vis_utils.tb_image(opt, self.tb, step, split, "mask_input", var.mask_input_map)
        vis_utils.tb_image(opt, self.tb, step, split, "level_input", var.dt_input_map, cmap="hot")

        if not (opt.impl.rand_sample and split == "train"):
            vis_utils.tb_image(
                opt, self.tb, step, split, "image_recon", var.rgb_recon_map, masks=var.mask_map,
                from_range=(-1, 1), poses=var.pose)
            vis_utils.tb_image(opt, self.tb, step, split, "depth", 1/(var.depth_map-opt.impl.init_depth+1))
            normal = self.compute_normal_from_depth(opt, var.depth_map, intr=var.intr)
            vis_utils.tb_image(opt, self.tb, step, split, "normal", normal, from_range=(-1, 1))
            vis_utils.tb_image(opt, self.tb, step, split, "mask", var.mask_map)
            vis_utils.tb_image(
                opt, self.tb, step, split, "depth_masked", 1/(var.depth_map-opt.impl.init_depth+1)*var.mask_map)
            mask_normal = var.mask_map[..., 1:-1, 1:-1]
            vis_utils.tb_image(opt, self.tb, step, split, "normal_masked",
                               normal*mask_normal+(-1)*(1-mask_normal), from_range=(-1, 1))
            vis_utils.tb_image(opt, self.tb, step, split, "level", var.level_map, cmap="hot")

        # visualize point cloud
        if opt.eval and opt.visdom:
            # suppress weird (though unharmful) visdom errors related to remote connections
            with utils.suppress(stdout=True, stderr=True):
                vis_utils.vis_pointcloud(
                    opt, self.vis, step, split, pred=var.dpc_pred, GT=var.dpc.points if "dpc" in var else None)

    @torch.no_grad()
    def dump_results(self, opt, var, write_new=False):
        os.makedirs("{}/dump/".format(opt.output_path), exist_ok=True)
        vis_utils.dump_images(
            opt, var.idx, "image_input", var.rgb_input_map, masks=var.mask_input_map, from_range=(-1, 1))
        vis_utils.dump_images(opt, var.idx, "image_recon", var.rgb_recon_map, masks=var.mask_map, from_range=(-1, 1))
        vis_utils.dump_images(opt, var.idx, "depth", 1 / (var.depth_map-opt.impl.init_depth+1))
        normal = self.compute_normal_from_depth(opt, var.depth_map, intr=var.intr)
        vis_utils.dump_images(opt, var.idx, "normal", normal, from_range=(-1, 1))
        vis_utils.dump_images(opt, var.idx, "mask", var.mask_map)
        vis_utils.dump_images(opt, var.idx, "mask_input", var.mask_input_map)
        vis_utils.dump_images(opt, var.idx, "depth_masked", 1/(var.depth_map-opt.impl.init_depth+1)*var.mask_map)
        mask_normal = var.mask_map[..., 1:-1, 1:-1]
        vis_utils.dump_images(
            opt, var.idx, "normal_masked", normal * mask_normal+(-1)*(1-mask_normal), from_range=(-1, 1))
        vis_utils.dump_meshes(opt, var.idx, "mesh", var.mesh_pred)
        # vis_utils.dump_point_cloud(opt,var.idx,"point_cloud_gt",var.dpc.points.data.cpu())
        vis_utils.dump_point_cloud(opt,var.idx,"point_cloud_pred",var.dpc_pred.data.cpu())
        vis_utils.dump_point_cloud(opt,var.idx,"point_cloud_pred_canonical",var.dpc_pred_canonical.data.cpu())

        # write/append to html for convenient visualization
        html_fname = "{}/dump/vis.html".format(opt.output_path)
        with open(html_fname, "w" if write_new else "a") as html:
            for i in var.idx:
                html.write("{} ".format(i))
                html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i, "image_input"))
                html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i, "image_recon"))
                html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i, "depth"))
                html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i, "normal"))
                html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i, "mask"))
                html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i, "mask_input"))
                html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i, "depth_masked"))
                html.write("<img src=\"{}_{}.png\" height=64 width=64> ".format(i, "normal_masked"))
                html.write("<br>\n")
        # write chamfer distance results
        # chamfer_fname = "{}/chamfer.txt".format(opt.output_path)
        # with open(chamfer_fname,"w" if write_new else "a") as file:
        #     for i,acc,comp in zip(var.idx,var.cd_acc,var.cd_comp):
        #         file.write("{} {:.8f} {:.8f}\n".format(i,acc,comp))

    @torch.no_grad()
    def compute_normal_from_depth(self, opt, depth, intr=None):
        batch_size = len(depth)
        pose = camera.pose(t=[0, 0, 0]).repeat(batch_size, 1, 1).to(opt.device)
        center, ray = camera.get_center_and_ray(opt, pose, intr=intr)  # [B,HW,3]
        depth = depth.view(batch_size, opt.H*opt.W, 1)
        pts_cam = camera.get_3D_points_from_depth(opt, center, ray, depth)  # [B,HW,3]
        pts_cam = pts_cam.view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2)  # [B,3,H,W]
        dy = pts_cam[..., 2:, :]-pts_cam[..., :-2, :]  # [B,3,H-2,W]
        dx = pts_cam[..., :, 2:]-pts_cam[..., :, :-2]  # [B,3,H,W-2]
        dy = torch_F.normalize(dy, dim=1)[..., :, 1:-1]  # [B,3,H-2,W-2]
        dx = torch_F.normalize(dx, dim=1)[..., 1:-1, :]  # [B,3,H-2,W-2]
        normal = dx.cross(dy, dim=1)
        return normal
