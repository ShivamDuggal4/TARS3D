import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
import PIL
import scipy.io
import os
import os.path as osp

from . import base
from misc import camera


class Dataset(base.Dataset):

    def __init__(self, opt, split="train", subset=None):
        super().__init__(opt, split)

        self.opt = opt
        self.data_dir = os.path.join(opt.data.dataset_root_path, "CUB_200_2011")
        self.data_cache_dir = os.path.join(opt.data.dataset_root_path, "cachedir/cub/")

        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(
            self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % split)
        self.anno_sfm_path = osp.join(
            self.data_cache_dir, 'sfm', 'anno_%s.mat' % split)

        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import ipdb
            ipdb.set_trace()

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = scipy.io.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = scipy.io.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)

    def get_camera(self, data_sfm):
        R = torch.from_numpy(np.copy(data_sfm.rot)).float()
        R_trans = torch.tensor([[0, 0, 1],
                                [-1, 0, 0],
                                [0, 1, 0]], dtype=torch.float32)
        pose = camera.pose(R=R@R_trans)
        return pose

    def get_metadata(self, data):
        # Adjust to 0 indexing
        bbox = np.array([data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], float) - 1
        return torch.from_numpy(bbox)

    def get_image(self, data):
        image_fname = osp.join(self.img_dir, str(data.rel_path))
        image = PIL.Image.open(image_fname).convert("RGB")
        mask = PIL.Image.fromarray((data.mask * 255).astype(np.uint8))
        image = PIL.Image.merge("RGBA", (*image.split(), mask))
        return image

    def preprocess_image(self, opt, image, bbox, aug=None):
        if aug is not None:
            image = self.apply_color_jitter(opt, image, aug.color_jitter)
            image = torchvision_F.hflip(image) if aug.flip else image
            x1, y1, x2, y2 = bbox
            image = image.rotate(aug.rot_angle, center=((x1+x2)/2, (y1+y2)/2), resample=PIL.Image.BICUBIC)
            image = self.square_crop(opt, image, bbox=bbox, crop_ratio=aug.crop_ratio)
        else:
            image = self.square_crop(opt, image, bbox=bbox)

        # torchvision_F.resize/torchvision_F.resized_crop will make masks really thick....
        crop_size = image.size[0]  # assume square
        image = image.resize((opt.W, opt.H))
        image = torchvision_F.to_tensor(image)
        rgb, mask = image[:3], image[3:]
        mask = (mask != 0).float()
        if opt.data.bgcolor:
            # replace background color using mask
            rgb = rgb*mask+opt.data.bgcolor*(1-mask)
        rgb = rgb*2-1
        return rgb, mask, crop_size

    def augment_camera(self, opt, pose, aug, pose_cam=None):
        if aug.flip:
            raise NotImplementedError
        if aug.rot_angle:
            angle = torch.tensor(aug.rot_angle)*np.pi/180
            # in-plane rotation
            R = camera.angle_to_rotation_matrix(-angle, axis="Z")
            rot_inplane = camera.pose(R=R)
            pose = camera.pose.compose([pose, camera.pose.invert(pose_cam), rot_inplane, pose_cam])
        return pose

    def square_crop(self, opt, image, bbox=None, crop_ratio=1.):
        # crop to canonical image size
        x1, y1, x2, y2 = bbox
        h, _ = y2-y1, x2-x1
        yc, xc = (y1+y2)/2, (x1+x2)/2
        S = h*3  # if opt.data.pascal3d.cat=="car" else max(h,w)*1.2
        # crop with random size (cropping out of boundary = padding)
        S2 = S*crop_ratio
        image = torchvision_F.crop(image, int(yc-S2/2), int(xc-S2/2), int(S2), int(S2))
        return image

    def forward_img(self, idx):
        opt = self.opt
        data = self.anno[idx]
        data_sfm = self.anno_sfm[idx]

        sample = dict(idx=idx, name=str(data.rel_path))

        aug = self.generate_augmentation(opt) if self.augment else None
        pose_cam = camera.pose(t=[0, 0, opt.camera.dist])
        assert(opt.camera.model == "orthographic")
        pose = self.get_camera(data_sfm)
        pose = camera.pose.compose([pose, pose_cam])

        if aug is not None:
            pose = self.augment_camera(opt, pose, aug, pose_cam=pose_cam)

        intr = False  # there are no None tensors
        sample.update(pose=pose, intr=intr,)

        bbox = self.get_metadata(data)
        image = self.get_image(data)
        rgb, mask, crop_size = self.preprocess_image(self.opt, image, bbox, aug)

        dt = self.compute_dist_transform(opt, mask)
        sample.update(rgb_input_map=rgb, mask_input_map=mask, dt_input_map=dt,)

        # vectorize images (and randomly sample)
        rgb = rgb.permute(1, 2, 0).view(opt.H*opt.W, 3)
        mask = mask.permute(1, 2, 0).view(opt.H*opt.W, 1)
        dt = dt.permute(1, 2, 0).view(opt.H*opt.W, 1)
        if self.split == "train" and opt.impl.rand_sample:
            ray_idx = torch.randperm(opt.H*opt.W)[:opt.impl.rand_sample]
            rgb, mask, dt = rgb[ray_idx], mask[ray_idx], dt[ray_idx]
            sample.update(ray_idx=ray_idx)

        sample.update(rgb_input=rgb, mask_input=mask, dt_input=dt,)
        return sample

    def __getitem__(self, idx):
        sample = self.forward_img(idx)
        return sample

    def __len__(self):
        return self.num_imgs
