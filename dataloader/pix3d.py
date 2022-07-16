import numpy as np
import os
import torch
import torchvision.transforms.functional as torchvision_F
import PIL
import scipy.io
import warnings
import json
import math
from easydict import EasyDict as edict

from dataloader import base
from misc import camera


class Dataset(base.Dataset):

    def __init__(self, opt, split="train", subset=None):
        super().__init__(opt, split)
        self.cat = dict(
            car="car",
            chair="chair",
            plane="aeroplane"
        )[opt.data.pix3d.cat]
        self.path = os.path.join(opt.data.dataset_root_path, "pix3D")
        self.path_mask = os.path.join(opt.data.dataset_root_path, "pix3D/mask/{}".format(self.cat))
        self.path_pc = os.path.join(opt.data.dataset_root_path, "pix3D/pointcloud/{}".format(self.cat))
        if split == 'train':
            self.list = self.get_list(opt, split)#[:20]
        else:
            self.list = self.get_list(opt, split)#[:20]
        self.img_list = self.generate_all_images()

    def get_list(self, opt, split):
        list_fname = os.path.join(opt.data.dataset_root_path, "pix3D/pix3d_clean_{}.json".format(split))
        list_all = json.load(open(list_fname, 'r'))
        print("total images: {}".format(len(list_all)))
        return list_all


    def generate_all_images(self):
        img_list = []
        for idx in range(len(self.list)):
            if idx % 100 == 0:
                print('processed: {} samples'.format(str(idx)))
            bbox = self.list[idx]["bbox"]
            image = self.get_image(self.opt,idx)
            image = self.square_crop(self.opt,image,bbox=bbox,crop_ratio=1.)
            crop_size = image.size[0] # assume square
            image = image.resize((self.opt.W, self.opt.H))
            img_list.append(image)
        return img_list

    
    def __getitem__(self, idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        
        # load camera
        meta = self.get_metadata(opt, idx)
        pose_cam = camera.pose(t=[0, 0, opt.camera.dist])
        assert(opt.camera.model == "orthographic")
        pose = self.get_camera(opt, idx, meta=meta)
        pose = camera.pose.compose([pose, pose_cam])
        if aug is not None:
            pose = self.augment_camera(opt,pose,aug,pose_cam=pose_cam)
        intr = False # there are no None tensors
        sample.update(
            pose=pose,
            intr=intr,
        )
        
        # load images and compute distance transform
        image = self.img_list[idx] #self.get_image(opt,idx)
        rgb,mask = self.preprocess_image(opt,idx,image,bbox=meta.bbox,aug=aug)
        
        # meta.crop_size = crop_size
        dt = self.compute_dist_transform(opt,mask)
        sample.update(
            rgb_input_map=rgb,
            mask_input_map=mask,
            dt_input_map=dt,
        )
        
        # vectorize images (and randomly sample)
        rgb = rgb.permute(1,2,0).view(opt.H*opt.W,3)
        mask = mask.permute(1,2,0).view(opt.H*opt.W,1)
        dt = dt.permute(1,2,0).view(opt.H*opt.W,1)
        if self.split=="train" and opt.impl.rand_sample:
            ray_idx = torch.randperm(opt.H*opt.W)[:opt.impl.rand_sample]
            rgb,mask,dt = rgb[ray_idx],mask[ray_idx],dt[ray_idx]
            sample.update(ray_idx=ray_idx)
        sample.update(
            rgb_input=rgb,
            mask_input=mask,
            dt_input=dt,
        )

        # load GT point cloud (only for validation!)
        dpc = self.get_pointcloud(opt,idx,meta=meta)
        sample.update(dpc=dpc)
        return sample



    def get_metadata(self,opt,idx):
        sample_data = self.list[idx]
        bbox = sample_data["bbox"]
        cam_position = sample_data["cam_position"]
        inplane_rotation = sample_data["inplane_rotation"]
        # cam_position[0] *= -1
        # inplane_rotation *= -1
        elevation = math.degrees(math.atan2(cam_position[1], math.sqrt(cam_position[0]**2 + cam_position[2]**2)))
        azimuth = math.degrees(math.pi/2. + math.atan2(cam_position[0], cam_position[2]))    
        azimuth = math.radians(azimuth)
        elevation = math.radians(elevation)

        meta = edict(
            cam=edict(
                azim=float(azimuth),
                elev=float(elevation),
                theta=float(inplane_rotation),
            ),
            model_name=sample_data["model"].split('/')[-2],
            bbox=torch.tensor(bbox),
        )
        return meta


    def get_camera(self,opt,idx,meta=None):
        azim = torch.tensor(meta.cam.azim).float()
        elev = torch.tensor(meta.cam.elev).float()
        theta = torch.tensor(meta.cam.theta).float()
        rot_mat = camera.rot_roll(theta) @ camera.rot_pitch(elev) @ camera.rot_yaw(azim)
        R_trans = torch.tensor([[0,0,1],
                                [0,1,0],
                                [-1,0,0]],dtype=torch.float32)
        rot_mat = torch.tensor(self.list[idx]["rot_mat"],dtype=torch.float32) @ R_trans
        pose = camera.pose(R=rot_mat)
        return pose


    def augment_camera(self,opt,pose,aug,pose_cam=None):
        if aug.flip:
            raise NotImplementedError
        if aug.rot_angle:
            angle = torch.tensor(aug.rot_angle)*np.pi/180
            R = camera.angle_to_rotation_matrix(-angle,axis="Z") # in-plane rotation
            rot_inplane = camera.pose(R=R)
            pose = camera.pose.compose([pose,camera.pose.invert(pose_cam),rot_inplane,pose_cam])
        return pose

    def get_image(self,opt,idx):
        sample_data = self.list[idx]
        image_fname = os.path.join(self.path, sample_data["img"])
        with warnings.catch_warnings(): # some images might contain corrupted EXIF data?
            warnings.simplefilter("ignore")
            image = PIL.Image.open(image_fname).convert("RGB")
        mask_fname = os.path.join(self.path, sample_data["mask"])
        mask = PIL.Image.open(mask_fname).convert("L")
        image = PIL.Image.merge("RGBA",(*image.split(),mask))
        return image

    def preprocess_image(self,opt,idx,image,bbox,aug=None):
        if aug is not None:
            x1,y1,x2,y2 = bbox
            image = self.apply_color_jitter(opt,image,aug.color_jitter)
            image = torchvision_F.hflip(image) if aug.flip else image
            
        image = torchvision_F.to_tensor(image)
        rgb,mask = image[:3],image[3:]
        mask = (mask!=0).float()
        if opt.data.bgcolor:
            # replace background color using mask
            rgb = rgb*mask+opt.data.bgcolor*(1-mask)
        rgb = rgb*2-1
        return rgb,mask

    def square_crop(self,opt,image,bbox=None,crop_ratio=1.):
        # crop to canonical image size
        x1,y1,x2,y2 = bbox
        h,w = y2-y1,x2-x1
        yc,xc = (y1+y2)/2,(x1+x2)/2
        S = max(h,w)*1.2
        # crop with random size (cropping out of boundary = padding)
        S2 = S*crop_ratio
        image = torchvision_F.crop(image,int(yc-S2/2),int(xc-S2/2),int(S2),int(S2))
        return image



    def get_pointcloud(self,opt,idx,meta=None):
        model_name = meta.model_name
        pc_fname = os.path.join(self.path, "pointcloud/{}/{}.npy".format(self.cat, model_name))
        pc = torch.from_numpy(np.load(pc_fname)).float()
        pc = torch.stack([pc[:,2],-pc[:,1],pc[:,0]],dim=-1)
        dpc = dict(
            points=pc,
            normals=torch.zeros_like(pc),
        )
        return dpc



    def __len__(self):
        return len(self.list)
