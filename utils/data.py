from configparser import Interpolation
from distutils.log import debug
import logging
import numpy as np
import os
import torch
import torch.utils.data
from torchvision import transforms
import trimesh
import cv2
import pickle
import time
import random
import lmdb
import json
from .img_utils import load_img, load_seg, get_aug_config, generate_patch_image, load_img_lmdb, load_seg_lmdb
from .sdf_utils import unpack_normal_params_lmdb, unpack_meta_params_lmdb, unpack_sdf_samples_lmdb, gen_instance_annos, unpack_sdf_samples, unpack_test_params


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        dataset_name="obman",
        image_source="rgb",
        hand_branch=True,
        obj_branch=True,
        mano_branch=False,
        depth_branch=False,
        disable_aug=False,
        background_aug=False,
        same_point=True,
        filter_dist=False,
        image_size=(224, 224),
        sdf_scale_factor=1,
        clamp=None,
        model_type="1encoder1decoder",
        use_lmdb=True
    ):
        self.subsample = subsample
        self.split = split
        self.image_size = image_size
        self.use_lmdb = use_lmdb

        self.dataset_name = dataset_name
        self.data_source = os.path.join(data_source, self.dataset_name, 'train')
        self.image_source = os.path.join(self.data_source, image_source)
        self.sdf_scale_factor = sdf_scale_factor

        self.hand_branch = hand_branch
        self.obj_branch = obj_branch
        self.mano_branch = mano_branch
        self.depth_branch = depth_branch

        self.filter_dist = filter_dist
        self.model_type = model_type

        self.disable_aug = disable_aug
        self.background_aug = background_aug
        self.same_point = same_point
        self.clamp = clamp

        self.raw_image_size = (3, 256, 256)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.sdf_db = [filename for filename in self.split['train']]
        
        if self.use_lmdb:
            self.img_lmdb = f'data/{self.dataset_name}/train/rgb.lmdb'
            self.img_env = lmdb.open(self.img_lmdb, readonly=True, lock=False, readahead=False, meminit=False)

            self.norm_lmdb = f'data/{self.dataset_name}/train/norm.lmdb'
            self.norm_env = lmdb.open(self.norm_lmdb, readonly=True, lock=False, readahead=False, meminit=False)

            self.meta_lmdb = f'data/{self.dataset_name}/train/meta.lmdb'
            self.meta_env = lmdb.open(self.meta_lmdb, readonly=True, lock=False, readahead=False, meminit=False)

            if self.hand_branch:
                self.hand_lmdb = f'data/{self.dataset_name}/train/sdf_hand.lmdb'
                self.hand_env = lmdb.open(self.hand_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
                with open(os.path.join(self.hand_lmdb, 'meta_info.json'), 'r') as f:
                   self.hand_meta = json.load(f)

            if self.obj_branch:
                self.obj_lmdb = f'data/{self.dataset_name}/train/sdf_obj.lmdb'
                self.obj_env = lmdb.open(self.obj_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
                with open(os.path.join(self.obj_lmdb, 'meta_info.json'), 'r') as f:
                     self.obj_meta = json.load(f)
            
            if self.background_aug:
                self.bg_lmdb = f'data/inria/rgb.lmdb'
                self.bg_env = lmdb.open(self.bg_lmdb, readonly=True, lock=False, readahead=False, meminit=False)

            logging.info(f'Using lmdb training files')
        else:
            self.anno_db = gen_instance_annos(self.data_source, self.sdf_db)

        logging.info(f'Finish constructing the dataset')

    def __len__(self):
        return len(self.sdf_db)

    def __getitem__(self, idx):
        data_key = self.sdf_db[idx]

        if 'obman' in self.dataset_name:
            cam_extr = torch.Tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        else:
            cam_extr = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        if self.use_lmdb:
            scale, offset = unpack_normal_params_lmdb(self.norm_env, data_key)
            mano_labels = unpack_meta_params_lmdb(self.meta_env, data_key, self.dataset_name) 
        else:
            scale = self.anno_db[data_key]['scale']
            offset = self.anno_db[data_key]['offset']
            mano_labels = self.anno_db[data_key]['mano_labels']
        
        cam_intr = mano_labels['cam_intr']
        if self.use_lmdb:
            img = load_img_lmdb(self.img_env, data_key, self.raw_image_size)
        else:
            img = load_img(os.path.join(self.image_source, data_key + '.jpg'))
            
        img_scale, rot, do_flip, color_scale, do_occlusion = get_aug_config(self.dataset_name, enable_flip=False)
        img_scale = 1.
        if self.disable_aug:
            rot = 0.
        rot_aug_mat = torch.from_numpy(np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32))
        center_crop_box = [(img.shape[1] - self.image_size[1]) // 2, (img.shape[0] - self.image_size[0]) // 2, self.image_size[1], self.image_size[0]]
        cam_intr[0, 0] = cam_intr[0, 0] / img_scale
        cam_intr[0, 2] = cam_intr[0, 2] / (img.shape[1]) * self.image_size[1]
        cam_intr[1, 1] = cam_intr[1, 1] / img_scale
        cam_intr[1, 2] = cam_intr[1, 2] / (img.shape[0]) * self.image_size[0]
        img, _ = generate_patch_image(img, center_crop_box, self.image_size, do_flip, img_scale, rot, False)

        encoder_input = self.transform(img)
        img_id = data_key

        hand_joints_3d = torch.mm(cam_extr, mano_labels['joints_3d'][:, 0:3].transpose(1, 0)).transpose(1, 0)
        obj_corners_3d = torch.mm(cam_extr, mano_labels['obj_corners_3d'][:, 0:3].transpose(1, 0)).transpose(1, 0)
        obj_rest_corners_3d = torch.mm(cam_extr, mano_labels['obj_rest_corners_3d'][:, 0:3].transpose(1, 0)).transpose(1, 0)
        
        # If only hand branch or obj branch is used, subsample is not reduced by half
        # to maintain the same number of samples used when trained with two branches.
        if not self.same_point or not (self.hand_branch and self.obj_branch):
            num_sample = self.subsample
        else:
            num_sample = int(self.subsample / 2)

        if self.hand_branch:
            if self.use_lmdb:
                hand_samples, hand_labels = unpack_sdf_samples_lmdb(self.hand_env, data_key, self.hand_meta, num_sample, hand=True, clamp=self.clamp, filter_dist=self.filter_dist)
            else:
                hand_samples, hand_labels = unpack_sdf_samples(self.data_source, data_key, num_sample, hand=True, clamp=self.clamp, filter_dist=self.filter_dist)
        else:
            hand_samples = torch.zeros((num_sample, 5), dtype=torch.float32)
            hand_labels = -torch.ones(num_sample, dtype=torch.float32)

        if self.obj_branch:
            if self.use_lmdb:
                obj_samples, obj_labels = unpack_sdf_samples_lmdb(self.obj_env, data_key, self.obj_meta, num_sample, hand=False, clamp=self.clamp, filter_dist=self.filter_dist)
            else:
                obj_samples, obj_labels = unpack_sdf_samples(self.data_source, data_key, num_sample, hand=False, clamp=self.clamp, filter_dist=self.filter_dist)
        else:
            obj_samples = torch.zeros((num_sample, 5), dtype=torch.float32)
            obj_labels = -torch.ones(num_sample, dtype=torch.float32)
        
        # transform points into the camera coordinate system
        hand_samples[:, 0:3] = hand_samples[:, 0:3] / scale - offset
        obj_samples[:, 0:3] = obj_samples[:, 0:3] / scale - offset

        if do_flip:
            hand_samples[:, 0] = -hand_samples[:, 0]
            obj_samples[:, 0] = -obj_samples[:, 0]
            hand_joints_3d[:, 0] = -hand_joints_3d[:, 0]
            obj_corners_3d[:, 0] = -obj_corners_3d[:, 0]

        hand_samples[:, 0:3] = torch.mm(rot_aug_mat, hand_samples[:, 0:3].transpose(1,0)).transpose(1,0)
        obj_samples[:, 0:3] = torch.mm(rot_aug_mat, obj_samples[:, 0:3].transpose(1,0)).transpose(1,0)
        hand_joints_3d[:, 0:3] = torch.mm(rot_aug_mat, hand_joints_3d[:, 0:3].transpose(1,0)).transpose(1,0)
        obj_corners_3d[:, 0:3] = torch.mm(rot_aug_mat, obj_corners_3d[:, 0:3].transpose(1,0)).transpose(1,0)

        mano_root = hand_joints_3d[0]
        hand_samples[:, 0:3] = (hand_samples[:, 0:3] - mano_root) * self.sdf_scale_factor
        obj_samples[:, 0:3] = (obj_samples[:, 0:3] - mano_root) * self.sdf_scale_factor

        hand_samples[:, 3:] = hand_samples[:, 3:] / scale * self.sdf_scale_factor
        obj_samples[:, 3:] = obj_samples[:, 3:] / scale * self.sdf_scale_factor

        hand_samples[:, 0:5] = hand_samples[:, 0:5] / 2
        obj_samples[:, 0:5] = obj_samples[:, 0:5] / 2

        input_iter = dict(img=encoder_input)
        label_iter = dict(pc_hand=hand_samples, cls_hand=hand_labels, pc_obj=obj_samples, cls_obj=obj_labels, hand_joints_3d=hand_joints_3d, obj_corners=obj_corners_3d[1:, :]-obj_corners_3d[[0], :], obj_center=obj_corners_3d[0, :])

        meta_iter = dict(cam_intr=cam_intr, img_id=img_id, mano_root=mano_root, rest_obj_corners=obj_rest_corners_3d[1:, :])

        return input_iter, label_iter, meta_iter


class ImagesInput(torch.utils.data.Dataset):
    def __init__(self, data_list, specs, task):
        self.input_files = data_list
        self.task = task
        self.image_source = f'data/{task}/test/rgb/' 
        self.cam_source = f'data/{task}/test/meta/' 
        self.image_size = specs['ImageSize']

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.test_db = []
        for filename in self.input_files:
            key = filename.split('/')[-1].split('.')[0]
            self.test_db.append(key)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        data_key = self.test_db[idx]
        cam_intr, mano_root, obj_rest_corners = unpack_test_params(os.path.join(self.cam_source, data_key + '.pkl'), self.task)

        img = load_img(os.path.join(self.image_source, data_key + '.jpg'))

        center_crop_box = [(img.shape[1] - self.image_size[1]) // 2, (img.shape[0] - self.image_size[0]) // 2, self.image_size[1], self.image_size[0]]
        cam_intr[0, 0] = cam_intr[0, 0]
        cam_intr[0, 2] = cam_intr[0, 2] / (img.shape[1]) * self.image_size[1]
        cam_intr[1, 1] = cam_intr[1, 1]
        cam_intr[1, 2] = cam_intr[1, 2] / (img.shape[0]) * self.image_size[0]
        img, _ = generate_patch_image(img, center_crop_box, self.image_size, False, 1.0, 0.0, False)

        img = self.transform(img)
        input_iter = dict(img=img)

        meta_iter = dict(img_id=self.input_files[idx], cam_intr=cam_intr, mano_root=mano_root, obj_rest_corners=obj_rest_corners)

        return input_iter, meta_iter