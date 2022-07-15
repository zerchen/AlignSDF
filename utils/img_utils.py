#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :img_utils.py
#@Date        :2021/10/22 22:47:22
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import random
import math
import cv2
import numpy as np
from .geometry_utils import rotate_2d


def load_img_lmdb(env, key, size, order='RGB'):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    return img
    

def load_seg_lmdb(env, key, size, task):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)

    seg_maps = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
    if 'obman' in task:
        #visible hand, visible obj, full hand
        seg_maps[:, :, 0][np.where(img[:, :, 0] == 100)] = 1
        seg_maps[:, :, 1][np.where(img[:, :, 0] == 100)] = 1
        seg_maps[:, :, 1][np.where(img[:, :, 0] == 22)] = 1
        seg_maps[:, :, 1][np.where(img[:, :, 0] == 24)] = 1
        seg_maps[:, :, 1][np.where(img[:, :, 0] == 20)] = 1

    return seg_maps


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.uint8)
    return img


def load_seg(path, task):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    seg_maps = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
    if 'obman' in task:
        #visible hand, visible obj, full hand
        seg_maps[:, :, 0][np.where(img[:, :, 0] == 100)] = 1
        seg_maps[:, :, 1][np.where(img[:, :, 0] == 100)] = 1
        seg_maps[:, :, 1][np.where(img[:, :, 0] == 22)] = 1
        seg_maps[:, :, 1][np.where(img[:, :, 0] == 24)] = 1
        seg_maps[:, :, 1][np.where(img[:, :, 0] == 20)] = 1
    
    return seg_maps


def preserve_img_aspect_ratio(img, cam_intr):
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]

    cx = cam_intr[0, 2]
    cy = cam_intr[1, 2]

    if H >= W:
        num_pixel = (H - W) // 2
        resize_img = cv2.copyMakeBorder(img, 0, 0, num_pixel, num_pixel, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        new_cx = cx + num_pixel
        new_cy = cy
    else:
        num_pixel = (W - H) // 2
        resize_img = cv2.copyMakeBorder(img, num_pixel, num_pixel, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        new_cx = cx
        new_cy = cy + num_pixel

    resize_img = resize_img.astype(np.uint8)
    cam_intr[0, 2] = new_cx
    cam_intr[1, 2] = new_cy

    return resize_img, cam_intr


def get_aug_config(dataset, enable_flip=False):
    if 'obman' in dataset:
        scale_factor = 0.25
        rot_factor = 45
        color_factor = 0.3
    else:
        scale_factor = 0.25
        rot_factor = 15
        color_factor = 0.2

    if 'obman' in dataset:
        scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    else:
        scale = np.clip(np.random.randn(), -1.0, 0.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= 0.6 else 0
    if enable_flip:
        do_flip = random.random() <= 0.5
    else:
        do_flip = False
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    do_occlusion = random.random() <= 0.5

    return scale, rot, do_flip, color_scale, do_occlusion


def generate_patch_image(cvimg, bbox, input_shape, do_flip, scale, rot, do_occlusion):
        img = cvimg.copy()
        img_height, img_width, _ = img.shape
    
        # synthetic occlusion
        if do_occlusion:
            while True:
                area_min = 0.0
                area_max = 0.7
                synth_area = (random.random() * (area_max - area_min) + area_min) * bbox[2] * bbox[3]
    
                ratio_min = 0.3
                ratio_max = 1 / 0.3
                synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)
    
                synth_h = math.sqrt(synth_area * synth_ratio)
                synth_w = math.sqrt(synth_area / synth_ratio)
                synth_xmin = random.random() * (bbox[2] - synth_w - 1) + bbox[0]
                synth_ymin = random.random() * (bbox[3] - synth_h - 1) + bbox[1]
    
                if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < img_width and synth_ymin + synth_h < img_height:
                    xmin = int(synth_xmin)
                    ymin = int(synth_ymin)
                    w = int(synth_w)
                    h = int(synth_h)
                    img[ymin:ymin + h, xmin:xmin + w, :] = np.random.rand(h, w, 3) * 255
                    break

        bb_c_x = float(bbox[0] + 0.5 * bbox[2])
        bb_c_y = float(bbox[1] + 0.5 * bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])
    
        if do_flip:
            img = img[:, ::-1, :]
            bb_c_x = img_width - bb_c_x - 1
    
        trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot, inv=False)
        img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
    
        return img_patch, trans


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans