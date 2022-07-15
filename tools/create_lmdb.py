#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :create_lmdb.py
#@Date        :2021/10/24 22:36:29
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr


import lmdb
import cv2
import numpy as np
import os
import sys
import pickle
from glob import glob
from fire import Fire
from tqdm import tqdm
import json


def main(dataset, mode):
    opt = dict()

    opt['image'] = dict()
    opt['image']['name'] = f'{dataset}_rgb_{mode}'
    opt['image']['data_folder'] = f'data/{dataset}/{mode}/rgb/'
    opt['image']['lmdb_save_path'] = f'data/{dataset}/{mode}/rgb.lmdb'
    opt['image']['commit_interval'] = 100

    opt['segm'] = dict()
    opt['segm']['name'] = f'{dataset}_segm_{mode}'
    opt['segm']['data_folder'] = f'data/{dataset}/{mode}/segm/'
    opt['segm']['lmdb_save_path'] = f'data/{dataset}/{mode}/segm.lmdb'
    opt['segm']['commit_interval'] = 100

    opt['norm'] = dict()
    opt['norm']['name'] = f'{dataset}_norm_{mode}'
    opt['norm']['data_folder'] = f'data/{dataset}/{mode}/norm/'
    opt['norm']['lmdb_save_path'] = f'data/{dataset}/{mode}/norm.lmdb'
    opt['norm']['commit_interval'] = 100

    opt['meta'] = dict()
    opt['meta']['name'] = f'{dataset}_meta_{mode}'
    opt['meta']['data_folder'] = f'data/{dataset}/{mode}/meta/'
    opt['meta']['lmdb_save_path'] = f'data/{dataset}/{mode}/meta.lmdb'
    opt['meta']['commit_interval'] = 100

    if mode == 'train' or mode == 'val':
        opt['sdf_hand'] = dict()
        opt['sdf_hand']['name'] = f'{dataset}_sdf_hand_{mode}'
        opt['sdf_hand']['data_folder'] = f'data/{dataset}/{mode}/sdf_hand/'
        opt['sdf_hand']['lmdb_save_path'] = f'data/{dataset}/{mode}/sdf_hand.lmdb'
        opt['sdf_hand']['commit_interval'] = 100
        opt['sdf_hand']['is_hand'] = True

        opt['sdf_obj'] = dict()
        opt['sdf_obj']['name'] = f'{dataset}_sdf_obj_{mode}'
        opt['sdf_obj']['data_folder'] = f'data/{dataset}/{mode}/sdf_obj/'
        opt['sdf_obj']['lmdb_save_path'] = f'data/{dataset}/{mode}/sdf_obj.lmdb'
        opt['sdf_obj']['commit_interval'] = 100
        opt['sdf_obj']['is_hand'] = False

    general_image_folder(opt['image'])
    general_image_folder(opt['segm'])
    general_norm_folder(opt['norm'])
    general_meta_folder(opt['meta'])
    if mode == 'train' or mode == 'val':
        general_sdf_folder(opt['sdf_hand'])
        general_sdf_folder(opt['sdf_obj'])


def general_meta_folder(opt):
    meta_folder = opt['data_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    print('Reading meta path list ...')
    all_meta_list = sorted(glob(os.path.join(meta_folder, '*')))

    keys = []
    for meta_path in all_meta_list:
        keys.append(os.path.basename(meta_path).split('.')[0])
    
    # create lmdb environment
    # estimate the space of the file
    data_size_per_meta = np.zeros((61, 3), dtype=np.float32).nbytes
    print('data size per meta is: ', data_size_per_meta)
    data_size = data_size_per_meta * len(all_meta_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    txn = env.begin(write=True)
    tqdm_iter = tqdm(enumerate(zip(all_meta_list, keys)), total=len(all_meta_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))
        
        key_byte = key.encode('ascii')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        data_sample = np.zeros((61, 3), dtype=np.float32)
        data_sample[:21, :3] = data['coords_3d']
        data_sample[21:30, :3] = data['obj_corners_3d']
        data_sample[30:39, :3] = data['obj_rest_corners_3d']
        data_sample[39:54, :3] = data['hand_pose'].reshape([-1, 3])
        data_sample[54:58, :3] = data['affine_transform'][:3, :].reshape([-1, 3])
        if ('ho3d' in opt['name']) or ('dexycb' in opt['name']):
            data_sample[58:61, :3] = data['cam_intr'][:3, :3].reshape([-1, 3])
        elif 'obman' in opt['name']:
            data_sample[58:61, :3] = np.array([[480., 0., 128.], [0., 480., 128.], [0., 0., 1.]])

        txn.put(key_byte, data_sample)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    meta_info['keys'] = keys
    with open(os.path.join(lmdb_save_path, 'meta_info.json'), "w") as f:
        json.dump(meta_info, f, indent=2)


def general_norm_folder(opt):
    norm_folder = opt['data_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    print('Reading norm path list ...')
    all_norm_list = sorted(glob(os.path.join(norm_folder, '*')))

    keys = []
    for norm_path in all_norm_list:
        keys.append(os.path.basename(norm_path).split('.')[0])
    
    # create lmdb environment
    # estimate the space of the file
    data_size_per_norm = np.zeros((4), dtype=np.float32).nbytes
    print('data size per norm is: ', data_size_per_norm)
    data_size = data_size_per_norm * len(all_norm_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    txn = env.begin(write=True)
    tqdm_iter = tqdm(enumerate(zip(all_norm_list, keys)), total=len(all_norm_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))
        
        key_byte = key.encode('ascii')
        data = np.load(path)
        data_sample = np.zeros(4, dtype=np.float32)
        data_sample[0:3] = data['offset']
        data_sample[3] = data['scale']

        txn.put(key_byte, data_sample)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    meta_info['keys'] = keys
    with open(os.path.join(lmdb_save_path, 'meta_info.json'), "w") as f:
        json.dump(meta_info, f, indent=2)


def general_sdf_folder(opt):
    sdf_folder = opt['data_folder']
    lmdb_save_path = opt['lmdb_save_path']
    is_hand = opt['is_hand']
    meta_info = {'name': opt['name']}

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    print('Reading sdf path list ...')
    all_sdf_list = sorted(glob(os.path.join(sdf_folder, '*')))

    keys = []
    for sdf_path in all_sdf_list:
        keys.append(os.path.basename(sdf_path).split('.')[0])
    
    # create lmdb environment
    # estimate the space of the file
    data_size_per_sdf = np.zeros((20000, 6), dtype=np.float32).nbytes
    print('data size per sdf is: ', data_size_per_sdf)
    data_size = data_size_per_sdf * len(all_sdf_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    txn = env.begin(write=True)
    tqdm_iter = tqdm(enumerate(zip(all_sdf_list, keys)), total=len(all_sdf_list), leave=False)
    pos_num = []
    neg_num = []
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))
        
        key_byte = key.encode('ascii')
        data = np.load(path)

        pos_data = data['pos']
        pos_other_data = data['pos_other']
        if is_hand:
            lab_pos_data = data['lab_pos'][:, [0]]
        else:
            lab_pos_data = data['lab_pos_other'][:, [0]]

        neg_data = data['neg']
        neg_other_data = data['neg_other']
        if is_hand:
            lab_neg_data = data['lab_neg'][:, [0]]
        else:
            lab_neg_data = data['lab_neg_other'][:, [0]]

        pos_num.append(pos_data.shape[0])
        neg_num.append(neg_data.shape[0])

        if is_hand:
            pos_sample = np.concatenate((pos_data, pos_other_data, lab_pos_data), axis=1)
            neg_sample = np.concatenate((neg_data, neg_other_data, lab_neg_data), axis=1)
            data_sample_real = np.concatenate((pos_sample, neg_sample), axis=0)
            data_sample = np.zeros((20000, 6), dtype=np.float32)
            data_sample[:pos_data.shape[0] + neg_data.shape[0], :] = data_sample_real
        else:
            pos_sample = np.concatenate((pos_data[:, :3], pos_other_data, pos_data[:, [3]], lab_pos_data), axis=1)
            neg_sample = np.concatenate((neg_data[:, :3], neg_other_data, neg_data[:, [3]], lab_neg_data), axis=1)
            data_sample_real = np.concatenate((pos_sample, neg_sample), axis=0)
            data_sample = np.zeros((20000, 6), dtype=np.float32)
            data_sample[:pos_data.shape[0] + neg_data.shape[0], :] = data_sample_real
        
        txn.put(key_byte, data_sample)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    meta_info['pos_num'] = pos_num
    meta_info['neg_num'] = neg_num
    meta_info['dim'] = 6
    meta_info['keys'] = keys
    with open(os.path.join(lmdb_save_path, 'meta_info.json'), "w") as f:
        json.dump(meta_info, f, indent=2)


def general_image_folder(opt):
    img_folder = opt['data_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}
    
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)
    
    print('Reading image path list ...')
    all_img_list = sorted(glob(os.path.join(img_folder, '*')))

    keys = []
    for img_path in all_img_list:
        keys.append(os.path.basename(img_path).split('.')[0])

    # create lmdb environment
    # estimate the space of the file
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    # write data to lmdb
    txn = env.begin(write=True)
    tqdm_iter = tqdm(enumerate(zip(all_img_list, keys)), total=len(all_img_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))
        
        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    with open(os.path.join(lmdb_save_path, 'meta_info.json'), "w") as f:
        json.dump(meta_info, f, indent=2)


if __name__ == '__main__':
    Fire(main)