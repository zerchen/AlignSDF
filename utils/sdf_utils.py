import numpy as np
import os
import torch
import pickle


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def filter_invalid_sdf_lmdb(tensor, dist):
    keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
    return tensor[keep, :]


def filter_invalid_sdf(tensor, lab_tensor, dist):
    keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
    return tensor[keep, :], lab_tensor[keep, :]


def unpack_normal_params_lmdb(env, key):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))

    norm = np.frombuffer(buf, dtype=np.float32)
    offset = torch.from_numpy(norm[0:3])
    scale = torch.from_numpy(norm[[3]])

    return scale, offset


def unpack_normal_params(data_source, key):
    npz = np.load(os.path.join(data_source, 'norm', key + '.npz'))
    scale = torch.from_numpy(npz["scale"])
    offset = torch.from_numpy(npz["offset"])

    return scale, offset


def unpack_meta_params_lmdb(env, key, dataset):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    
    meta_info = np.frombuffer(buf, dtype=np.float32).reshape([-1, 3])
    label_info = {}
    label_info["joints_3d"] = torch.from_numpy(meta_info[:21, :3])
    label_info["obj_corners_3d"] = torch.from_numpy(meta_info[21:30, :3].astype(np.float32))
    label_info["obj_rest_corners_3d"] = torch.from_numpy(meta_info[30:39, :3].astype(np.float32))
    label_info["hand_pose"] = torch.from_numpy(meta_info[39:54, :3]).reshape([-1])
    affine_matrix = torch.zeros((4, 4))
    affine_matrix[3, 3] = 1.
    affine_matrix[:3, :4] = torch.from_numpy(meta_info[54:58, :3]).reshape([3, -1])
    label_info["affine_transform"] = affine_matrix
    cam_intr = torch.zeros((3, 4))
    if dataset == 'obman':
        cam_intr[:3, :3] = torch.from_numpy(np.array([[480., 0., 128.], [0., 480., 128.], [0., 0., 1.]]))
    else:
        cam_intr[:3, :3] = torch.from_numpy(meta_info[58:61, :3]).reshape([3, -1])
    label_info["cam_intr"] = cam_intr

    return label_info


def unpack_meta_params(data_source, key):
    with open(os.path.join(data_source, 'meta', key + '.pkl'), 'rb') as f:
        pkl = pickle.load(f)

    label_info = {}
    label_info['joints_3d'] = torch.from_numpy(pkl['coords_3d'])
    label_info['obj_corners_3d'] = torch.from_numpy(pkl['obj_corners_3d'].astype(np.float32))
    label_info['obj_rest_corners_3d'] = torch.from_numpy(pkl['obj_rest_corners_3d'].astype(np.float32))
    label_info['hand_pose'] = torch.from_numpy(pkl['hand_pose']).reshape([-1])
    affine_matrix = torch.zeros((4, 4))
    affine_matrix[3, 3] = 1.
    affine_matrix[:3, :4] = torch.from_numpy(pkl['affine_transform'][:3, :])
    label_info["affine_transform"] = affine_matrix
    cam_intr = torch.zeros((3, 4))
    if 'obman' in data_source:
        cam_intr[:3, :3] = torch.from_numpy(np.array([[480., 0., 128.], [0., 480., 128.], [0., 0., 1.]]))
    else:
        cam_intr[:3, :3] = torch.from_numpy(pkl['cam_intr'][:3, :3])
    label_info['cam_intr'] = cam_intr

    return label_info


def unpack_test_params(path, task):
    cam_intr = torch.zeros((3, 4))
    if 'obman' in task or 'ho3d' in task:
        cam_extr = torch.Tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    else:
        cam_extr = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    if 'obman' in task:
        cam_intr[:3, :3] = torch.from_numpy(np.array([[480., 0., 128.], [0., 480., 128.], [0., 0., 1.]]))
        with open(os.path.join(path), 'rb') as f:
            pkl = pickle.load(f)
        mano_root = torch.from_numpy(pkl['coords_3d'][[0]])
        mano_root = (cam_extr @ mano_root.transpose(0, 1)).transpose(0, 1)
    else:
        with open(os.path.join(path), 'rb') as f:
            pkl = pickle.load(f)
        if 'ho3d' in task:
            cam_intr[:3, :3] = torch.from_numpy(pkl['camMat'][:3, :3])
            mano_root = torch.from_numpy(pkl['handJoints3D'].astype(np.float32)).unsqueeze(0)
            mano_root = (cam_extr @ mano_root.transpose(0, 1)).transpose(0, 1)
        elif 'dexycb' in task:
            cam_intr[:3, :3] = torch.from_numpy(pkl['cam_intr'][:3, :3])
            mano_root = torch.from_numpy(pkl['coords_3d'][[0]])
            mano_root = (cam_extr @ mano_root.transpose(0, 1)).transpose(0, 1)

    obj_rest_corners = torch.from_numpy(pkl['obj_rest_corners_3d'][1:, :])
    obj_rest_corners = (cam_extr @ obj_rest_corners.transpose(0, 1)).transpose(0, 1)

    return cam_intr, mano_root, obj_rest_corners


def unpack_sdf_samples_lmdb(env, key, meta, subsample=None, hand=True, clamp=None, filter_dist=False):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))

    index = meta['keys'].index(key)
    npz = np.frombuffer(buf, dtype=np.float32)
    pos_num = meta['pos_num'][index]
    neg_num = meta['neg_num'][index]
    feat_dim = meta['dim']
    total_num  = pos_num + neg_num
    npz = npz.reshape((-1, feat_dim))[:total_num, :]

    if subsample is None:
        return npz
    try:
        pos_tensor = remove_nans(torch.from_numpy(npz[:pos_num, :]))
        neg_tensor = remove_nans(torch.from_numpy(npz[pos_num:, :]))
    except Exception as e:
        print("fail to load {}, {}".format(key, e))

    # split the sample into half
    half = int(subsample / 2)

    if filter_dist:
        pos_tensor = filter_invalid_sdf_lmdb(pos_tensor, 2.0)
        neg_tensor = filter_invalid_sdf_lmdb(neg_tensor, 2.0)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples_and_labels = torch.cat([sample_pos, sample_neg], 0)
    samples = samples_and_labels[:, :-1]
    labels = samples_and_labels[:, -1]

    if clamp:
        labels[samples[:, 3] < -clamp] = -1
        labels[samples[:, 3] > clamp] = -1

    if not hand:
        labels[:] = -1

    return samples, labels


def unpack_sdf_samples(data_source, key, subsample=None, hand=True, clamp=None, filter_dist=False):
    if hand:
        npz_path = os.path.join(data_source, 'sdf_hand', key + '.npz')
    else:
        npz_path = os.path.join(data_source, 'sdf_obj', key + '.npz')
    
    npz = np.load(npz_path)
    if subsample is None:
        return npz
    try:
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
        pos_sdf_other = torch.from_numpy(npz["pos_other"])
        neg_sdf_other = torch.from_numpy(npz["neg_other"])
        if hand:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
        else:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
    except Exception as e:
        print("fail to load {}, {}".format(key, e))
    
    if hand:
        pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
        neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
    else:
        xyz_pos = pos_tensor[:, :3]
        sdf_pos = pos_tensor[:, 3].unsqueeze(1)
        pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

        xyz_neg = neg_tensor[:, :3]
        sdf_neg = neg_tensor[:, 3].unsqueeze(1)
        neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

    # split the sample into half
    half = int(subsample / 2)

    if filter_dist:
        pos_tensor, lab_pos_tensor = filter_invalid_sdf(pos_tensor, lab_pos_tensor, 2.0)
        neg_tensor, lab_neg_tensor = filter_invalid_sdf(neg_tensor, lab_neg_tensor, 2.0)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    # label
    sample_pos_lab = torch.index_select(lab_pos_tensor, 0, random_pos)
    sample_neg_lab = torch.index_select(lab_neg_tensor, 0, random_neg)

    # hand part label
    # 0-finger corase, 1-finger fine, 2-contact, 3-sealed wrist
    hand_part_pos = sample_pos_lab[:, 0]
    hand_part_neg = sample_neg_lab[:, 0]
    samples = torch.cat([sample_pos, sample_neg], 0)
    labels = torch.cat([hand_part_pos, hand_part_neg], 0)

    if clamp:
        labels[samples[:, 3] < -clamp] = -1
        labels[samples[:, 3] > clamp] = -1

    if not hand:
        labels[:] = -1

    return samples, labels


def gen_instance_annos(data_source, db):
    anno = dict()
    for key in db:
        anno[key] = {}
        scale, offset = unpack_normal_params(data_source, key)
        anno[key]['scale'] = scale
        anno[key]['offset'] = offset
        anno[key]['mano_labels'] = unpack_meta_params(data_source, key)
    
    return anno