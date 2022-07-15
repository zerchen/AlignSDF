import numpy as np


def trans_point2d(pt_2d, trans):
    src_pt = np.ones((pt_2d.shape[0], 1))
    src_pt = np.concatenate((pt_2d, src_pt), 1).T
    dst_pt = np.dot(trans, src_pt).T
    return dst_pt[:, 0:2]


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)