from logging import debug
import numpy as np
import torch
from torch import nn
from manopth.manolayer import ManoLayer
from manopth.rodrigues_layer import batch_rodrigues


def recover_3d_proj(objpoints3d, camintr, est_scale, est_trans, off_z=0.4, input_res=(256, 256)):
    focal = camintr[:, :1, :1]
    batch_size = objpoints3d.shape[0]
    focal = focal.view(batch_size, 1)
    est_scale = est_scale.view(batch_size, 1)
    est_trans = est_trans.view(batch_size, 2)
    # est_scale is homogeneous to object scale change in pixels
    est_Z0 = focal * est_scale + off_z
    cam_centers = camintr[:, :2, 2]
    img_centers = (cam_centers.new(input_res) / 2).view(1, 2).repeat(batch_size, 1)
    est_XY0 = (est_trans + img_centers - cam_centers) * est_Z0 / focal
    est_c3d = torch.cat([est_XY0, est_Z0], -1).unsqueeze(1)
    recons3d = est_c3d + objpoints3d
    return recons3d, est_c3d
    

class AbsoluteBranch(nn.Module):
    def __init__(self, base_neurons=[512, 256], out_dim=3):
        super().__init__()
        layers = []
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Linear(inp_neurons, out_neurons))
            layers.append(nn.ReLU())
        self.final_layer = nn.Linear(out_neurons, out_dim)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)
        return out


class RotationBranch(nn.Module):
    def __init__(self, base_neurons=[512, 256], out_dim=3):
        super().__init__()
        layers = []
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Linear(inp_neurons, out_neurons))
            layers.append(nn.ReLU())
        self.final_layer = nn.Linear(out_neurons, out_dim)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)
        rot_matrix = batch_rodrigues(out).view(out.shape[0], 3, 3)
        return rot_matrix


class ManoBranch(nn.Module):
    def __init__(self, ncomps=15, base_neurons=[512,512,512], center_idx=0, use_shape=True, use_pca=True, mano_root="mano", dropout=0, absolute_depth=False, object_pose=False, use_obj_rot=False):
        """
        Args:
            mano_root (path): dir containing mano pickle files
        """
        super(ManoBranch, self).__init__()

        self.use_shape = use_shape
        self.use_pca = use_pca
        self.ncomps = ncomps
        self.absolute_depth = absolute_depth
        self.object_pose = object_pose
        self.use_obj_rot = use_obj_rot

        if self.use_pca:
            # pca comps + 3 global axis-angle params
            mano_pose_size = ncomps + 3
        else:
            # 15 joints + 1 global rotations, 9 comps per rot
            mano_pose_size = 16 * 9

        # Base layers
        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(zip(base_neurons[:-1], base_neurons[1:])):
            if dropout:
                base_layers.append(nn.Dropout(p=dropout))
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layer = nn.Sequential(*base_layers)

        # Pose layers
        self.pose_reg = nn.Linear(base_neurons[-1], mano_pose_size)

        # Shape layers
        if self.use_shape:
            self.shape_reg = torch.nn.Sequential(nn.Linear(base_neurons[-1], 10))
        
        if self.absolute_depth:
            self.trans_branch = AbsoluteBranch()
        
        if self.object_pose and self.use_obj_rot:
            self.object_rot_branch = RotationBranch()
        
        # Mano layers
        self.mano_layer = ManoLayer(
            ncomps=ncomps,
            center_idx=center_idx,
            side="right",
            mano_root=mano_root,
            use_pca=use_pca,
            flat_hand_mean=False
        )
        self.faces = torch.LongTensor(np.load('mano/closed_fmano.npy'))


    def forward(self, inp, cond_input=None):
        base_features = self.base_layer(inp)
        pose = self.pose_reg(base_features)

        if self.absolute_depth:
            scaletrans = self.trans_branch(inp)

        if self.object_pose and self.use_obj_rot:
            object_rot = self.object_rot_branch(inp)
        else:
            object_rot = None

        if not self.use_pca:
            mano_pose = pose.reshape(pose.shape[0], 16, 3, 3)
        else:
            mano_pose = pose

        # Get shape
        if self.use_shape:
            shape = self.shape_reg(base_features)
        else:
            shape = None

        # Pass through mano_right and mano_left layers
        if mano_pose is not None and shape is not None:
            verts, joints, hand_pose, global_trans, rot_center = self.mano_layer(mano_pose, th_betas=shape, root_palm=False)

        if self.absolute_depth:
            trans = scaletrans[:, 1:]
            scale = scaletrans[:, [0]]
            final_trans = trans * 100.0
            final_scale = scale * 0.0001
            cam_joints, center3d = recover_3d_proj(joints, cond_input['cam_intr'], final_scale, final_trans)
            cam_verts = center3d + verts
            results = {"verts": cam_verts, "joints": cam_joints, "shape": shape, "pcas": mano_pose, "pose": hand_pose, "center3d": center3d, "global_trans":global_trans, "rot_center": rot_center, "obj_rot": object_rot}
        else:
            center3d = cond_input['mano_root'].reshape((verts.shape[0], 1, 3))
            cam_joints = center3d + joints
            cam_verts = center3d + verts
            results = {"verts": cam_verts, "joints": cam_joints, "shape": shape, "pcas": mano_pose, "pose": hand_pose, "center3d": center3d, "global_trans":global_trans, "rot_center": rot_center, "obj_rot": object_rot}

        return results


def get_bone_ratio(pred_joints, target_joints, link=(9, 10)):
    bone_ref = torch.norm(
        target_joints[:, link[1]] - target_joints[:, link[0]], dim=1
    )
    bone_pred = torch.norm(
        pred_joints[:, link[1]] - pred_joints[:, link[0]], dim=1
    )
    bone_ratio = bone_ref / bone_pred
    return bone_ratio
