from distutils.log import debug
import logging
from manopth.rotproj import batch_rotprojs
import os
import json
import numpy as np
import torch
from torch.nn import functional as F
import trimesh


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def add_train_args(arg_parser):
    arg_parser.add_argument(
        "--epoch",
        dest="epoch",
        type=int
    )
    arg_parser.add_argument(
        "--add_epoch",
        dest="add_epoch",
        type=int
    )
    arg_parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int
    )
    arg_parser.add_argument(
        "--lr",
        dest="lr",
        type=float
    )
    arg_parser.add_argument(
        "--lr_interval",
        dest="lr_interval",
        type=int
    )
    arg_parser.add_argument(
        "--lr_factor",
        dest="lr_factor",
        type=float
    )
    arg_parser.add_argument(
        "--image_size",
        dest="image_size",
        type=int
    )
    arg_parser.add_argument(
        "--num_points",
        dest="num_points",
        type=int
    )
    arg_parser.add_argument(
        "--latent_size",
        dest="latent_size",
        type=int
    )
    arg_parser.add_argument(
        "--pose_size",
        dest="pose_size",
        type=int
    )
    arg_parser.add_argument(
        "--point_size",
        dest="point_size",
        type=int
    )
    arg_parser.add_argument(
        "--mano",
        dest="mano",
        action='store_true',
    )
    arg_parser.add_argument(
        "--cls",
        dest="cls",
        action='store_true',
    )
    arg_parser.add_argument(
        "--random_seed",
        dest="random_seed",
        type=int
    )
    arg_parser.add_argument(
        "--depth",
        dest="depth",
        action='store_true',
    )
    arg_parser.add_argument(
        "--obj_pose",
        dest="obj_pose",
        action='store_true',
    )
    arg_parser.add_argument(
        "--hsw",
        dest="hsw",
        type=float
    )
    arg_parser.add_argument(
        "--osw",
        dest="osw",
        type=float
    )
    arg_parser.add_argument(
        "--jw",
        dest="jw",
        type=float
    )
    arg_parser.add_argument(
        "--vw",
        dest="vw",
        type=float
    )
    arg_parser.add_argument(
        "--srw",
        dest="srw",
        type=float
    )
    arg_parser.add_argument(
        "--prw",
        dest="prw",
        type=float
    )
    arg_parser.add_argument(
        "--segw",
        dest="segw",
        type=float
    )
    arg_parser.add_argument(
        "--ocw",
        dest="ocw",
        type=float
    )
    arg_parser.add_argument(
        "--ocrw",
        dest="ocrw",
        type=float
    )
    arg_parser.add_argument(
        "--penw",
        dest="penw",
        type=float
    )
    arg_parser.add_argument(
        "--conw",
        dest="conw",
        type=float
    )
    arg_parser.add_argument(
        "--no_aug",
        dest="no_aug",
        action='store_true',
    )
    arg_parser.add_argument(
        "--render",
        dest="render",
        action='store_true',
    )
    arg_parser.add_argument(
        "--lmdb",
        dest="lmdb",
        action='store_true',
    )
    arg_parser.add_argument(
        "--resume",
        dest="resume",
        type=str
    )
    arg_parser.add_argument(
        "--freeze",
        dest="freeze",
        type=str
    )
    arg_parser.add_argument(
        "--encode",
        dest="encode",
        type=str
    )
    arg_parser.add_argument(
        "--penetration",
        dest="penetration",
        action='store_true',
    )
    arg_parser.add_argument(
        "--contact",
        dest="contact",
        action='store_true',
    )
    arg_parser.add_argument(
        "--bg_aug",
        dest="bg_aug",
        action='store_true',
    )
    arg_parser.add_argument(
        "--pa_feat",
        dest="pa_feat",
        action='store_true',
    )
    arg_parser.add_argument(
        "--scale_aug",
        dest="scale_aug",
        action='store_true',
    )
    arg_parser.add_argument(
        "--backbone",
        dest="backbone",
        type=str
    )


def update_exp_cfg(cfg, args):
    if args.batch_size is not None:
        cfg['ScenesPerBatch'] = args.batch_size

    if args.num_points is not None:
        cfg['SamplesPerScene'] = args.num_points

    if args.image_size is not None:
        cfg['ImageSize'] = [args.image_size, args.image_size]

    if args.latent_size is not None:
        cfg['LatentSize'] = args.latent_size

    if args.pose_size is not None:
        cfg['PoseFeatSize'] = args.pose_size

    if args.point_size is not None:
        cfg['PointFeatSize'] = args.point_size
    
    if args.lr is not None or args.lr_interval is not None or args.lr_factor is not None:
        for idx, _ in enumerate(cfg['LearningRateSchedule']):
            if args.lr is not None:
                cfg['LearningRateSchedule'][idx]['Initial'] = args.lr
            
            if args.lr_interval is not None:
                cfg['LearningRateSchedule'][idx]['Interval'] = args.lr_interval

            if args.lr_factor is not None:
                cfg['LearningRateSchedule'][idx]['Factor'] = args.lr_factor

    if args.epoch is not None:
        cfg['NumEpochs'] = args.epoch

    if args.add_epoch is not None:
        cfg['AdditionalLossStart'] = args.add_epoch
    
    if args.random_seed is not None:
        cfg['RandomSeed'] = args.random_seed

    if args.mano:
        cfg['ManoBranch'] = args.mano

    if args.cls:
        cfg['ClassifierBranch'] = args.cls

    if args.depth:
        cfg['DepthBranch'] = args.depth

    if args.obj_pose:
        cfg['ObjectPoseBranch'] = args.obj_pose

    if args.hsw is not None:
        cfg['HandSdfWeight'] = args.hsw

    if args.osw is not None:
        cfg['ObjSdfWeight'] = args.osw

    if args.jw is not None:
        cfg['JointWeight'] = args.jw

    if args.vw is not None:
        cfg['VertWeight'] = args.vw

    if args.srw is not None:
        cfg['ShapeRegWeight'] = args.srw

    if args.prw is not None:
        cfg['PoseRegWeight'] = args.prw

    if args.segw is not None:
        cfg['SegWeight'] = args.segw

    if args.ocw is not None:
        cfg['ObjCenterWeight'] = args.ocw

    if args.ocrw is not None:
        cfg['ObjCornerWeight'] = args.ocrw

    if args.penw is not None:
        cfg['PenetrationLossWeight'] = args.penw

    if args.conw is not None:
        cfg['ContactLossWeight'] = args.conw
    
    if args.no_aug is not None:
        cfg['DisableAug'] = args.no_aug

    if args.render is not None:
        cfg['Render'] = args.render

    if args.lmdb is not None:
        cfg['LMDB'] = args.lmdb

    if args.resume is not None:
        cfg['Resume'] = args.resume

    if args.freeze is not None:
        cfg['Freeze'] = args.freeze

    if args.encode is not None:
        cfg['EncodeStyle'] = args.encode

    if args.penetration is not None:
        cfg['PenetrationLoss'] = args.penetration

    if args.contact is not None:
        cfg['ContactLoss'] = args.contact

    if args.bg_aug is not None:
        cfg['BackgroundAug'] = args.bg_aug

    if args.pa_feat is not None:
        cfg['PixelAlign'] = args.pa_feat

    if args.scale_aug is not None:
        cfg['ScaleAug'] = args.scale_aug

    if args.backbone is not None:
        cfg['Backbone'] = args.backbone

    return cfg


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def kinematic_embedding(xyz, mano_results, num_points_per_scene, point_feat_size, scale_factor, obj_results, encode_style):
    xyz = xyz.reshape((-1, num_points_per_scene, 3))
    batch_size = xyz.shape[0]
    try:
        inverse_func = torch.linalg.inv
    except:
        inverse_func = torch.inverse

    wrist_xyz = (xyz * 2 / scale_factor).unsqueeze(2)
    if encode_style == 'hand' or encode_style == 'both':
        # convert points to the mano coordinate system
        mano_xyz = wrist_xyz + mano_results['rot_center'].unsqueeze(1)
        mano_xyz = mano_xyz.unsqueeze(2)
        one_tensor = torch.ones((batch_size, num_points_per_scene, 1, 1, 1)).to(mano_xyz.device)
        homo_mano_xyz = torch.cat([mano_xyz, one_tensor], axis=4)
    
        # inverse the global transformation
        inverse_global_trans = inverse_func(mano_results['global_trans']).unsqueeze(1)
        inverse_homo_mano_xyz = torch.matmul(inverse_global_trans, homo_mano_xyz.transpose(3, 4)).transpose(3, 4)
        inverse_homo_mano_xyz = inverse_homo_mano_xyz.squeeze(3)
        inverse_mano_xyz = inverse_homo_mano_xyz[:, :, :, :3] / inverse_homo_mano_xyz[:, :, :, [3]]

        # choice of dimension of the embedding
        if (point_feat_size == 6 and encode_style == 'hand') or (point_feat_size == 9 and encode_style == 'both'):
            inverse_mano_xyz = inverse_mano_xyz[:, :, :1, :]

        # generate hand embedding
        hand_embedding = torch.cat([mano_xyz.squeeze(2), inverse_mano_xyz], 2)
        if encode_style == 'both':
            hand_embedding = hand_embedding.reshape((batch_size, num_points_per_scene, point_feat_size - 3))
        else:
            hand_embedding = hand_embedding.reshape((batch_size, num_points_per_scene, point_feat_size))
        hand_embedding = hand_embedding * scale_factor / 2
    
    if encode_style == 'obj' or encode_style == 'both':
        one_tensor = torch.ones((batch_size, num_points_per_scene, 1)).to(wrist_xyz.device)
        homo_wrist_xyz = torch.cat([wrist_xyz.squeeze(2), one_tensor], axis=2)
        # inverse the object transformation
        obj_inv_trans = inverse_func(obj_results['obj_trans'])
        obj_embedding = torch.matmul(obj_inv_trans, homo_wrist_xyz.transpose(2, 1)).transpose(2, 1)
        obj_embedding = obj_embedding[:, :, :3] / obj_embedding[:, :, [3]]
        obj_embedding = obj_embedding * scale_factor / 2
        obj_embedding = torch.cat([xyz, obj_embedding], 2)
    
    if encode_style == 'hand':
        kinematic_embedding = hand_embedding.reshape((-1, point_feat_size))
    
    if encode_style == 'obj':
        kinematic_embedding = obj_embedding.reshape((-1, point_feat_size))

    if encode_style == 'both':
        kinematic_embedding = torch.cat([hand_embedding, obj_embedding[:, :, 3:]], axis=2)
        kinematic_embedding = kinematic_embedding.reshape((-1, point_feat_size))
    
    return kinematic_embedding


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def soft_argmax(heatmaps):
    depth_dim = heatmaps.shape[1] // 1
    H_heatmaps = heatmaps.shape[2]
    W_heatmaps = heatmaps.shape[3]
    heatmaps = heatmaps.reshape((-1, 1, depth_dim * H_heatmaps * W_heatmaps))
    heatmaps = F.softmax(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, 1, depth_dim, H_heatmaps, W_heatmaps))

    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(64).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(64).float().cuda()[None, None, :]
    accu_z = accu_z * torch.arange(64).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


def get_obj_trans(obj_center, mano_results, cam_intr, use_obj_rot):
    obj_center[:, :, :2] *= 4
    obj_center[:, :, 2] = (obj_center[:, :, 2] / 64.0 * 2 - 1) * 0.28 + mano_results['center3d'][:, :, 2]
    obj_center = obj_center[:, 0, :]

    fx = cam_intr[:, 0, 0]
    fy = cam_intr[:, 1, 1]
    cx = cam_intr[:, 0, 2]
    cy = cam_intr[:, 1, 2]

    cam_obj_center_x = ((obj_center[:, 0] - cx) / fx * obj_center[:, 2]).unsqueeze(1)
    cam_obj_center_y = ((obj_center[:, 1] - cy) / fy * obj_center[:, 2]).unsqueeze(1)
    cam_obj_center_z = obj_center[:, [2]]
    cam_obj_center = torch.cat([cam_obj_center_x, cam_obj_center_y, cam_obj_center_z], 1)
    obj_t = cam_obj_center - mano_results['center3d'].squeeze()

    if use_obj_rot:
        obj_rot = mano_results['obj_rot']
        obj_trans = torch.zeros((obj_t.shape[0], 4, 4)).to(obj_t.device)
        obj_trans[:, 3, 3] = 1.
        obj_trans[:, :3, :3] = obj_rot
        obj_trans[:, :3, 3] = obj_t
    else:
        obj_trans = torch.zeros((obj_t.shape[0], 4, 4)).to(obj_t.device)
        obj_trans[:, 3, 3] = 1.
        obj_trans[:, :3, :3] = torch.eye(3).to(obj_t.device)
        obj_trans[:, :3, 3] = obj_t

    return obj_trans, cam_obj_center


def get_nerf_embedder(multires):
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def pixel_alignment(img_feat, xyz, cam_intr, mano_results, image_size, scale_factor):
    pred_root = mano_results['joints'][:, [0]]
    xyz = xyz.reshape((img_feat.shape[0], -1, 3))
    xyz_cam = (xyz * 2 / scale_factor) + pred_root
    batch_size = img_feat.shape[0]
    num_points_per_scene = xyz.shape[1]

    one_tensor = torch.ones([batch_size, num_points_per_scene, 1]).to(xyz.device)
    homo_xyz_cam = torch.cat([xyz_cam, one_tensor], 2)
    xy_img = torch.bmm(cam_intr, homo_xyz_cam.transpose(1, 2)).transpose(1, 2)
    xy_img = (xy_img[:, :, :2] / xy_img[:, :, [2]]).unsqueeze(2)

    # scale image coordinate into [-1, 1]
    uv_coord = xy_img / image_size * 2 - 1
    sample_feat = torch.nn.functional.grid_sample(img_feat, uv_coord, align_corners=True, mode='bicubic')[:, :, :, 0].transpose(1, 2)
    uv_coord = uv_coord.squeeze().reshape((-1, 2))

    in_img_mask = (uv_coord[:, 0] >= -1.0) & (uv_coord[:, 0] <= 1.0) & (uv_coord[:, 1] >= -1.0) & (uv_coord[:, 1] <= 1.0)
    out_img_mask = (~in_img_mask).reshape((batch_size, num_points_per_scene, -1))
    sample_feat[torch.where(out_img_mask)[:2]] = img_feat.mean(3).mean(2)[torch.where(out_img_mask)[:1]]
    sample_feat = sample_feat.reshape((batch_size * num_points_per_scene, -1))

    return sample_feat


def decode_sdf_multi_output(decoder, latent_vector, queries, mano_results, cam_intr, specs):
    num_samples = queries.shape[0]
    if specs['PixelAlign']:
        latent_repeat = pixel_alignment(latent_vector, queries[:, :3], cam_intr, mano_results, specs['ImageSize'][0], specs['SdfScaleFactor'])
        inputs = torch.cat([latent_repeat, queries], 1)
        sdf_hand, sdf_obj, predicted_class = decoder(inputs)
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)
        sdf_hand, sdf_obj, predicted_class = decoder(inputs)

    return sdf_hand, sdf_obj, predicted_class


def decode_model_output(model, input, meta_input, specs):
    sdf_feat, mano_feat, aux_feat = model.module.encoder(input)

    mano_results = None
    if specs['ManoBranch']:
        mano_results = model.module.mano_decoder(mano_feat, dict(cam_intr=meta_input['cam_intr'], mano_root=meta_input['mano_root']))
        mano_verts = mano_results['verts'][0].cpu().numpy()
        mano_joints = mano_results['joints'][0].cpu().numpy()
        mano_shape = mano_results['shape'][0].cpu().numpy()
        mano_pcas = mano_results['pcas'][0].cpu().numpy()
        mano_para = dict(joints=mano_joints.tolist(), vertices=mano_verts.tolist(), shape=mano_shape.tolist(), pose=mano_pcas.tolist())
        with open(os.path.join(meta_input['pred_mano_dir'], meta_input['prefix'] + '.json'), 'w') as f:
            json.dump(mano_para, f)
        mano_output_mesh = trimesh.Trimesh(vertices=mano_verts, faces=meta_input['mano_face'], process=False)
        mano_output_mesh.export(os.path.join(meta_input['pred_mano_dir'], meta_input['prefix'] + '.ply'))
    
    obj_results = None
    if specs['ObjectPoseBranch'] and specs['ManoBranch']:
        heatmaps_obj = model.module.volume_layer(aux_feat)
        obj_center_2d = soft_argmax(heatmaps_obj)
        obj_trans, _ = get_obj_trans(obj_center_2d, mano_results, meta_input['cam_intr'], specs['ObjCornerWeight']>0)
        obj_results = dict(obj_trans=obj_trans)
        homo_obj_corners = torch.ones((obj_trans.shape[0], 8, 4)).to(input.device)
        homo_obj_corners[:, :, :3] = meta_input['obj_rest_corners']
        homo_obj_corners = torch.matmul(obj_trans, homo_obj_corners.transpose(2, 1)).transpose(2, 1)
        obj_corners = homo_obj_corners[:, :, :3] / homo_obj_corners[:, :, [3]]
        obj_corners = obj_corners + mano_results['center3d']
        obj_corners = obj_corners[0].cpu().numpy()
        obj_transform = obj_trans.clone()
        obj_transform = obj_transform[0].cpu().numpy()

        obj_rest_mesh = trimesh.load(os.path.join('data/' + specs['Dataset'], 'test/mesh_obj_rest/', meta_input['prefix'] + '.obj'), process=False)
        obj_face = obj_rest_mesh.faces
        obj_verts = obj_rest_mesh.vertices
        homo_obj_verts = np.ones((obj_verts.shape[0], 4))
        homo_obj_verts[:, :3] = obj_verts
        trans_obj_verts = np.dot(obj_transform, homo_obj_verts.transpose(1, 0)).transpose(1, 0)
        trans_obj_verts = trans_obj_verts[:, :3] / trans_obj_verts[:, [3]]
        trans_obj_verts = trans_obj_verts + mano_results['center3d'][0].cpu().numpy()
        trans_obj_mesh = trimesh.Trimesh(vertices=trans_obj_verts, faces=obj_face)
        trans_obj_mesh.export(os.path.join(meta_input['obj_dir'], meta_input['prefix'] + '.obj'))
        obj_para = dict(obj_corners=obj_corners.tolist(), obj_trans=obj_transform.tolist())
        with open(os.path.join(meta_input['obj_dir'], meta_input['prefix'] + '.json'), 'w') as f:
            json.dump(obj_para, f)

    if specs['PixelAlign']:
        latent = aux_feat
    else:
        latent = sdf_feat

    return latent, mano_results, obj_results