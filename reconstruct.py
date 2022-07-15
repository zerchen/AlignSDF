from distutils.log import debug
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import numpy as np
import json
import pickle
import time
import cv2
from tqdm import tqdm
import trimesh
import utils
import networks.model as arch
import utils.misc as misc_utils
from networks.model_utils import get_model

def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def reconstruct(loaded_model, specs, split_filename, output_dir, start_point, end_point, task="obman", device="cpu", scale=None, cube_dim=128, label_out=False, viz=False, eval_mode=False):
    output_mesh_dir = os.path.join(output_dir, "meshes")
    os.makedirs(output_mesh_dir, exist_ok=True)

    output_pred_mano_dir = os.path.join(output_dir, "pred_mano")
    os.makedirs(output_pred_mano_dir, exist_ok=True)

    output_optim_mano_dir = os.path.join(output_dir, "optim_mano")
    os.makedirs(output_optim_mano_dir, exist_ok=True)

    output_mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(output_mask_dir, exist_ok=True)

    output_obj_dir = os.path.join(output_dir, "object")
    os.makedirs(output_obj_dir, exist_ok=True)

    with open(split_filename, "r") as f:
        input_list = json.load(f)["filenames"][int(start_point):int(end_point)]
    
    mano_face_right = np.load('mano/closed_fmano.npy')
    
    dataset = utils.data.ImagesInput(input_list, specs=specs, task=task)

    # load data
    num_data_loader_threads = 1
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    data_loader = data_utils.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )
    loaded_model.eval()

    tqdm_iter = tqdm(data_loader, total=len(data_loader), leave=False)
    for batch in tqdm_iter:
        input_iter, meta_iter = batch
        tqdm_iter.set_description('Process {}'.format(meta_iter['img_id'][0].split('/')[-1].split('.')[0]))

        encoder_input = input_iter['img'].to(device)
        cam_intr = meta_iter['cam_intr'].to(device)
        mano_root = meta_iter['mano_root'].to(device)
        obj_rest_corners = meta_iter['obj_rest_corners'].to(device)
        save_filename_prefix = meta_iter['img_id'][0].split('/')[-1].split('.')[0]

        meta_input = dict(cam_intr=cam_intr, mano_root=mano_root, mano_face=mano_face_right, obj_rest_corners=obj_rest_corners, pred_mano_dir=output_pred_mano_dir, optim_mano_dir=output_optim_mano_dir, mask_dir=output_mask_dir, obj_dir=output_obj_dir, prefix=save_filename_prefix)

        mano_results = None
        with torch.no_grad():
            latent, mano_results, obj_results = utils.decode_model_output(loaded_model, encoder_input, meta_input, specs)

        out_filename = save_filename_prefix
        mesh_filename = os.path.join(output_mesh_dir, out_filename)
        
        obj_branch = specs['ObjectBranch']
        hand_branch = specs['HandBranch']
        cls_branch = specs['ClassifierBranch']
        with torch.no_grad():
            utils.mesh.create_mesh_combined_decoder(hand_branch, obj_branch, cls_branch, loaded_model.module.decoder, latent, mano_results, obj_results, cam_intr, specs, mesh_filename, N=cube_dim, max_batch=int(2 ** 18), scale=scale, device=device, label_out=label_out, viz=viz, eval_mode=eval_mode, task=task)

        del latent


def get_default_args(args):
    if args.task == "obman":
        args.split_filename = "input/obman.json"
    elif args.task == "dexycb":
        args.split_filename = "input/dexycb.json"

    return args


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Use a trained GraspingField model to reconstruct hand shapes given "
        + " object shapes."
    )
    arg_parser.add_argument(
        "--model",
        "-e",
        dest="model_directory",
        default="./pretrained_model",
        help="The experiment directory which includes specifications and pretrained model",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        default="input.json",
        help="The json file containin a list of inputs.",
    )
    arg_parser.add_argument(
        '--label',
        dest='label_out',
        action='store_true',
        help="If true, output npy files containing hand-part label for each points "
        + "in the output meshes. Required for MANO fitting."
    )
    arg_parser.add_argument(
        '--viz', dest='viz', 
        action='store_true',
        help="If true, output easy-to-visualized obj files containing hand-part labels"
    )
    arg_parser.add_argument(
        "--task",
        "-t",
        dest="task",
        default="obman",
        choices=["obman", "dexycb"],
        help="task to perform"
    )
    arg_parser.add_argument(
        "--start_point",
        dest='start_point'
    )
    arg_parser.add_argument(
        "--end_point",
        dest="end_point"
    )
    arg_parser.add_argument(
        "--eval_mode",
        dest="eval_mode",
        action='store_true',
        help="If true, optimize scale and trans with regard to gt"
    )

    args = arg_parser.parse_args()
    args = get_default_args(args)
    output_dir = os.path.join(args.model_directory, 'Eval_' + args.task)
    os.makedirs(output_dir, exist_ok=True)

    specs_filename = os.path.join(args.model_directory, "specs.json")
    specs = json.load(open(specs_filename))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaded_model = get_model(os.path.join(args.model_directory, 'ModelParameters'), specs, device)

    if args.start_point is None or args.end_point is None:
        with open(args.split_filename, 'r') as f:
            input_filenames = json.load(f)['filenames']
        args.start_point = 0
        args.end_point = len(input_filenames)
    
    reconstruct(loaded_model, specs, args.split_filename, output_dir, start_point=args.start_point, end_point=args.end_point, task=args.task, device=device, cube_dim=128, label_out=args.label_out, viz=args.viz, eval_mode=args.eval_mode)