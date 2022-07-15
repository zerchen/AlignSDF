#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
from doctest import debug
import logging
import json
from time import process_time
import numpy as np
import os
import shutil
from multiprocessing import Process, Queue
import trimesh
from tqdm import tqdm
import pickle
import deep_sdf


def evaluate(queue, experiment_directory, data_dir, start_point, end_point, optim, mano, optim_mano, fit, rot, obj, task):
    if 'obman' in task or 'ho3d' in task:
        cam_extr = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    else:
        cam_extr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    pred_mesh_path = os.path.join(experiment_directory, 'Eval_' + task, 'meshes')
    pred_mano_path = os.path.join(experiment_directory, 'Eval_' + task, 'pred_mano')
    if optim_mano:
        pred_mano_path = pred_mano_path.replace('pred_mano', 'optim_mano')

    if fit:
        all_pred_filenames = [filename for filename in os.listdir(pred_mesh_path) if '_hand.ply' in filename]
    else:
        if mano:
            all_pred_filenames = [filename for filename in os.listdir(pred_mano_path) if '.ply' in filename]
        else:
            if obj:
                all_pred_filenames = [filename for filename in os.listdir(pred_mesh_path) if '_obj.ply' in filename]
            else:
                all_pred_filenames = [filename for filename in os.listdir(pred_mesh_path) if '_hand.ply' in filename]

    if mano:
        all_pred_filenames = [filename.split('.')[0] for filename in all_pred_filenames]
    else:
        all_pred_filenames = [filename.split('_')[0] for filename in all_pred_filenames]

    for filename in tqdm(all_pred_filenames[start_point:end_point]):
        if fit:
            reconstructed_mesh_filename = os.path.join(pred_mesh_path, filename + '_hand.ply')
            groundtruth_mesh_filename = os.path.join(pred_mano_path, filename + '_hand.ply')
        else:
            if mano:
                reconstructed_mesh_filename = os.path.join(pred_mano_path, filename + '.ply')
                groundtruth_mesh_filename = os.path.join(data_dir, 'mesh_hand', filename + '.obj')
            else:
                if obj:
                    reconstructed_mesh_filename = os.path.join(pred_mesh_path, filename + '_obj.ply')
                    groundtruth_mesh_filename = os.path.join(data_dir, 'mesh_obj', filename + '.obj')
                else:
                    reconstructed_mesh_filename = os.path.join(pred_mesh_path, filename + '_hand.ply')
                    groundtruth_mesh_filename = os.path.join(data_dir, 'mesh_hand', filename + '.obj')

        if os.path.exists(groundtruth_mesh_filename) and os.path.exists(reconstructed_mesh_filename):
            try:
                chamfer_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(groundtruth_mesh_filename, reconstructed_mesh_filename, optim, rot)
            except:
                continue
            if mano:
                try:
                    with open(os.path.join(pred_mano_path, filename + '.json'), 'r') as f:
                        pred_mano_data = json.load(f)
                    pred_joints = np.array(pred_mano_data['joints'])
                    pred_verts = np.array(pred_mano_data['vertices'])

                    all_gt_mano_filename = os.path.join(data_dir, 'meta', filename + '.pkl')
                    with open(all_gt_mano_filename, 'rb') as f:
                        gt_mano_data = pickle.load(f)
                    gt_joints = cam_extr.dot(gt_mano_data['coords_3d'].transpose(1, 0)).transpose(1, 0)
                    gt_verts = cam_extr.dot(gt_mano_data['verts_3d'].transpose(1, 0)).transpose(1, 0)

                    pred_verts = pred_verts - pred_joints[0]
                    pred_joints = pred_joints - pred_joints[0]
                    gt_verts = gt_verts - gt_joints[0]
                    gt_joints = gt_joints - gt_joints[0]

                    joints_dist = np.mean(np.linalg.norm(gt_joints - pred_joints, axis=1))
                    verts_dist = np.mean(np.linalg.norm(gt_verts - pred_verts, axis=1))
                except:
                    joints_dist = 0
                    verts_dist = 0
            elif obj:
                try:
                    with open(os.path.join(pred_mesh_path.replace('meshes', 'object'), filename + '.json'), 'r') as f:
                        pred_obj_data = json.load(f)
                    pred_obj_center = np.array(pred_obj_data['obj_trans'])[:3, 3]
                    pred_obj_corners = np.array(pred_obj_data['obj_corners'])

                    all_gt_obj_filename = os.path.join(data_dir, 'meta', filename + '.pkl')
                    with open(all_gt_obj_filename, 'rb') as f:
                        gt_obj_data = pickle.load(f)
                    gt_obj_center = cam_extr.dot(gt_obj_data['affine_transform'][:3, 3] - gt_obj_data['coords_3d'][0, :])
                    gt_obj_corners = cam_extr.dot(gt_obj_data['obj_corners_3d'][1:, :].transpose(1, 0)).transpose(1, 0)

                    joints_dist = np.linalg.norm(gt_obj_center - pred_obj_center)
                    verts_dist = np.mean(np.linalg.norm(gt_obj_corners - pred_obj_corners, axis=1))
                except:
                    joints_dist = 0
                    verts_dist = 0
            else:
                joints_dist = 0.
                verts_dist = 0.

            queue.put([(filename, chamfer_dist, joints_dist, verts_dist)])


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a sdf network")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--task",
        "-t",
        dest="task",
        default='obman',
        help="the default task",
    )
    arg_parser.add_argument(
        "--num_proc",
        dest="num_proc",
        default=10,
        type=int,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--optim",
        dest="optim",
        action='store_true',
        help="already aligned pred mesh with gt",
    )
    arg_parser.add_argument(
        "--obj",
        dest="obj",
        action='store_true',
        help="cal obj mesh distance",
    )
    arg_parser.add_argument(
        "--mano",
        dest="mano",
        action='store_true',
        help="eval the mesh predicted by the mano branch",
    )
    arg_parser.add_argument(
        "--optim_mano",
        dest="optim_mano",
        action='store_true',
        help="whether to evaluate the optimized mano",
    )
    arg_parser.add_argument(
        "--fit",
        dest="fit",
        action='store_true',
        help="eval the consistency between mano mesh and sdf mesh",
    )
    arg_parser.add_argument(
        "--rot",
        dest="rot",
        action='store_true',
        help="whether to use rot to fit the pred mesh and gt",
    )

    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    q = Queue()
    num_proc = args.num_proc
    data_source = f'data/{args.task}/test/'

    pred_mesh_path = os.path.join(args.experiment_directory, 'Eval_' + args.task, 'meshes')
    pred_mano_path = os.path.join(args.experiment_directory, 'Eval_' + args.task, 'pred_mano')
    if args.optim_mano:
        pred_mano_path = pred_mano_path.replace('pred_mano', 'optim_mano')

    if args.fit:
        all_pred_filenames = [filename for filename in os.listdir(pred_mesh_path) if '_hand.ply' in filename]
    else:
        if args.mano:
            all_pred_filenames = [filename for filename in os.listdir(pred_mano_path) if '.ply' in filename]
        else:
            if args.obj:
                all_pred_filenames = [filename for filename in os.listdir(pred_mesh_path) if '_obj.ply' in filename]
            else:
                all_pred_filenames = [filename for filename in os.listdir(pred_mesh_path) if '_hand.ply' in filename]

    division = len(all_pred_filenames) // num_proc

    start_points = []
    end_points = []
    for i in range(num_proc):
        start_point = i * division
        if i != num_proc - 1:
            end_point = start_point + division
        else:
            end_point = len(all_pred_filenames)
        start_points.append(start_point)
        end_points.append(end_point)
    
    process_list = []
    for i in range(num_proc):
        p = Process(target=evaluate, args=(q, args.experiment_directory, data_source, start_points[i], end_points[i],args.optim, args.mano, args.optim_mano, args.fit, args.rot, args.obj, args.task))
        p.start()
        process_list.append(p)

    summary = []
    for p in process_list:
        while p.is_alive():
            while False == q.empty():
                data = q.get()
                summary = summary + data
    
    for p in process_list:
        p.join()
    
    summary = sorted(summary, reverse=True, key=lambda result: result[1])
    if args.mano:
        best_dir = pred_mesh_path.replace('meshes', 'best_mano')
        worst_dir = pred_mesh_path.replace('meshes', 'worst_mano')
    else:
        if args.obj:
            best_dir = pred_mesh_path.replace('meshes', 'best_obj')
            worst_dir = pred_mesh_path.replace('meshes', 'worst_obj')
        else:
            best_dir = pred_mesh_path.replace('meshes', 'best_hand')
            worst_dir = pred_mesh_path.replace('meshes', 'worst_hand')

    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    if args.fit:
        summary_filename = "fit.txt"
    else:
        if args.mano:
            summary_filename = "chamfer_mano.txt" 
        else:
            if args.obj:
                summary_filename = "chamfer_obj.txt" 
            else:
                summary_filename = "chamfer_hand.txt" 

    with open(os.path.join(args.experiment_directory, 'Eval_' + args.task, summary_filename), "w") as f:
        f.write("summary of chamfer_dist\n")
        chamfer_stat = []
        joints_stat = []
        verts_stat = []
        for idx, result in enumerate(summary):
            chamfer_stat.append(result[1])
            joints_stat.append(result[2])
            verts_stat.append(result[3])
            f.write("{}, {}, {}, {}\n".format(result[0], result[1], result[2] * 1000, result[3] * 1000))
            if not args.fit:
                if idx < 20 or idx > len(summary) - 21:
                    if args.obj:
                        gt_mesh_file = os.path.join(data_source, 'mesh_obj', result[0] + '.obj')
                    else:
                        gt_mesh_file = os.path.join(data_source, 'mesh_hand', result[0] + '.obj')

                    if args.mano:
                        pred_hand_file = os.path.join(pred_mano_path, result[0] + '.ply')
                        pred_obj_file = os.path.join(pred_mesh_path, result[0] + '_obj.ply')
                    else:
                        if args.obj:
                            pred_obj_file = os.path.join(pred_mesh_path, result[0] + '_obj.ply')
                            pred_hand_file = os.path.join(pred_mesh_path, result[0] + '_hand.ply')
                        else:
                            pred_hand_file = os.path.join(pred_mesh_path, result[0] + '_hand.ply')
                            pred_obj_file = os.path.join(pred_mesh_path, result[0] + '_obj.ply')

                    input_img_file = os.path.join(data_source, 'rgb', result[0] + '.jpg')
                    if idx < 20:
                        shutil.copy2(gt_mesh_file, worst_dir)
                        try:
                            shutil.copy2(pred_hand_file, worst_dir)
                            shutil.copy2(pred_obj_file, worst_dir)
                        except:
                            pass
                        shutil.copy2(input_img_file, worst_dir)
                    else:
                        shutil.copy2(gt_mesh_file, best_dir)
                        try:
                            shutil.copy2(pred_hand_file, best_dir)
                            shutil.copy2(pred_obj_file, best_dir)
                        except:
                            pass
                        shutil.copy2(input_img_file, best_dir)

        overall_mean = "mean chamfer distance:{}\n".format(np.mean(chamfer_stat)) 
        overall_median = "median chamfer distance:{}\n".format(np.median(chamfer_stat))
        if args.obj:
            overall_mpjpe = "mean obj center error:{}\n".format(np.mean(joints_stat) * 1000)  # in millimeters
            overall_mpvpe = "mean obj corners error:{}\n".format(np.mean(verts_stat) * 1000)  # in millimeters
        else:
            overall_mpjpe = "mean joints error:{}\n".format(np.mean(joints_stat) * 1000)  # in millimeters
            overall_mpvpe = "mean verts error:{}\n".format(np.mean(verts_stat) * 1000)  # in millimeters
        print(overall_mean)
        print(overall_median)
        print(overall_mpjpe)
        print(overall_mpvpe)
        f.write(overall_mean)
        f.write(overall_median)
        f.write(overall_mpjpe)
        f.write(overall_mpvpe)

        failure_info = "failure count:{}\n".format(len(all_pred_filenames) - len(summary)) 
        print(failure_info)
        f.write(failure_info)

        all_eval_filenames = [result[0] for result in summary]
        for filename in all_pred_filenames:
            if filename.split('.')[0].split('_')[0] not in all_eval_filenames:
                f.write('{}\n'.format(filename))
