#!/usr/bin/env python3
# Copyright 2004-2019 Facebook. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import pickle
from tqdm import tqdm

import deep_sdf
import deep_sdf.workspace as ws

import shutil


def process_mesh(mesh_filepath_hand,
                 mesh_filepath_obj, 
                 target_filepath_hand,
                 target_filepath_obj, 
                 norm_filepath,
                 executable):
    logging.info(mesh_filepath_hand + " --> " + target_filepath_hand + " and obj")

    command = [
        executable,
        "--hand",
        mesh_filepath_hand,
        "--obj",
        mesh_filepath_obj,
        "--outhand",
        target_filepath_hand,
        "--outobj",
        target_filepath_obj,
        "--normalize",
        norm_filepath,
    ]

    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    subproc.wait()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to a dataset.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        default='/data/zerui/dexycb_processed/train',
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)
    additional_general_args = []

    executable = "bin/PreprocessMesh"
    extension = ".npz"

    logging.info(f"Start to preprocess data from {args.source_dir.split('/')[-1]}")

    meshes_targets_and_specific_args = []

    hmesh_path = os.path.join(args.source_dir, 'mesh_hand')
    omesh_path = os.path.join(args.source_dir, 'mesh_obj')
    meta_path = os.path.join(args.source_dir, 'meta')

    hsdf_path = os.path.join(args.source_dir, 'sdf_hand')
    osdf_path = os.path.join(args.source_dir, 'sdf_obj')
    norm_path = os.path.join(args.source_dir, 'norm')

    os.makedirs(hsdf_path, exist_ok=True)
    os.makedirs(osdf_path, exist_ok=True)
    os.makedirs(norm_path, exist_ok=True)

    hand_meshes = os.listdir(hmesh_path)
    right_hand_meshes = []
    for idx, hmesh in tqdm(enumerate(hand_meshes)):
        abs_hmesh_path = os.path.join(hmesh_path, hmesh)
        abs_meta_path = abs_hmesh_path.replace('mesh_hand', 'meta').replace('obj', 'pkl')
        with open(abs_meta_path, 'rb') as f:
            meta_data = pickle.load(f)
        if meta_data['side'] == 'right':
            right_hand_meshes.append(hmesh)
    
    print('To generate {} samples !!!!!!'.format(len(right_hand_meshes)))

    for idx, hmesh in enumerate(right_hand_meshes):
        abs_hmesh_path = os.path.join(hmesh_path, hmesh)
        abs_omesh_path = abs_hmesh_path.replace('mesh_hand', 'mesh_obj')
        if not os.path.isfile(abs_omesh_path):
            continue

        abs_hsdf_path = os.path.join(hsdf_path, hmesh.split('.')[0] + extension)
        abs_osdf_path = abs_hsdf_path.replace('sdf_hand', 'sdf_obj')
        abs_norm_path = os.path.join(norm_path, hmesh.split('.')[0] + extension)

        meshes_targets_and_specific_args.append(
                (
                    abs_hmesh_path,
                    abs_omesh_path,
                    abs_hsdf_path,
                    abs_osdf_path,
                    abs_norm_path,
                )
        )
    

    print(" Start sampling using C++")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:
        for (
            abs_hmesh_path,
            abs_omesh_path,
            abs_hsdf_path,
            abs_osdf_path,
            abs_norm_path,
        ) in meshes_targets_and_specific_args:
            executor.submit(
                process_mesh,
                abs_hmesh_path,
                abs_omesh_path,
                abs_hsdf_path,
                abs_osdf_path,
                abs_norm_path,
                executable,
            )

        executor.shutdown()
