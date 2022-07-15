import os
import argparse
import concurrent.futures
import subprocess
import json


def reconstruct_mano_mesh(model_dir, task, start_point, end_point, use_eval_mode, optim):
    if use_eval_mode:
        if optim:
            cmd = f'python reconstruct.py -e {model_dir} --task {task} --start_point {start_point} --end_point {end_point} --eval_mode --label'
        else:
            cmd = f'python reconstruct.py -e {model_dir} --task {task} --start_point {start_point} --end_point {end_point} --eval_mode'
    else:
        if optim:
            cmd = f'python reconstruct.py -e {model_dir} --task {task} --start_point {start_point} --end_point {end_point} --label'
        else:
            cmd = f'python reconstruct.py -e {model_dir} --task {task} --start_point {start_point} --end_point {end_point}'

    subprocess_env = os.environ.copy()
    subprocess_env['CUDA_VISIBLE_DEVICES'] = gpus_list[idx]

    subproc = subprocess.Popen(cmd, shell=True, env=subprocess_env, stdout=subprocess.PIPE)
    subproc.communicate()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generate meshes in parallel")
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
        default="obman",
        choices=["obman", "dexycb"],
        help="task to perform"
    )
    arg_parser.add_argument(
        "--optim",
        dest="optim",
        action='store_true',
        help="whether to fit sdf to mano"
    )
    args = arg_parser.parse_args()

    gpus_list = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    num_gpus = len(gpus_list)

    if args.task == 'obman':
        image_source = 'input/obman.json'
        use_eval_mode = True
    elif args.task == 'dexycb':
        image_source = 'input/dexycb.json'
        use_eval_mode = True

    all_filenames = json.load(open(image_source))['filenames']
    division = len(all_filenames) // num_gpus

    start_points = []
    end_points = []
    for i in range(num_gpus):
        start_point = i * division
        if i != num_gpus - 1:
            end_point = start_point + division
        else:
            end_point = len(all_filenames)

        start_points.append(start_point)
        end_points.append(end_point)

    model_dir = args.experiment_directory

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        for idx, (start, end) in enumerate(zip(start_points, end_points)):
            print(f'Subprocess for {model_dir} using GPU {gpus_list[idx]} from {start} to {end - 1}')
            executor.submit(reconstruct_mano_mesh, model_dir, args.task, start, end, use_eval_mode, args.optim)
        executor.shutdown()