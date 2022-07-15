from distutils.log import debug
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import torch
import signal
import sys
import os
import logging
import numpy as np
import json
import time
import random
import subprocess
import re
import socket

from torch.utils.tensorboard import SummaryWriter
from torch import distributed as dist
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as NativeDDP
import utils
import networks.model as arch
import networks.manobranch as mano
from reconstruct import reconstruct

logger = logging.getLogger(__name__)


def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    else:
        logger.warning("Not the master process, no need to requeue.")
    sys.exit(-1)


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    logger.warning("Signal handler installed.")


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class LinearWeightSchedule:
    def __init__(self, start_ep, interval, initial=0.0, target=1.0):
        self.start_ep = start_ep
        self.interval = interval
        self.initial = initial
        self.target = target

    def get_weight(self, epoch):
        if epoch < self.start_ep:
            return self.initial
        return min(self.target, self.initial + (self.target - self.initial) * (epoch - self.start_ep) / self.interval)
    

def get_kl_weight_schedules(specs):

    kl_schedules_specs = specs["KLSchedule"]

    return LinearWeightSchedule(
        kl_schedules_specs["Start"],
        kl_schedules_specs["Interval"],
        0.0,
        get_spec_with_default(kl_schedules_specs, "Target", 0.1)
    )


def get_learning_rate_schedules(specs, world_size):

    schedule_specs_list = specs["LearningRateSchedule"]
    schedules = []

    for schedule_specs in schedule_specs_list:
        scale_factor = 1
        if schedule_specs["Type"] == "Step":
            schedules.append(StepLearningRateSchedule(schedule_specs["Initial"] * scale_factor, schedule_specs["Interval"], schedule_specs["Factor"]))
        else:
            raise Exception('no known learning rate schedule of type "{}"'.format(schedule_specs["Type"]))

    return schedules


def save_model(experiment_directory, filename, model, epoch):

    model_params_dir = utils.misc.get_model_params_dir(experiment_directory, True)

    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, os.path.join(model_params_dir, filename))


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = utils.misc.get_optimizer_params_dir(experiment_directory, True)

    torch.save({"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()}, os.path.join(optimizer_params_dir, filename))


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(utils.misc.get_optimizer_params_dir(experiment_directory), filename)

    if not os.path.isfile(full_filename):
        raise Exception('optimizer state dict "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename, map_location=torch.device('cpu'))

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, utils.misc.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (data["loss"], data["learning_rate"], data["timing"], data["epoch"],)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    host_vectors = np.array([vec.detach().cpu().numpy().squeeze() for vec in latent_vectors])
    return np.mean(np.linalg.norm(host_vectors, axis=1))


def append_parameter_magnitudes(param_mag_log, model, writer, step):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())
        writer.add_scalar(name + 'mag', param.data.norm().item(), step)


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                torch.distributed.broadcast(bn_buf, 0)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def main_function(specs, continue_from, local_rank, opt_level, use_slurm):
    if 'RandomSeed' in specs:
        set_random_seeds(specs['RandomSeed'])

    if use_slurm:
        world_size = int(os.environ['SLURM_NTASKS'])
        assert world_size >= 1
        local_rank = int(os.environ['SLURM_LOCALID'])
        global_rank = int(os.environ['SLURM_PROCID'])

        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        master_addr = hostnames.split()[0].decode('utf-8')

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(global_rank)
        logging.info('Training in distributed mode, 1 GPU per process. Process %d, total %d.' % (local_rank, world_size))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        assert int(os.environ['WORLD_SIZE']) >= 1
        device = 'cuda:%d' % local_rank
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        logging.info('Training in distributed mode, 1 GPU per process. Process %d, total %d.' % (rank, world_size))
        assert rank >= 0

    data_source = specs["DataSource"]
    image_source = specs["ImageSource"]
    train_split_file = specs["TrainSplit"]
    sdf_scale_factor = specs['SdfScaleFactor']

    if local_rank == 0:
        logging.info("Experiment description: \n" + specs["Description"])

    dataset_name = get_spec_with_default(specs, "Dataset", "obman")
    use_lmdb = get_spec_with_default(specs, "LMDB", True)
    resume_checkpoint = get_spec_with_default(specs, "Resume", 'latest.pth')

    ### Model Type
    model_type = get_spec_with_default(specs, "ModelType", "1encoder1decoder")
    backbone = get_spec_with_default(specs, "Backbone", "resnet18")
    hand_branch = get_spec_with_default(specs, "HandBranch", True)
    obj_branch = get_spec_with_default(specs, "ObjectBranch", True)
    mano_branch = get_spec_with_default(specs, "ManoBranch", False)
    obj_pose_branch = get_spec_with_default(specs, "ObjectPoseBranch", False) if mano_branch else False
    obj_center_weight = get_spec_with_default(specs, "ObjCenterWeight", 1) if obj_pose_branch else 0
    obj_corner_weight = get_spec_with_default(specs, "ObjCornerWeight", 0.2) if obj_pose_branch else 0
    depth_branch = get_spec_with_default(specs, "DepthBranch", False) if mano_branch else False
    use_render = get_spec_with_default(specs, "Render", False) if mano_branch else False
    pixel_align = get_spec_with_default(specs, "PixelAlign", False)

    classifier_branch = get_spec_with_default(specs, "ClassifierBranch", False)
    classifier_weight = get_spec_with_default(specs, "ClassifierWeight", 0.005) if classifier_branch else 0
    disable_aug = get_spec_with_default(specs, "DisableAug", False)
    background_aug = get_spec_with_default(specs, "BackgroundAug", False)

    do_penetration_loss = get_spec_with_default(specs, "PenetrationLoss", False)
    penetration_loss_weight = get_spec_with_default(specs, "PenetrationLossWeight", 15.0) if do_penetration_loss else 0
    do_contact_loss = get_spec_with_default(specs, "ContactLoss", False)
    contact_loss_weight = get_spec_with_default(specs, "ContactLossWeight", 0.005) if do_contact_loss else 0
    contact_loss_sigma = get_spec_with_default(specs, "ContactLossSigma", 0.005)
    
    start_additional_loss = get_spec_with_default(specs, "AdditionalLossStart", 1201)
    latent_size = get_spec_with_default(specs, "LatentSize", 256)
    pose_feat_size = get_spec_with_default(specs, "PoseFeatSize", 15)
    point_feat_size = get_spec_with_default(specs, "PointFeatSize", 3)
    encode_style = get_spec_with_default(specs, "EncodeStyle", "nerf")

    hand_sdf_weight = get_spec_with_default(specs, "HandSdfWeight", 1)
    obj_sdf_weight = get_spec_with_default(specs, "ObjSdfWeight", 1)
    joint_weight = get_spec_with_default(specs, "JointWeight", 1) if mano_branch else 0
    vert_weight = get_spec_with_default(specs, "VertWeight", 1) if mano_branch else 0
    shape_weight = get_spec_with_default(specs, "ShapeRegWeight", 1) if mano_branch else 0
    pose_weight = get_spec_with_default(specs, "PoseRegWeight", 1) if mano_branch else 0

    checkpoints = list(range(specs["SnapshotFrequency"], specs["NumEpochs"] + 1, specs["SnapshotFrequency"],))
    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs, world_size)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))
    
    data_info = train_split_file.split('/')[-1].split('.')[:-1][0]
    model_abbr = ''.join(re.findall('\d+', specs['ModelType']))
    model_info = f"m{model_abbr}_{backbone}_mlp{len(specs['NetworkSpecs']['dims'])+1}_pa{int(pixel_align)}_h{int(hand_branch)}_o{int(obj_branch)}_d{int(depth_branch)}_cls{int(classifier_branch)}_mano{int(mano_branch)}_obj{int(obj_pose_branch)}_pose{pose_feat_size}_point{point_feat_size}_{encode_style}"
    train_info = f"e{specs['NumEpochs']}_ae{specs['AdditionalLossStart']}_b{world_size*specs['ScenesPerBatch']}_np{specs['SamplesPerScene']}_ims{specs['ImageSize'][0]}_lr{specs['LearningRateSchedule'][0]['Initial']}_aug{int(not disable_aug)}_bg{int(background_aug)}_hsw{hand_sdf_weight}_osw{obj_sdf_weight}_jw{joint_weight}_vw{vert_weight}_prw{pose_weight}_srw{shape_weight}_ocw{obj_center_weight}_ocrw{obj_corner_weight}_clsw{classifier_weight}_penw{penetration_loss_weight}_conw{contact_loss_weight}"

    experiment_directory = os.path.join('outputs', '_'.join([data_info, model_info, train_info]))
    if local_rank == 0:
        os.makedirs(experiment_directory, exist_ok=True)
        with open(os.path.join(experiment_directory, 'specs.json'), 'w') as f:
            json.dump(specs, f, indent=2)

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", encoderDecoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    # If true, use the data as-is. If false, multiply and offset obj location with normalized params
    indep_obj_scale = get_spec_with_default(specs, "IndependentObjScale", False)
    # Ignore points from other mesh in the begining when train 1 decoder
    ignore_other = get_spec_with_default(specs, "IgnorePointFromOtherMesh", False)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    scene_per_subbatch = scene_per_batch
    clamp_dist = specs["ClampingDistance"]
    input_image_size = tuple(specs["ImageSize"])
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    nb_classes = get_spec_with_default(specs["NetworkSpecs"], "num_class", 6)

    ## Define Model
    if model_type == '1encoder1decoder':
        same_point = True
        filter_dist = True
        latent_size = int(latent_size)

        encoder = arch.get_encoder(backbone, output_size=latent_size, specs=specs)
        if mano_branch:
            mano_decoder = mano.ManoBranch(ncomps=pose_feat_size, absolute_depth=depth_branch, object_pose=obj_pose_branch, use_obj_rot=obj_corner_weight>0)
        else:
            mano_decoder = None

        combined_decoder = arch.CombinedDecoder(latent_size, point_feat_size, encode_style, **specs["NetworkSpecs"], use_classifier=classifier_branch)

        encoder = encoder.cuda()
        combined_decoder = combined_decoder.cuda()

        encoderDecoder = arch.ModelOneEncoderOneDecoder(encoder, combined_decoder, mano_decoder, specs).cuda()
        encoderDecoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoderDecoder)

    elif model_type == '1encoder2decoder':
        same_point = True
        filter_dist = True
        latent_size = int(latent_size)

        encoder = arch.get_encoder(backbone, output_size=latent_size, specs=specs)
        if mano_branch:
            mano_decoder = mano.ManoBranch(ncomps=pose_feat_size, absolute_depth=depth_branch, object_pose=obj_pose_branch, use_obj_rot=obj_corner_weight>0)
        else:
            mano_decoder = None

        separate_decoder = arch.SeparateDecoder(latent_size, point_feat_size, encode_style, **specs["NetworkSpecs"], use_classifier=classifier_branch)

        encoder = encoder.cuda()
        separate_decoder = separate_decoder.cuda()

        encoderDecoder = arch.ModelOneEncoderOneDecoder(encoder, separate_decoder, mano_decoder, specs).cuda()
        encoderDecoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoderDecoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 5)
    log_frequency_step = get_spec_with_default(specs, "LogFrequencyStep", 10)

    if local_rank == 0:
        print("Hand branch:", hand_branch)
        print("Object branch:", obj_branch)
        print("Mano branch:", mano_branch)
        print("Depth branch:", depth_branch)
        print("Classifier Weight:", classifier_weight)
        print("Penetration Loss:", do_penetration_loss)
        print("Penetration Loss Weight:", penetration_loss_weight)
        print("Additional Loss start at epoch:", start_additional_loss)
        print("Contact Loss:", do_contact_loss)
        print("Contact Loss Weight:", contact_loss_weight)
        print("Contact Loss Sigma (m):", contact_loss_sigma)
        print("Independent Obj Scale:", indep_obj_scale)
        print("Ignore other:", ignore_other)
        print("nb_label_class: ", nb_classes)
        print("Image encoder, the branch has latent size", latent_size)

    loss_l1 = torch.nn.L1Loss(reduction='sum')
    loss_l2 = torch.nn.MSELoss()

    criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer_all = torch.optim.Adam([{"params": encoderDecoder.parameters()}])

    encoderDecoder = NativeDDP(encoderDecoder, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank)

    # Tensorboard summary
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(experiment_directory, 'log'))

    start_epoch = 1

    if os.path.isfile(resume_checkpoint):
        data = torch.load(resume_checkpoint, map_location=torch.device('cpu'))
        encoderDecoder.load_state_dict(data["model_state_dict"], strict=False)
        logging.info('Successfully load a pretrained checkpoint')

    # continue from latest checkpoint if exists
    if (continue_from is None and utils.misc.is_checkpoint_exist(experiment_directory, 'latest')):
        continue_from = 'latest'

    if continue_from is not None:
        logging.info('continuing from {} using local rank {}'.format(continue_from, local_rank))
        model_epoch = utils.misc.load_model_parameters(experiment_directory, continue_from, encoderDecoder)
        optimizer_epoch = load_optimizer(experiment_directory, continue_from + ".pth", optimizer_all)
        start_epoch = model_epoch + 1
        logging.debug("loaded")
    
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    
    sdf_dataset = utils.data.SDFSamples(
        data_source,
        train_split,
        num_samp_per_scene,
        dataset_name=dataset_name,
        image_source=image_source,
        hand_branch=hand_branch,
        obj_branch=obj_branch,
        mano_branch=mano_branch,
        depth_branch=depth_branch,
        disable_aug=disable_aug,
        background_aug=background_aug,
        same_point=same_point,
        filter_dist=filter_dist,
        image_size=input_image_size,
        sdf_scale_factor=sdf_scale_factor,
        clamp=clamp_dist,
        model_type=model_type,
        use_lmdb=use_lmdb
    )

    if use_slurm:
        num_data_loader_threads = 10
    else:
        num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 5)
    sdf_loader = utils.loader.create_loader(sdf_dataset, batch_size=scene_per_batch, num_workers=num_data_loader_threads, is_training=True, use_prefetcher=False, pin_memory=True, distributed=True)

    # training loop
    logging.info(f'start_epoch:{start_epoch}, current_rank:{local_rank}')
    if use_slurm:
        init_signal_handler()

    for epoch in range(start_epoch, num_epochs + 1):
        sdf_loader.sampler.set_epoch(epoch)
        start = time.time()
        logging.info(f'epoch:{epoch}, current_rank:{local_rank}')
        encoderDecoder.train()
        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for i, (input_iter, label_iter, meta_iter) in enumerate(sdf_loader):
            optimizer_all.zero_grad()

            if hand_branch and obj_branch:
                samples = torch.cat([label_iter['pc_hand'], label_iter['pc_obj']], 1).cuda()
                labels = torch.cat([label_iter['cls_hand'], label_iter['cls_obj']], 1).cuda()
                # Ignore points from other shape in the begining of the training
                if ignore_other or epoch < start_additional_loss:
                    mask_hand = torch.cat([torch.ones(label_iter['pc_hand'].size()[:2]), torch.zeros(label_iter['pc_obj'].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(num_samp_per_scene * scene_per_subbatch).unsqueeze(1)
                    mask_obj = torch.cat([torch.zeros(label_iter['pc_hand'].size()[:2]), torch.ones(label_iter['pc_obj'].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(num_samp_per_scene * scene_per_subbatch).unsqueeze(1)
                else:
                    mask_hand = torch.ones(num_samp_per_scene * scene_per_subbatch).unsqueeze(1).cuda()
                    mask_obj = torch.cat([torch.ones(label_iter['pc_hand'].size()[:2]), torch.ones(label_iter['pc_obj'].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(num_samp_per_scene * scene_per_subbatch).unsqueeze(1)
            elif hand_branch:
                samples = label_iter['pc_hand'].cuda()
                labels = label_iter['cls_hand'].cuda()
                mask_hand = torch.ones(num_samp_per_scene * scene_per_subbatch).unsqueeze(1).cuda()
            elif obj_branch:
                samples = label_iter['pc_obj'].cuda()
                labels = label_iter['cls_obj'].cuda()
                mask_obj = torch.ones(num_samp_per_scene * scene_per_subbatch).unsqueeze(1).cuda()
            
            samples.requires_grad = False
            labels.requires_grad = False
            
            input_img = input_iter['img'].cuda()
            sdf_data = samples.reshape(num_samp_per_scene * scene_per_subbatch, -1)
            labels = labels.to(torch.long).reshape(num_samp_per_scene * scene_per_subbatch)
            xyz_points = sdf_data[:, 0:-2]

            sdf_gt_hand = sdf_data[:, -2].unsqueeze(1)
            sdf_gt_obj = sdf_data[:, -1].unsqueeze(1)
            cam_intr = meta_iter['cam_intr'].cuda()
            mano_root = meta_iter['mano_root'].cuda()
            rest_obj_corners = meta_iter['rest_obj_corners'].cuda()
            hand_joints_3d = label_iter['hand_joints_3d'].cuda()
            gt_obj_center = label_iter['obj_center'].cuda()
            gt_obj_corners = label_iter['obj_corners'].cuda()

            if use_render:
                gt_mask = label_iter['seg_mask'].cuda()
            
            cond_input = dict(cam_intr=cam_intr, mano_root=mano_root, rest_obj_corners=rest_obj_corners, epoch=epoch)
            
            if enforce_minmax:
                if hand_branch:
                    sdf_gt_hand = torch.clamp(sdf_gt_hand, minT, maxT)
                if obj_branch:
                    sdf_gt_obj = torch.clamp(sdf_gt_obj, minT, maxT)
            
            pred_sdf_hand, pred_sdf_obj, pred_class, mano_results, obj_results = encoderDecoder(input_img, xyz_points, cond_input)

            if enforce_minmax:
                if hand_branch:
                    pred_sdf_hand = torch.clamp(pred_sdf_hand, minT, maxT)
                if obj_branch:
                    pred_sdf_obj = torch.clamp(pred_sdf_obj, minT, maxT)

            ## Compute losses
            if hand_branch:
                mask_hand.requires_grad = False
                loss_hand = hand_sdf_weight * loss_l1(pred_sdf_hand * mask_hand, sdf_gt_hand * mask_hand) / mask_hand.sum()
            else:
                loss_hand = 0.

            if obj_branch:
                mask_obj.requires_grad = False
                loss_obj = obj_sdf_weight * loss_l1(pred_sdf_obj * mask_obj, sdf_gt_obj * mask_obj) / mask_obj.sum()
            else:
                loss_obj = 0.
            
            if mano_branch:
                loss_joint = joint_weight * loss_l2(mano_results['joints'], hand_joints_3d)
                loss_pose = pose_weight * loss_l2(mano_results['pose'], torch.zeros_like(mano_results['pose']))
                loss_shape = shape_weight * loss_l2(mano_results['shape'], torch.zeros_like(mano_results['shape']))
            else:
                loss_joint = 0.
                loss_pose = 0.
                loss_shape = 0.
            
            if obj_pose_branch:
                loss_obj_center = loss_l2(obj_results['obj_center'], gt_obj_center) * obj_center_weight
                loss_obj_corner = loss_l2(obj_results['obj_corners'], gt_obj_corners) * obj_corner_weight
            else:
                loss_obj_center = 0.
                loss_obj_corner = 0.
            
            if classifier_branch:
                if epoch >= start_additional_loss:
                    loss_ce = criterion_ce(pred_class, labels) * classifier_weight
                else:
                    loss_ce = criterion_ce(pred_class, labels) * 0.
            else:
                loss_ce = 0.
            
            if hand_branch:
                scaled_pred_sdf_hand = (pred_sdf_hand * 2.0).reshape((-1, num_samp_per_scene, 1)) / sdf_scale_factor
                scaled_pred_sdf_hand = scaled_pred_sdf_hand.reshape((-1, 1))

            if obj_branch:
                scaled_pred_sdf_obj = (pred_sdf_obj * 2.0).reshape((-1, num_samp_per_scene, 1)) / sdf_scale_factor
                scaled_pred_sdf_obj = scaled_pred_sdf_obj.reshape((-1, 1))

            if do_penetration_loss and epoch >= start_additional_loss:
                loss_pen = torch.max(-(scaled_pred_sdf_hand + scaled_pred_sdf_obj), torch.Tensor([0]).cuda()).mean() * penetration_loss_weight
            else:
                loss_pen = 0.
            
            if do_contact_loss and epoch >= start_additional_loss:
                alpha = 1. / contact_loss_sigma**2
                loss_contact = torch.min(alpha * (scaled_pred_sdf_hand**2 + scaled_pred_sdf_obj**2), torch.Tensor([1]).cuda()).mean() * contact_loss_weight
            else:
                loss_contact = 0.

            loss = loss_hand + loss_obj + loss_joint + loss_pose + loss_shape + loss_ce + loss_pen + loss_contact + loss_obj_center + loss_obj_corner

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(encoderDecoder.parameters(), grad_clip)

            loss_out = reduce_tensor(loss.data, world_size).item()
            loss_hand_out = reduce_tensor(loss_hand.data, world_size).item() if hand_branch else 0
            loss_obj_out = reduce_tensor(loss_obj.data, world_size).item() if obj_branch else 0
            loss_joint_out = reduce_tensor(loss_joint.data, world_size).item() if mano_branch else 0
            loss_pose_out = reduce_tensor(loss_pose.data, world_size).item() if mano_branch else 0
            loss_shape_out = reduce_tensor(loss_shape.data, world_size).item() if mano_branch else 0
            loss_oc_out = reduce_tensor(loss_obj_center.data, world_size).item() if obj_pose_branch else 0
            loss_ocr_out = reduce_tensor(loss_obj_corner.data, world_size).item() if obj_pose_branch else 0
            loss_ce_out = reduce_tensor(loss_ce.data, world_size).item() if classifier_branch and epoch >= start_additional_loss else 0
            loss_pen_out = reduce_tensor(loss_pen.data, world_size).item() if do_penetration_loss and epoch >= start_additional_loss else 0
            loss_contact_out = reduce_tensor(loss_contact.data, world_size).item() if do_contact_loss and epoch >= start_additional_loss else 0

            if local_rank == 0 and ((epoch - 1) * len(sdf_loader) + i) % log_frequency_step == 0:
                logging.info('step {}, hsdf {:.5f}, osdf {:.5f}, joint {:.5f}, pose {:.5f}, shape {:.5f}, cls {:.5f}, center {:.5f}, corner {:.5f}'.format((epoch-1) * len(sdf_loader) + i, loss_hand_out * 1000, loss_obj_out * 1000, loss_joint_out * 1000, loss_pose_out * 1000, loss_shape_out * 1000, loss_ce_out * 1000, loss_oc_out * 1000, loss_ocr_out * 1000))
                    
                writer.add_scalar('training_loss_1e-3', loss_out * 1000.0, (epoch-1) * len(sdf_loader) + i)
                writer.add_scalar('loss_hand_1e-3', loss_hand_out * 1000.0, (epoch-1) * len(sdf_loader) + i)
                writer.add_scalar('loss_obj_1e-3', loss_obj_out * 1000.0, (epoch-1) * len(sdf_loader) + i)

                if mano_branch:
                    writer.add_scalar('loss_pose_1e-3', loss_pose_out * 1000.0, (epoch-1) * len(sdf_loader) + i)
                    writer.add_scalar('loss_joint_1e-3', loss_joint_out * 1000.0, (epoch-1) * len(sdf_loader) + i)
                    writer.add_scalar('loss_shape_1e-3', loss_shape_out * 1000.0, (epoch-1) * len(sdf_loader) + i)

                if obj_pose_branch:
                    writer.add_scalar('loss_obj_center_1e-3', loss_oc_out * 1000.0, (epoch-1) * len(sdf_loader) + i)
                    writer.add_scalar('loss_obj_corner_1e-3', loss_ocr_out * 1000.0, (epoch-1) * len(sdf_loader) + i)

                if classifier_branch and epoch >= start_additional_loss:
                    writer.add_scalar('loss_cls_1e-3', loss_ce_out * 1000.0, (epoch-start_additional_loss) * len(sdf_loader) + i)

                if do_penetration_loss and epoch >= start_additional_loss:
                    writer.add_scalar('loss_pen_1e-3', loss_pen_out * 1000.0, (epoch-start_additional_loss) * len(sdf_loader) + i)

                if do_contact_loss and epoch >= start_additional_loss:
                    writer.add_scalar('loss_contact_1e-3', loss_contact_out * 1000.0, (epoch-start_additional_loss) * len(sdf_loader) + i)

            optimizer_all.step()
            torch.cuda.synchronize()
        
        end = time.time()
        seconds_elapsed = end - start
        if local_rank == 0:
            logging.info("time used: {}".format(seconds_elapsed))
            for idx, schedule in enumerate(lr_schedules):
                writer.add_scalar('learning_rate_' + str(idx), schedule.get_learning_rate(epoch), epoch * len(sdf_loader))

            if epoch % log_frequency == 0:
                save_latest(epoch)
                logging.info("save at {}".format(epoch))
            
            logging.info("Distributing BatchNorm running means and vars")
        distribute_bn(encoderDecoder, world_size, reduce=True)
    
    if local_rank == 0:
        writer.close()
    
    # begin test_session
    split_filename = f'input/{dataset_name}.json'
    output_path = os.path.join(experiment_directory, f'Eval_{dataset_name}')
    task = dataset_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_eval_mode = True if ('obman' in task) or ('dexycb' in task) else False
    use_optim_mano = True

    all_filenames = json.load(open(split_filename))['filenames']
    division = len(all_filenames) // world_size

    start_points = []
    end_points = []
    for i in range(world_size):
        start_point = i * division
        if i != world_size - 1:
            end_point = start_point + division
        else:
            end_point = len(all_filenames) + 1

        start_points.append(start_point)
        end_points.append(end_point)
    
    reconstruct(encoderDecoder, specs, split_filename, output_path, start_point=start_points[local_rank], end_point=end_points[local_rank], task=task, device=device, cube_dim=128, label_out=use_optim_mano, eval_mode=use_eval_mode)


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Reconstruct 3D objects using deepsdf")
    arg_parser.add_argument(
        "-e",
        dest="cfg",
        required=True,
        help="experiment config"
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from"
    )
    arg_parser.add_argument(
        "--local_rank",
        default=0,
        type=int
    )
    arg_parser.add_argument(
        "--opt_level",
        default='O0',
        type=str
    )
    arg_parser.add_argument(
        "--slurm",
        dest="slurm",
        action='store_true'
    )

    utils.add_common_args(arg_parser)
    utils.add_train_args(arg_parser)
    args = arg_parser.parse_args()
    utils.configure_logging(args)
    
    with open(args.cfg, 'r') as f:
        exp_cfg = json.load(f)
    
    exp_cfg = utils.update_exp_cfg(exp_cfg, args)
    
    main_function(exp_cfg, args.continue_from, args.local_rank, args.opt_level, args.slurm)
