import os
import torch
import networks.model as arch
import networks.manobranch as mano


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_model(model_directory, specs, device):
    model_type = specs["ModelType"]
    latent_size = specs["LatentSize"]
    backbone = specs['Backbone']
    encode_style = get_spec_with_default(specs, "EncodeStyle", "nerf")
    point_feat_size = get_spec_with_default(specs, "PointFeatSize", 3)
    classifier_branch = get_spec_with_default(specs, "ClassifierBranch", False)
    mano_branch = get_spec_with_default(specs, "ManoBranch", False)
    depth_branch = get_spec_with_default(specs, "DepthBranch", False)
    obj_pose_branch = get_spec_with_default(specs, "ObjectPoseBranch", False)

    if model_type == "1encoder1decoder":
        # use_combined_decoder = True
        encoder = arch.get_encoder(backbone, output_size=latent_size, specs=specs)
        sdf_decoder = arch.CombinedDecoder(latent_size, point_feat_size, encode_style, **specs["NetworkSpecs"], use_classifier=classifier_branch)
    elif model_type == "1encoder2decoder":
        encoder = arch.get_encoder(backbone, output_size=latent_size, specs=specs)
        sdf_decoder = arch.SeparateDecoder(latent_size, point_feat_size, encode_style, **specs["NetworkSpecs"], use_classifier=classifier_branch)

    if mano_branch:
        mano_decoder = mano.ManoBranch(ncomps=specs['PoseFeatSize'], absolute_depth=depth_branch, object_pose=obj_pose_branch, use_obj_rot=specs['ObjCornerWeight']>0)
    else:
        mano_decoder = None

    encoderDecoder = arch.ModelOneEncoderOneDecoder(encoder, sdf_decoder, mano_decoder, specs)

    encoderDecoder = torch.nn.DataParallel(encoderDecoder)

    # Load weights
    saved_model_state = torch.load(os.path.join(model_directory, "latest.pth"))
    saved_model_epoch = saved_model_state["epoch"]
    print("using model from epoch {}".format(saved_model_epoch))

    encoderDecoder.load_state_dict(saved_model_state["model_state_dict"])

    encoderDecoder = encoderDecoder.to(device)

    return encoderDecoder
