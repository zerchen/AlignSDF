from logging import debug
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torch.autograd import grad
from torchvision import models
import numpy as np
import cv2
from networks import resnet
from utils.utils import get_obj_trans, kinematic_embedding, get_nerf_embedder, pixel_alignment, soft_argmax, get_obj_trans
try:
    import soft_renderer as sr
    import soft_renderer.functional as srf
except:
    print('do not support renderer in this machine')


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def freeze_batchnorm_stats(model, freeze):
    if freeze:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = 0
        for name, child in model.named_children():
            freeze_batchnorm_stats(child, freeze)


class HeadNet(nn.Module):
    def __init__(self, num_layers=3):
        super(HeadNet, self).__init__()
        self.inplanes = 512
        self.outplanes = 256
        self.deconv_layers = self._make_deconv_layer(num_layers)

    def _make_deconv_layer(self, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=self.outplanes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        return x


def get_encoder(model_name, output_size, specs):
    use_pretrained = True
    mano_features = specs['ManoBranch']
    use_headnet = specs['ObjectPoseBranch'] or specs['Render'] or specs['PixelAlign']
    use_pixel_align = specs['PixelAlign']

    if model_name == "resnet18":
        model_ft = resnet.resnet18(pretrained=use_pretrained, mano_features=mano_features, use_headnet=use_headnet, use_pixel_align=use_pixel_align)
    elif model_name == "resnet34":
        model_ft = resnet.resnet34(pretrained=use_pretrained, mano_features=mano_features, use_headnet=use_headnet, use_pixel_align=use_pixel_align)
    elif model_name == "resnet50":
        model_ft = resnet.resnet50(pretrained=use_pretrained, mano_features=mano_features, use_headnet=use_headnet, use_pixel_align=use_pixel_align)

    if specs['PixelAlign']:
        model_ft.fc = nn.AvgPool2d(7, stride=1)
    else:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, output_size)

    if use_headnet:
        model_ft.aux_layer = HeadNet()

    return model_ft


class CombinedDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        point_feat_size,
        encode_style,
        dims,
        num_class,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        use_classifier=False,
    ):
        super(CombinedDecoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + point_feat_size] + dims + [2]  # <<<< 2 outputs instead of 1.
        self.point_feat_size = point_feat_size
        self.encode_style=encode_style
        self.num_layers = len(dims)
        self.num_class = num_class
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.use_classifier = use_classifier

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= self.point_feat_size
            # print("out dim", out_dim)

            if weight_norm and layer in self.norm_layers:
                setattr(self, "lin" + str(layer), nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),)
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if ((not weight_norm) and self.norm_layers is not None and layer in self.norm_layers):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

            # classifier
            if self.use_classifier and layer == self.num_layers - 2:
                # print("dim last_layer", dims[layer])
                self.classifier_head = nn.Linear(dims[layer], self.num_class)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -self.point_feat_size:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-self.point_feat_size]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            # classify
            if self.use_classifier and layer == self.num_layers - 2:
                predicted_class = self.classifier_head(x)

            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (self.norm_layers is not None and layer in self.norm_layers and not self.weight_norm):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        # hand, object, class label
        if self.use_classifier:
            return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1), predicted_class
        else:
            return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1), torch.Tensor([0]).cuda()


class SeparateDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        point_feat_size,
        encode_style,
        dims,
        num_class,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        use_classifier=False,
    ):
        super(SeparateDecoder, self).__init__()

        def make_sequence():
            return []

        self.point_feat_size = point_feat_size
        self.encode_style = encode_style
        if self.encode_style == 'nerf':
            dims_hand = [latent_size + point_feat_size] + dims + [1]
            dims_obj = [latent_size + point_feat_size] + dims + [1]
        elif self.encode_style == 'hand':
            dims_hand = [latent_size + point_feat_size] + dims + [1]
            dims_obj = [latent_size + 3] + dims + [1]
        elif self.encode_style == 'obj':
            dims_hand = [latent_size + 3] + dims + [1]
            dims_obj = [latent_size + point_feat_size] + dims + [1]
        elif self.encode_style == 'both':
            dims_hand = [latent_size + point_feat_size - 3] + dims + [1]
            dims_obj = [latent_size + 6] + dims + [1]

        self.latent_size = latent_size
        self.num_hand_layers = len(dims_hand)
        self.num_obj_layers = len(dims_hand)
        self.num_class = num_class
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.use_classifier = use_classifier

        for layer in range(0, self.num_hand_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims_hand[layer + 1] - dims_hand[0]
            else:
                out_dim = dims_hand[layer + 1]

            if weight_norm and layer in self.norm_layers:
                setattr(self, "linh" + str(layer), nn.utils.weight_norm(nn.Linear(dims_hand[layer], out_dim)),)
            else:
                setattr(self, "linh" + str(layer), nn.Linear(dims_hand[layer], out_dim))

            if ((not weight_norm) and self.norm_layers is not None and layer in self.norm_layers):
                setattr(self, "bnh" + str(layer), nn.LayerNorm(out_dim))

            # classifier
            if self.use_classifier and layer == self.num_layers - 2:
                self.classifier_head = nn.Linear(dims_hand[layer], self.num_class)

        for layer in range(0, self.num_obj_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims_obj[layer + 1] - dims_obj[0]
            else:
                out_dim = dims_obj[layer + 1]

            if weight_norm and layer in self.norm_layers:
                setattr(self, "lino" + str(layer), nn.utils.weight_norm(nn.Linear(dims_obj[layer], out_dim)),)
            else:
                setattr(self, "lino" + str(layer), nn.Linear(dims_obj[layer], out_dim))

            if ((not weight_norm) and self.norm_layers is not None and layer in self.norm_layers):
                setattr(self, "bno" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -self.point_feat_size:-self.point_feat_size + 3]

        if self.encode_style == 'nerf':
            xh = input
            xo = input
        elif self.encode_style == 'hand':
            xh = input
            xo = input[:, :self.latent_size + 3]
        elif self.encode_style == 'obj':
            xh = input[:, :self.latent_size + 3]
            xo = input
        elif self.encode_style == 'both':
            xh = input[:, :-3]
            xo = torch.cat([input[:, :self.latent_size + 3], input[:, -3:]], 1)
        
        input_hand = xh
        input_obj = xo

        for layer in range(0, self.num_hand_layers - 1):
            # classify
            if self.use_classifier and layer == self.num_hand_layers - 2:
                predicted_class = self.classifier_head(xh)

            lin = getattr(self, "linh" + str(layer))
            if layer in self.latent_in:
                xh = torch.cat([xh, input_hand], 1)
            xh = lin(xh)
            # last layer Tanh
            if layer == self.num_hand_layers - 2 and self.use_tanh:
                xh = self.tanh(xh)
            if layer < self.num_hand_layers - 2:
                if (self.norm_layers is not None and layer in self.norm_layers and not self.weight_norm):
                    bn = getattr(self, "bnh" + str(layer))
                    xh = bn(xh)
                xh = self.relu(xh)
                if self.dropout is not None and layer in self.dropout:
                    xh = F.dropout(xh, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            xh = self.th(xh)

        for layer in range(0, self.num_obj_layers - 1):
            lin = getattr(self, "lino" + str(layer))
            if layer in self.latent_in:
                xo = torch.cat([xo, input_obj], 1)
            xo = lin(xo)
            # last layer Tanh
            if layer == self.num_obj_layers - 2 and self.use_tanh:
                xo = self.tanh(xo)
            if layer < self.num_obj_layers - 2:
                if (self.norm_layers is not None and layer in self.norm_layers and not self.weight_norm):
                    bn = getattr(self, "bno" + str(layer))
                    xo = bn(xo)
                xo = self.relu(xo)
                if self.dropout is not None and layer in self.dropout:
                    xo = F.dropout(xo, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            xo = self.th(xo)

        # hand, object, class label
        if self.use_classifier:
            return xh[:, 0].unsqueeze(1), xo[:, 0].unsqueeze(1), predicted_class
        else:
            return xh[:, 0].unsqueeze(1), xo[:, 0].unsqueeze(1), torch.Tensor([0]).cuda()


class ModelOneEncoderOneDecoder(nn.Module):
    def __init__(self, encoder, decoder, mano_decoder, specs):
        super(ModelOneEncoderOneDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mano_decoder = mano_decoder
        self.scale_aug = specs["ScaleAug"]
        self.dataset = specs["Dataset"]
        self.sdf_scale_factor = specs["SdfScaleFactor"]
        self.num_samp_per_scene = specs["SamplesPerScene"]
        self.img_size = specs['ImageSize']
        self.pose_feat_size = specs["PoseFeatSize"]
        self.point_feat_size = specs["PointFeatSize"]
        self.sdf_latent_size = specs["LatentSize"]
        self.use_obj_pose = specs['ObjectPoseBranch'] if self.mano_decoder is not None else False
        self.use_depth = specs['DepthBranch'] if self.mano_decoder is not None else False
        self.use_render = specs['Render'] if self.mano_decoder is not None else False
        self.encode_style = specs['EncodeStyle']
        self.pixel_align = specs["PixelAlign"] if self.mano_decoder is not None else False
        self.use_obj_rot = specs["ObjCornerWeight"] > 0
        self.num_epochs = specs["NumEpochs"]
        self.freeze = specs['Freeze']

        if self.use_obj_pose:
            self.volume_layer = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x1, xyz, cond_input):
        sdf_feat, mano_feat, aux_feat = self.encoder(x1)

        mano_results = None
        if self.mano_decoder is not None:
            mano_results = self.mano_decoder(mano_feat, cond_input)

        obj_results = None
        if self.use_obj_pose and mano_results is not None:
            heatmaps_obj = self.volume_layer(aux_feat)
            obj_center_2d = soft_argmax(heatmaps_obj)
            obj_trans, obj_center = get_obj_trans(obj_center_2d, mano_results, cond_input['cam_intr'], self.use_obj_rot)
            homo_obj_corners = torch.ones((obj_trans.shape[0], 8, 4)).to(obj_trans.device)
            homo_obj_corners[:, :, :3] = cond_input['rest_obj_corners']
            homo_obj_corners = torch.matmul(obj_trans, homo_obj_corners.transpose(2, 1)).transpose(2, 1)
            obj_corners = homo_obj_corners[:, :, :3] / homo_obj_corners[:, :, [3]]
            if not self.training:
                obj_corners = obj_corners + mano_results['center3d']
            obj_results = dict(obj_center=obj_center, obj_corners=obj_corners, obj_trans=obj_trans)

        if self.pixel_align:
            latent = pixel_alignment(aux_feat, xyz, cond_input['cam_intr'], mano_results, self.img_size[0], self.sdf_scale_factor) 
        else:
            latent = sdf_feat.repeat_interleave(self.num_samp_per_scene, dim=0)

        if self.point_feat_size > 3:
            if mano_results is not None and self.encode_style != "nerf":
                xyz = kinematic_embedding(xyz, mano_results, self.num_samp_per_scene, self.point_feat_size, self.sdf_scale_factor, obj_results=obj_results, encode_style=self.encode_style)
            else:
                nerf_embedding, _ = get_nerf_embedder((self.point_feat_size - 3) // 6)
                xyz = nerf_embedding(xyz)

        decoder_inputs = torch.cat([latent, xyz], 1)
        x_hand, x_obj, x_class = self.decoder(decoder_inputs)

        return x_hand, x_obj, x_class, mano_results, obj_results
