from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import torch.nn as nn

import lib.utils.etw_pytorch_utils as pt_utils
from lib.backbone.network_resUNet_edge_pointp_vlad_se import Feature_Backbone


class KP_SaMfENet(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        pcld_input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        pcld_use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
        num_kps: int = 8
            Number of keypoints to predict
        num_points: int 8192
            Number of sampled points from point clouds.
    """

    def __init__(
            self, num_classes, num_kps=8, num_points=8192):
        super(KP_SaMfENet, self).__init__()

        self.num_kps = num_kps
        self.backbone = Feature_Backbone(num_points=num_points)

        self.SEG_layer = (
            pt_utils.Seq(1792)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(num_classes, activation=None)
        )

        self.KpOF_layer = (
            pt_utils.Seq(1792)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(256, bn=True, activation=nn.ReLU())
            .conv1d(num_kps * 3, activation=None)
        )

        self.CtrOf_layer = (
            pt_utils.Seq(1792)
            .conv1d(1024, bn=True, activation=nn.ReLU())
            .conv1d(512, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

    def forward(self, pointcloud, rgb, choose):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
            rgb: Variable(torch.cuda.FloatTensor)
                (B, C, H, W) tensor
            choose: Variable(torch.cuda.LongTensor)
                (B, 1, N) tensor
                indexs of choosen points(pixels).
        """

        bs, _, _, _ = rgb.size()
        _, N, _ = pointcloud.size()
        rgbd_feature, pred_edge = self.backbone(rgb, pointcloud, choose)

        pred_rgbd_seg = self.SEG_layer(rgbd_feature).transpose(1, 2).contiguous()
        pred_kp_of = self.KpOF_layer(rgbd_feature).view(
            bs, self.num_kps, 3, N
        )
        # return [bs, n_kps, n_pts, c]
        pred_kp_of = pred_kp_of.permute(0, 1, 3, 2).contiguous()
        pred_ctr_of = self.CtrOf_layer(rgbd_feature).view(
            bs, 1, 3, N
        )
        pred_ctr_of = pred_ctr_of.permute(0, 1, 3, 2).contiguous()

        return pred_kp_of, pred_rgbd_seg, pred_ctr_of, pred_edge
