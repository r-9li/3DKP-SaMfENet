import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from lib.backbone.NetVlad import NetVLADLoupe
from lib.backbone.SK_SE_Net import SEBlock
from lib.backbone.UNet import UNet
from lib.backbone.pointnet_plus_sknet_gap import SK_PointNet


class Feature_Backbone(nn.Module):
    def __init__(self, num_points):
        super(Feature_Backbone, self).__init__()
        self.num_points = num_points
        self.Encoder_Decoder = UNet(bilinear=True)
        self.pointnet_plus_sk = SK_PointNet(num_branches=6)  # TODO
        self.vlad = NetVLADLoupe(feature_size=640, max_samples=num_points, cluster_size=24, output_dim=768,
                                 gating=True, add_batch_norm=True)

        self.conv1 = torch.nn.Conv1d(640, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 384, 1)

        self.senet = SEBlock(channels=1792)

    def forward(self, img, x, choose):
        out_edge, out_img = self.Encoder_Decoder(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        # x = x.transpose(2, 1).contiguous()
        feat_x = self.pointnet_plus_sk(x)

        x_feature = torch.cat([emb, feat_x], 1)  # 256+384=640

        x_feature_1 = F.relu(self.conv1(x_feature))
        x_feature_1 = F.relu(self.conv2(x_feature_1))  # 384

        ap_x = self.vlad(x_feature)
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)

        ap_x = torch.cat([x_feature, ap_x, x_feature_1], 1)  # 640+384+768=1792

        ap_x = self.senet(ap_x)

        return ap_x, out_edge
