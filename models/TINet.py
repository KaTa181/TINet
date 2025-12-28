import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers import *
from CSAF import AFF
from MSDC import MSDC
from FGM import FGM


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = []
        for _ in range(num_res):
            layers.append(FGM(dim=out_channel))
            layers.append(MSDC(c=out_channel, 
                                           DW_Expand=2, 
                                           dilations=[1, 3, 5], 
                                           extra_depth_wise=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = []
        for _ in range(num_res):
            layers.append(FGM(dim=channel))
            layers.append(MSDC(c=channel, 
                                           DW_Expand=2, 
                                           dilations=[1, 3, 5], 
                                           extra_depth_wise=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class TINet(nn.Module):
    def __init__(self, num_res=8):
        super(TINet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])
        
        self.Decoder_AFFs = nn.ModuleList([
            AFF(base_channel*7, base_channel*4),
            AFF(base_channel*7, base_channel*2),
            AFF(base_channel*7, base_channel)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        res2 = res2
        res1 = res1

        z = self.Decoder[0](z)
        
        res1_scaled = F.interpolate(res1, scale_factor=0.25)
        res2_scaled = F.interpolate(res2, scale_factor=0.5)
        
        z_aff = self.Decoder_AFFs[0](z, res2_scaled, res1_scaled)
        
        z_ = self.ConvsOut[0](z_aff)
        outputs.append(z_ + x_4)
        
        z = self.feat_extract[3](z_aff)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        
        res1_scaled_2 = F.interpolate(res1, scale_factor=0.5)
        encoder3_scaled = F.interpolate(self.feat_extract[2](res2), scale_factor=2)
        
        z_aff = self.Decoder_AFFs[1](z, encoder3_scaled, res1_scaled_2)
        
        z_ = self.ConvsOut[1](z_aff)
        outputs.append(z_ + x_2)
        
        z = self.feat_extract[4](z_aff)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        
        res2_scaled_2 = F.interpolate(res2, scale_factor=2)
        encoder3_scaled_2 = F.interpolate(self.feat_extract[2](res2), scale_factor=4)
        
        z_aff = self.Decoder_AFFs[2](z, res2_scaled_2, encoder3_scaled_2)
        
        z = self.feat_extract[5](z_aff)
        outputs.append(z + x)

        return outputs


def build_net(model_name):
    if model_name == 'TINet':
        return TINet()
    elif model_name == 'TINetPlus':
        return TINetPlus()
    else:
        raise NotImplementedError('Model {} is not implemented'.format(model_name))


class TINetPlus(nn.Module):
    def __init__(self, num_res = 20):
        super(TINetPlus, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])
        
        self.Decoder_AFFs = nn.ModuleList([
            AFF(base_channel*7, base_channel*4),
            AFF(base_channel*7, base_channel*2),
            AFF(base_channel*7, base_channel)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        res2 = res2
        res1 = res1

        z = self.Decoder[0](z)
        
        res1_scaled = F.interpolate(res1, scale_factor=0.25)
        res2_scaled = F.interpolate(res2, scale_factor=0.5)
        
        z_aff = self.Decoder_AFFs[0](z, res2_scaled, res1_scaled)
        
        z_ = self.ConvsOut[0](z_aff)
        outputs.append(z_ + x_4)
        
        z = self.feat_extract[3](z_aff)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        
        res1_scaled_2 = F.interpolate(res1, scale_factor=0.5)
        encoder3_scaled = F.interpolate(self.feat_extract[2](res2), scale_factor=2)
        
        z_aff = self.Decoder_AFFs[1](z, encoder3_scaled, res1_scaled_2)
        
        z_ = self.ConvsOut[1](z_aff)
        outputs.append(z_ + x_2)
        
        z = self.feat_extract[4](z_aff)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        
        res2_scaled_2 = F.interpolate(res2, scale_factor=2)
        encoder3_scaled_2 = F.interpolate(self.feat_extract[2](res2), scale_factor=4)
        
        z_aff = self.Decoder_AFFs[2](z, res2_scaled_2, encoder3_scaled_2)
        
        z = self.feat_extract[5](z_aff)
        outputs.append(z + x)

        return outputs