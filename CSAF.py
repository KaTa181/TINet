import torch
import torch.nn as nn
import torch.nn.functional as F
from FGM import FreMLPWithIntegration

class FeatureEnhancementBlock(FreMLPWithIntegration):
    def __init__(self, in_channels, expand=2, use_norm=True):
        super(FeatureEnhancementBlock, self).__init__(
            nc=in_channels,
            expand=expand,
            use_norm=use_norm
        )

class CrossScaleAttentionFusion(nn.Module):
    def __init__(self, in_dims=[24, 48, 96], output_scale='first', use_enhancement=True):
        super(CrossScaleAttentionFusion, self).__init__()
        
        self.dim1, self.dim2, self.dim3 = in_dims
        self.output_scale = output_scale
        self.use_enhancement = use_enhancement
        
        total_dim = self.dim1 + self.dim2 + self.dim3
        
        if self.use_enhancement:
            self.enhance1 = FeatureEnhancementBlock(self.dim1)
            self.enhance2 = FeatureEnhancementBlock(self.dim2)
            self.enhance3 = FeatureEnhancementBlock(self.dim3)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.ca = nn.Sequential(
            nn.Conv2d(total_dim, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1, padding=0, bias=True),
        )
        
        self.fuse_conv_first = nn.Sequential(
            nn.Conv2d(total_dim, self.dim1, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.dim1),
            nn.ReLU(True)
        )
        
        self.fuse_conv_second = nn.Sequential(
            nn.Conv2d(total_dim, self.dim2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.dim2),
            nn.ReLU(True)
        )
    
    def forward(self, x1, x2, x3):
        if self.use_enhancement:
            x1 = self.enhance1(x1)
            x2 = self.enhance2(x2)
            x3 = self.enhance3(x3)
        
        x_avg1 = self.avg_pool(x1)
        x_avg2 = self.avg_pool(x2)
        x_avg3 = self.avg_pool(x3)
        
        fea_avg = torch.cat([x_avg1, x_avg2, x_avg3], dim=1)
        
        attention_score = self.ca(fea_avg)
        attention_score = F.softmax(attention_score, dim=1)
        
        w1, w2, w3 = torch.chunk(attention_score, 3, dim=1)
        
        x1_att = x1 * w1
        x2_att = x2 * w2
        x3_att = x3 * w3
        
        if self.output_scale == 'first':
            x2_att_aligned = F.interpolate(x2_att, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
            x3_att_aligned = F.interpolate(x3_att, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
            
            fuse_feature = torch.cat([x1_att, x2_att_aligned, x3_att_aligned], dim=1)
            
            return self.fuse_conv_first(fuse_feature)
        elif self.output_scale == 'second':
            x1_att_aligned = F.interpolate(x1_att, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=False)
            x3_att_aligned = F.interpolate(x3_att, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=False)
            
            fuse_feature = torch.cat([x1_att_aligned, x2_att, x3_att_aligned], dim=1)
            
            return self.fuse_conv_second(fuse_feature)
        else:
            x2_att_aligned = F.interpolate(x2_att, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
            x3_att_aligned = F.interpolate(x3_att, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
            
            fuse_feature = torch.cat([x1_att, x2_att_aligned, x3_att_aligned], dim=1)
            
            return self.fuse_conv_first(fuse_feature)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, use_enhancement=True):
        super(AFF, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_enhancement = use_enhancement
        
        if self.use_enhancement:
            self.use_dynamic_enhancement = True
        else:
            self.use_dynamic_enhancement = False
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.ca = nn.Sequential(
            nn.Conv2d(in_channel, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1, padding=0, bias=True),
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        )
    
    def forward(self, x1, x2, x4):
        if hasattr(self, 'use_dynamic_enhancement') and self.use_dynamic_enhancement:
            with torch.no_grad():
                enhance1 = FeatureEnhancementBlock(x1.size(1)).to(x1.device)
                enhance2 = FeatureEnhancementBlock(x2.size(1)).to(x2.device)
                enhance3 = FeatureEnhancementBlock(x4.size(1)).to(x4.device)
            
            x1 = enhance1(x1)
            x2 = enhance2(x2)
            x4 = enhance3(x4)
        
        x_avg1 = self.avg_pool(x1)
        x_avg2 = self.avg_pool(x2)
        x_avg4 = self.avg_pool(x4)
        
        fea_avg = torch.cat([x_avg1, x_avg2, x_avg4], dim=1)
        
        attention_score = self.ca(fea_avg)
        attention_score = F.softmax(attention_score, dim=1)
        
        w1, w2, w4 = torch.chunk(attention_score, 3, dim=1)
        
        x1_att = x1 * w1
        x2_att = x2 * w2
        x4_att = x4 * w4
        
        fuse_feature = torch.cat([x1_att, x2_att, x4_att], dim=1)
        
        return self.conv(fuse_feature)

