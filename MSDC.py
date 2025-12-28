import torch
import torch.nn as nn
import torch.nn.functional as F


class Branch(nn.Module):
    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        
        self.branch = nn.Sequential(
                       nn.Conv2d(in_channels=self.dw_channel, 
                                 out_channels=self.dw_channel, 
                                 kernel_size=3, 
                                 padding=dilation, 
                                 stride=1, 
                                 groups=self.dw_channel,
                                 bias=True, 
                                 dilation=dilation)
        )
    
    def forward(self, input):
        return self.branch(input)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
               grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class MSDC(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        self.dw_channel = DW_Expand * c 

        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(in_channels=c, 
                               out_channels=self.dw_channel, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True, 
                               dilation=1)
        self.extra_conv = nn.Conv2d(self.dw_channel, 
                                   self.dw_channel, 
                                   kernel_size=3, 
                                   padding=1, 
                                   stride=1, 
                                   groups=c, 
                                   bias=True, 
                                   dilation=1) if extra_depth_wise else nn.Identity()
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand=1, dilation=dilation))
        
        assert len(dilations) == len(self.branches)
        
        self.conv_3x3_1 = nn.Conv2d(in_channels=self.dw_channel, 
                                   out_channels=self.dw_channel // 2, 
                                   kernel_size=3, 
                                   padding=1, 
                                   stride=1, 
                                   groups=1, 
                                   bias=True)
        self.conv_1x1 = nn.Conv2d(in_channels=self.dw_channel // 2, 
                                 out_channels=self.dw_channel // 2, 
                                 kernel_size=1, 
                                 padding=0, 
                                 stride=1, 
                                 groups=1, 
                                 bias=True)
        self.conv_3x3_2 = nn.Conv2d(in_channels=self.dw_channel // 2, 
                                   out_channels=c, 
                                   kernel_size=3, 
                                   padding=1, 
                                   stride=1, 
                                   groups=1, 
                                   bias=True)
        
        self.relu = nn.ReLU(inplace=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        y = inp
        
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.extra_conv(x)
        
        z = 0
        for branch in self.branches:
            z += branch(x)
        
        x = self.relu(self.conv_3x3_1(z))
        x = self.relu(self.conv_1x1(x))
        x = self.conv_3x3_2(x)
        
        out = y + self.beta * x
        
        return out


if __name__ == '__main__':
    channels = 64
    dilations = [1, 3, 5]
    multi_branch = MSDC(c=channels, DW_Expand=2, dilations=dilations, extra_depth_wise=True)
    
    input_tensor = torch.randn(4, channels, 64, 64)
    
    output_tensor = multi_branch(input_tensor)
    
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")
    
    assert input_tensor.shape == output_tensor.shape, "输入输出形状不匹配！"