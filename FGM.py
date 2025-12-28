import torch
import torch.nn as nn
from einops import rearrange


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.dim = in_channels

        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                      kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True),
            nn.GELU(),
        )

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.conv_layer(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class OurTokenMixer_For_Gloal(nn.Module):
    def __init__(self, dim):
        super(OurTokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1)
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.FFC(x)
        x = self.conv_fina(x)
        return x


class OurMixer(nn.Module):
    def __init__(self, dim, token_mixer_for_gloal=OurTokenMixer_For_Gloal):
        super(OurMixer, self).__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = self.mixer_gloal(x)
        x = self.gelu(x)
        x = self.ca_conv(x)
        return x


class FGM(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super(FGM, self).__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        self.mixer = OurMixer(dim=self.dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm(x)
        x = self.mixer(x)
        x = x * self.beta + copy
        return x


class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class FreMLPWithIntegration(nn.Module):
    def __init__(self, nc, expand=2, use_norm=True):
        super(FreMLPWithIntegration, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            try:
                from .arch_util import LayerNorm2d
                self.norm = LayerNorm2d(nc)
            except ImportError:
                class LayerNorm2d(nn.Module):
                    def __init__(self, channels):
                        super().__init__()
                        self.weight = nn.Parameter(torch.ones(channels))
                        self.bias = nn.Parameter(torch.zeros(channels))
                    
                    def forward(self, x):
                        mean = x.mean(dim=(1, 2, 3), keepdim=True)
                        var = x.var(dim=(1, 2, 3), keepdim=True)
                        x = (x - mean) / torch.sqrt(var + 1e-5)
                        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
                self.norm = LayerNorm2d(nc)
                self.freq = FreMLP(nc=nc, expand=expand)
        
        self.gamma = nn.Parameter(torch.zeros((1, nc, 1, 1)), requires_grad=True)

    def forward(self, y):
        if self.use_norm:
            x_step2 = self.norm(y)
        else:
            x_step2 = y
        x_freq = self.freq(x_step2)
        x = y * x_freq
        x = y + x * self.gamma
        return x


