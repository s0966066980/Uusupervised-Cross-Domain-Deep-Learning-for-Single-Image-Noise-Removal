import torch
import math
import torch.nn as nn
import torch.nn.init as init

from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torchvision.models as models
from utils import get_fft_feature


# We adopt the network in https://github.com/nmhkahn/CARN-pytorch as the Generator_N2C 
class Generator_N2C(nn.Module):
    def __init__(self, input_channel, output_channel, middle_channel):
        super(Generator_N2C, self).__init__()

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(input_channel, middle_channel, 3, 1, 1)

        self.b1 = Cascading_Block(middle_channel)
        self.b2 = Cascading_Block(middle_channel)
        self.b3 = Cascading_Block(middle_channel)
        self.c1 = BasicBlock(middle_channel*2, middle_channel, 1, 1, 0)
        self.c2 = BasicBlock(middle_channel*3, middle_channel, 1, 1, 0)
        self.c3 = BasicBlock(middle_channel*4, middle_channel, 1, 1, 0)

        
        self.exit = nn.Conv2d(middle_channel, output_channel, 3, 1, 1)
        
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = o3

        out = self.exit(out)
        out = self.add_mean(out)

        return out


# We adopt smilar UNet-like network in https://github.com/terryoo/AINDNet as the Generator_C2N 
class Generator_C2N(nn.Module):
    def __init__(self, input_channel, output_channel, middle_channel=64, n_blocks=6):
        super(Generator_C2N, self).__init__()

        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, middle_channel, kernel_size=3, stride=1, padding=1),
            InResBlock(middle_channel),
            InResBlock(middle_channel),
        )
        
        self.conv2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channel, middle_channel*2, kernel_size=3, stride=1, padding=1),
            InResBlock(middle_channel*2),
            InResBlock(middle_channel*2),
        )

        conv3 = [nn.AvgPool2d(2), nn.Conv2d(middle_channel*2, middle_channel*4, kernel_size=3, stride=1, padding=1)]

        for i in range(n_blocks):
            conv3 += [InResBlock(middle_channel*4)]

        self.conv3 = nn.Sequential(*conv3)

        self.up1 = nn.ConvTranspose2d(middle_channel*4, middle_channel*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(middle_channel*2, middle_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Sequential(
            InResBlock(middle_channel*2),
            InResBlock(middle_channel*2)
        )

        self.conv5 = nn.Sequential(
            InResBlock(middle_channel),
            InResBlock(middle_channel),
            nn.Conv2d(middle_channel, output_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(self.up1(conv3) + conv2)
        conv5 = self.conv5(self.up2(conv4) + conv1 )
        out = conv5 + x 
        return out


# We adopt the PatchGAN in https://github.com/phillipi/pix2pix as clean discriminator and texture discriminator
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
            ),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                    )
                ),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                )
            ),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [
            spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
            ),
            nn.LeakyReLU(0.2, True),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


# We adopt the Spectral discrimiantor in https://github.com/cyq373/SSD-GAN1
class Spectral_Discriminator(nn.Module):
    def __init__(self, height):
        super(Spectral_Discriminator, self).__init__()
        self.thresh = int(height / (2*math.sqrt(2)))
        self.linear = nn.Linear(self.thresh, 1)
    
    def forward(self, input: torch.Tensor):
        az_fft_feature = get_fft_feature(input)
        az_fft_feature[torch.isnan(az_fft_feature)] = 0
        
        return self.linear(az_fft_feature[:,-self.thresh:])


class Cascading_Block(nn.Module):
    def __init__(self, middle_channel):
        super(Cascading_Block, self).__init__()

        self.b1 = ResidualBlock(middle_channel, middle_channel)
        self.b2 = ResidualBlock(middle_channel, middle_channel)
        self.b3 = ResidualBlock(middle_channel, middle_channel)
        self.c1 = BasicBlock(middle_channel*2, middle_channel, 1, 1, 0)
        self.c2 = BasicBlock(middle_channel*3, middle_channel, 1, 1, 0)
        self.c3 = BasicBlock(middle_channel*4, middle_channel, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class IN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(IN, self).__init__()

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.betta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(
            x, dim=[2, 3], keepdim=True
        )

        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)

        out = self.gamma.expand(x.shape[0], -1, -1, -1) * out_in + self.betta.expand(
            x.shape[0], -1, -1, -1
        )

        return out


class InResBlock(nn.Module):
    def __init__(self, dim):
        super(InResBlock, self).__init__()

        self.norm1 = IN(dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0)

        self.norm2 = IN(dim)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.pad1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.pad2(out)
        out = self.conv2(out)

        out = out + x

        return out


class vgg_19(nn.Module):
    def __init__(self):
        super(vgg_19, self).__init__()
        vgg_model = models.vgg19(pretrained=True)
        self.feature_ext = nn.Sequential(*list(vgg_model.features.children())[:20])
    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat((x, x, x), 1)
        out = self.feature_ext(x)
        return out


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
'''
import numpy as np
from scunet import ConvTransBlock
class Generator_N2C(nn.Module):

    def __init__(self, in_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(Generator_N2C, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[0])] + \
                      [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[1])] + \
                      [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[2])] + \
                      [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8*dim, 4*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[4])]
                      
        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4*dim, 2*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[5])]
                      
        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + \
                    [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)  
        #self.apply(self._init_weights)

    def forward(self, x0):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h/64)*64-h)
        paddingRight = int(np.ceil(w/64)*64-w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]
        
        return x

'''

from thop import profile
if __name__ == '__main__':


    c2nd = Generator_C2N(3,3,64)
    input = torch.randn(3,3,64,64)
    flops, params = profile(c2nd, inputs=(input,))
    print("%.2fG" % (flops/1e9), "%.2fM" % (params/1e6))
