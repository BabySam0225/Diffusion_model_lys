import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision import transforms
from .module_util import SinusoidalPosEmb, LayerNorm, exists
# from mamba_ssm import Mamba
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))


def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff


def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


# Edge-Guided Attention Module
class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)

    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]
        pred = torch.sigmoid(pred)

        # reverse attention
        background_att = 1 - pred
        background_x = x * background_att

        # boudary attention
        edge_pred = make_laplace(pred, 1)
        pred_feature = x * edge_pred

        # high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.cbam(out)
        return out

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        # self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               # stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        return x

# 简单的门控模块，将输入张量沿着通道维度切分成两部分，然后对两部分进行逐元素相乘
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))

# --------------------------------------------------------------------------------------------------

class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv2_5 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=5, padding=2, stride=1, groups=dw_channel, bias=True)
        self.conv2_7 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=7, padding=3, stride=1, groups=dw_channel, bias=True)
        self.blending = nn.Conv2d(in_channels=dw_channel * 3 // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention+

        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        self.sca_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sca_5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sca_7 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        # e = cv2.Canny(x, threshold1=100, threshold2=200)

        x = inp
        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        # ----------- multi-scale -----------
        x3 = self.conv2_3(x)
        x3 = self.sg(x3)
        x3 = self.sca_3(x3)

        # x5 = self.conv2_5(x)
        # x5 = self.sg(x5)
        # x5 = self.sca_5(x5)
        #
        # x7 = self.conv2_7(x)
        # x7 = self.sg(x7)
        # x7 = self.sca_7(x7)
        #
        # # x = x3 + x5 + x7
        # x = torch.cat([x3, x5, x7], dim=1)
        # x = self.blending(x)
        # # -----------------------------------
        # # x = self.sg(x)
        # # x = x * self.sca(x)
        # x = self.conv3(x)

        x = self.dropout1(x3)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time

# --------------------------------------------------------------------------------------------------
# --------------------------------------- RCAB modules----------------------------------------------
# --------------------------------------------------------------------------------------------------
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.残差通道注意力块
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """
    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """
    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x














# --------------------------------------------------------------------------------------------------
# ---------------------------------------Conditional NAF Net----------------------------------------
# --------------------------------------------------------------------------------------------------
class ConditionalNAFNet(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1, n_classes=1,
                 dims=[32,64,128, 256],
                 norm_layer=nn.LayerNorm, patch_norm=True):  # width是模型的基础通道数，也是编码器和解码器中特征通道的初始数量，upscale是上采样倍数，默认为1
        super().__init__()
        self.upscale = upscale
        self.n_classes = n_classes
        fourier_dim = width  # fourier_dim是用于时间嵌入的维度，与模型的基础通道数相同
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)  # 是一个用于添加傅立叶基的位置编码器，用于将时间信息嵌入到模型中
        time_dim = width * 4  # 是时间信息的维度，是基础通道数的4倍

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim * 2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )
        # 初始卷积层，intro是一个卷积层，用于将输入的图像特征进行初步处理，将通道数变为width
        self.intro = nn.Conv2d(in_channels=img_channel * 2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        # 中间增强块，enhance是一个残差组模块，包含了6个残差通道注意力块（RCAB），用于增强特征表示
        self.enhance = ResidualGroup(num_feat=width, num_block=6)
        # 最终卷积，将通道数由width变为img_channel，一般为3
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        # 编码部分
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # self.ega = EGA(256)  # =================================================================================
        # self.out = Out(3, 1)



        chan = width  # chan是一个变量，用于跟踪当前编码器的通道数
        # down
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        # mid  middle_blks是一个包含了多个NAFBlock的序列，用于在编码器和解码器之间进行特征融合和特征增强
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
            )
        # up
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward_features(self, x, fe):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x, fe)
        return x, skip_list

    def forward(self, inp, cond, time):
        c0 = inp.clone()  # 复制输入张量inp    torch.Size([1, 3, 256, 256])

        # 检查time的类型，如果是int或float类型，则将其转换为torch.tensor类型，并移动到与inp相同的设备上
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)

        # grayscale_image = rgb_to_grayscale(c0)  # ==================================================================
        # # print(grayscale_image.shape)   # torch.Size([2, 1, 256, 256])
        # edge_feature = make_laplace_pyramid(grayscale_image,5, 1)
        # edge_feature = edge_feature[1]   # torch.Size([2, 1, 128, 128])

        x = inp - cond  # 输入张量inp与条件张量cond做差  # torch.Size([2, 3, 256, 256])    cond=torch.Size([2, 3, 256, 256])

        # pred = self.out(x) # torch.Size([2, 1, 256, 256])

        # edg_c0 = self.ega(edge_feature, x, pred)
        # print(pred.shape)
        # if c0.is_cuda:
        #     c0 =c0.cpu()
        # c0 = c0.numpy()    # (2, 3, 256, 256)
        # # print(c0.shape)
        # gray_c0 = cv2.cvtColor(c0, cv2.COLOR_RGB2BGR)
        #
        # # c0 = rgb_to_grayscale(c0)
        # cond0 = cv2.Canny(cv2.convertScaleAbs(gray_c0*255), threshold1=100, threshold2=200)   # (2, 3)
        # # print(cond.shape)
        # # transform = transforms.ToTensor()
        # # cond = transform(cond)  # torch.Size([1, 2, 3])
        # expanded_cond = np.broadcast_to(cond0[:,:,np.newaxis,np.newaxis], (2,3,256,256))
        # tensor_cond = torch.from_numpy(expanded_cond).float()
        # # print(cond.shape)
        # cond0 = tensor_cond.to('cuda')
        # print(cond.shape)
        cond0 = cv2.laplacian(cond, cv2.CV_64F)

       # x = x + cond
        x = torch.cat([x, cond0], dim=1)  # 将x和cond在通道维度上拼接起来   # torch.Size([1, 6, 256, 256])

        # fe = self.EDGE(cond)
        # x, skip_list = self.forward_features(x,fe)

        t = self.time_mlp(time)  # 通过time_mlp模块处理时间信息time，得到时间编码张量t   # torch.Size([1, 256])

        B, C, H, W = x.shape  # 获取x的形状信息
        x = self.check_image_size(x)  # 检查图像的尺寸是否符合要求  # torch.Size([1, 6, 256, 256])

        x = self.intro(x)  # 将拼接后的张量x送入初始卷积层intro进行初步处理，得到特征张量x   # torch.Size([1, 64, 256, 256])

        # RCAB enhance
        x = self.enhance(x)  # 将特征张量x输入到中间增强块enhance中，用于增强特征表示   # torch.Size([1, 64, 256, 256])

        encs = []

        for encoder, down in zip(self.encoders, self.downs):  # 遍历编码器列表encoders和下采样层列表downs，对每个编码器和下采样层进行处理
            # c0 = self.process_cond(c0, x)  # c0 高度和宽度设为与 x 相同==============================================
            # assert x.shape[2:] == c0.shape[2:]  # ====================================================================
            # x_last_channels = x[:, -3:, :, :]  # =======================================================================
            # x[:, -3:, :, :] = c0  # ==================================================================================
            cond = self.process_cond(cond, x)  # cond 高度和宽度设为与 x 相同==============================================
            assert x.shape[2:] == cond.shape[2:]  # ====================================================================
            x_last_channels = x[:, -3:, :, :]  # =======================================================================
            x[:, -3:, :, :] = cond  # ==================================================================================
            x, _ = encoder([x, t])  # 将x和t送入编码器中进行处理，得到编码后的特征张量x        torch.Size([1, 64, 256, 256])    [1, 128, 128, 128]  [1, 256, 64, 64]  [1, 512, 32, 32]
            encs.append(x)  # 将编码后的x存储到encs列表中，以备后续使用
            x = down(x)  # 将编码后的特征张量x送入下采样层进行尺寸减小，得到下采样后的特征张量x   torch.Size([1, 128, 128, 128])  [1, 256, 64, 64]    [1, 512, 32, 32]   [1, 1024, 16, 16]


        x, _ = self.middle_blks([x, t])  # 下采样后的特征张量x和时间编码张量t送入中间块middle_blks中进行处理，得到经过中间块处理后的特征张量x

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)  # 将特征张量x经过上采样层进行尺寸恢复，得到上采样后的特征张量x
            x = x + enc_skip  # 将上采样后的特征张量x与对应编码器的特征张量enc_skip相加，实现跳跃连接
            x, _ = decoder([x, t])  # 将加和后的特征张量x和时间编码张量t送入解码器中进行处理，得到解码后的特征张量x

        x = self.ending(x)  # 将解码后的特征张量x送入结束层ending进行处理，得到生成的输出张量

        x = x[..., :H, :W]  # 对生成的输出张量进行裁剪，使其尺寸与输入图像的尺寸一致

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def process_cond(self, cond, inp):
        # 对条件进行处理以匹配输入的尺寸
        cond = F.interpolate(cond, size=(inp.size(2), inp.size(3)), mode='bilinear', align_corners=False)
        return cond



if __name__ == '__main__':
    # input = torch.rand(1, 3, 128, 128).cuda()  # B C H W
    # model = ConditionalNAFNet(width=64, enc_blk_nums=[1,1,1,1], dec_blk_nums=[1,1,1,1]).cuda()
    model = RCAB(num_feat=64)
    print_network(model)







# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, reduce
#
# from .module_util import SinusoidalPosEmb, LayerNorm, exists
#
#
# class SimpleGate(nn.Module):
#     def forward(self, x):
#         x1, x2 = x.chunk(2, dim=1)
#         return x1 * x2
#
# def print_network(net):
#     num_params = 0
#     for param in net.parameters():
#         num_params += param.numel()
#     print(net)
#     print('Total number of parameters: %f M' % (num_params / 1e6))
#
# class NAFBlock(nn.Module):
#     def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
#         ) if time_emb_dim else None
#
#         dw_channel = c * DW_Expand
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#
#         self.conv2_3 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
#                                bias=True)
#         self.conv2_5 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=5, padding=2, stride=1,
#                                  groups=dw_channel,
#                                  bias=True)
#         self.conv2_7 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=7, padding=3, stride=1,
#                                  groups=dw_channel,
#                                  bias=True)
#
#         self.blending = nn.Conv2d(in_channels=dw_channel*3 // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#
#         self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#
#         # Simplified Channel Attention+
#
#         # self.sca = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d(1),
#         #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#         #               groups=1, bias=True),
#         # )
#
#         self.sca_3 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=True),
#         )
#         self.sca_5 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=True),
#         )
#         self.sca_7 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=True),
#         )
#
#         # SimpleGate
#         self.sg = SimpleGate()
#
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#
#         self.norm1 = LayerNorm(c)
#         self.norm2 = LayerNorm(c)
#
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#
#     def time_forward(self, time, mlp):
#         time_emb = mlp(time)
#         time_emb = rearrange(time_emb, 'b c -> b c 1 1')
#         return time_emb.chunk(4, dim=1)
#
#     def forward(self, x):
#         inp, time = x
#         shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)
#
#         x = inp
#
#         x = self.norm1(x)
#         x = x * (scale_att + 1) + shift_att
#         x = self.conv1(x)
#         # ----------- multi-scale -----------
#         x3 = self.conv2_3(x)
#         x3 = self.sg(x3)
#         x3 = self.sca_3(x3)
#
#         x5 = self.conv2_5(x)
#         x5 = self.sg(x5)
#         x5 = self.sca_5(x5)
#
#         x7 = self.conv2_7(x)
#         x7 = self.sg(x7)
#         x7 = self.sca_7(x7)
#
#         # x = x3 + x5 + x7
#         x = torch.cat([x3, x5, x7], dim=1)
#         x = self.blending(x)
#         # -----------------------------------
#         # x = self.sg(x)
#         # x = x * self.sca(x)
#         x = self.conv3(x)
#
#         x = self.dropout1(x)
#
#         y = inp + x * self.beta
#
#         x = self.norm2(y)
#         x = x * (scale_ffn + 1) + shift_ffn
#         x = self.conv4(x)
#         x = self.sg(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#
#         x = y + x * self.gamma
#
#         return x, time
# # --------------------------------------- RCAB modules-----------------------------------------------------------
# def make_layer(basic_block, num_basic_block, **kwarg):
#     """Make layers by stacking the same blocks.
#
#     Args:
#         basic_block (nn.module): nn.module class for basic block.
#         num_basic_block (int): number of blocks.
#
#     Returns:
#         nn.Sequential: Stacked blocks in nn.Sequential.
#     """
#     layers = []
#     for _ in range(num_basic_block):
#         layers.append(basic_block(**kwarg))
#     return nn.Sequential(*layers)
# class ChannelAttention(nn.Module):
#     """Channel attention used in RCAN.
#     Args:
#         num_feat (int): Channel number of intermediate features.
#         squeeze_factor (int): Channel squeeze factor. Default: 16.
#     """
#
#     def __init__(self, num_feat, squeeze_factor=16):
#         super(ChannelAttention, self).__init__()
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
#             nn.Sigmoid())
#
#     def forward(self, x):
#         y = self.attention(x)
#         return x * y
#
# class RCAB(nn.Module):
#     """Residual Channel Attention Block (RCAB) used in RCAN.
#
#     Args:
#         num_feat (int): Channel number of intermediate features.
#         squeeze_factor (int): Channel squeeze factor. Default: 16.
#         res_scale (float): Scale the residual. Default: 1.
#     """
#
#     def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
#         super(RCAB, self).__init__()
#         self.res_scale = res_scale
#
#         self.rcab = nn.Sequential(
#             nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
#             ChannelAttention(num_feat, squeeze_factor))
#
#     def forward(self, x):
#         res = self.rcab(x) * self.res_scale
#         return res + x
#
# class ResidualGroup(nn.Module):
#     """Residual Group of RCAB.
#
#     Args:
#         num_feat (int): Channel number of intermediate features.
#         num_block (int): Block number in the body network.
#         squeeze_factor (int): Channel squeeze factor. Default: 16.
#         res_scale (float): Scale the residual. Default: 1.
#     """
#
#     def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
#         super(ResidualGroup, self).__init__()
#
#         self.residual_group = make_layer(
#             RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
#         self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#
#     def forward(self, x):
#         res = self.conv(self.residual_group(x))
#         return res + x
#
# # ---------------------------------------- -----------------------------------------------------------------------
#
# class ConditionalNAFNet(nn.Module):
#
#     def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1):
#         super().__init__()
#         self.upscale = upscale
#         fourier_dim = width
#         sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
#         time_dim = width * 4
#
#         self.time_mlp = nn.Sequential(
#             sinu_pos_emb,
#             nn.Linear(fourier_dim, time_dim*2),
#             SimpleGate(),
#             nn.Linear(time_dim, time_dim)
#         )
#
#         self.intro = nn.Conv2d(in_channels=img_channel*2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
#                               bias=True)
#
#         self.enhance = ResidualGroup(num_feat=width, num_block=6)
#
#         self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
#                               bias=True)
#
#         self.encoders = nn.ModuleList()
#         self.decoders = nn.ModuleList()
#         self.middle_blks = nn.ModuleList()
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()
#
#         chan = width
#         for num in enc_blk_nums:
#             self.encoders.append(
#                 nn.Sequential(
#                     *[NAFBlock(chan, time_dim) for _ in range(num)]
#                 )
#             )
#             self.downs.append(
#                 nn.Conv2d(chan, 2*chan, 2, 2)
#             )
#             chan = chan * 2
#
#         self.middle_blks = \
#             nn.Sequential(
#                 *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
#             )
#
#         for num in dec_blk_nums:
#             self.ups.append(
#                 nn.Sequential(
#                     nn.Conv2d(chan, chan * 2, 1, bias=False),
#                     nn.PixelShuffle(2)
#                 )
#             )
#             chan = chan // 2
#             self.decoders.append(
#                 nn.Sequential(
#                     *[NAFBlock(chan, time_dim) for _ in range(num)]
#                 )
#             )
#
#         self.padder_size = 2 ** len(self.encoders)
#
#     def forward(self, inp, cond, time):
#         inp_res = inp.clone()
#
#         if isinstance(time, int) or isinstance(time, float):
#             time = torch.tensor([time]).to(inp.device)
#
#         x = inp - cond
#         x = torch.cat([x, cond], dim=1)
#
#         t = self.time_mlp(time)
#
#         B, C, H, W = x.shape
#         x = self.check_image_size(x)
#
#         x = self.intro(x)
#
#         # RCAB enhance
#         x = self.enhance(x)
#
#         encs = []
#
#         for encoder, down in zip(self.encoders, self.downs):
#             x, _ = encoder([x, t])
#             encs.append(x)
#             x = down(x)
#
#         x, _ = self.middle_blks([x, t])
#
#         for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
#             x = up(x)
#             x = x + enc_skip
#             x, _ = decoder([x, t])
#
#         x = self.ending(x)
#
#         x = x[..., :H, :W]
#
#         return x
#
#     def check_image_size(self, x):
#         _, _, h, w = x.size()
#         mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
#         mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
#         x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
#         return x
#
# if __name__ == '__main__':
#     # input = torch.rand(1, 3, 128, 128).cuda()  # B C H W
#     # model = ConditionalNAFNet(width=64, enc_blk_nums=[14,1,1,1], dec_blk_nums=[1,1,1,1]).cuda()
#     model = RCAB(num_feat=64)
#     print_network(model)