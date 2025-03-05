import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from torch.autograd import Variable
from math import exp


class RGB2HSV(nn.Module):
    def __init__(self):
        super(RGB2HSV, self).__init__()

    def forward(self, rgb):
        batch, c, w, h = rgb.size()
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        V, max_index = torch.max(rgb, dim=1)
        min_rgb = torch.min(rgb, dim=1)[0]
        v_plus_min = V - min_rgb
        S = v_plus_min / (V + 0.0001)
        H = torch.zeros_like(rgb[:, 0, :, :])
        # if rgb.type() == 'torch.cuda.FloatTensor':
        #     H = torch.zeros(batch, w, h).type(torch.cuda.FloatTensor)
        # else:
        #     H = torch.zeros(batch, w, h).type(torch.FloatTensor)
        mark = max_index == 0
        H[mark] = 60 * (g[mark] - b[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 1
        H[mark] = 120 + 60 * (b[mark] - r[mark]) / (v_plus_min[mark] + 0.0001)
        mark = max_index == 2
        H[mark] = 240 + 60 * (r[mark] - g[mark]) / (v_plus_min[mark] + 0.0001)

        mark = H < 0
        H[mark] += 360
        H = H % 360
        H = H / 360
        HSV_img = torch.cat([H.view(batch, 1, w, h), S.view(batch, 1, w, h), V.view(batch, 1, w, h)], 1)
        return HSV_img


class HSVLoss(nn.Module):
    def __init__(self):
        super(HSVLoss, self).__init__()
        self.criterionloss = nn.L1Loss()
        self.pi = math.pi

    def forward(self, pred, gt):
        rgb2hsv = RGB2HSV()
        pred = rgb2hsv(pred)
        gt = rgb2hsv(gt)
        # ----------------------------------------------------------------------------------------#
        hi, si, vi = pred[:, 0:1, :, :], pred[:, 1:2, :, :], pred[:, 2:, :, :]
        hj, sj, vj = gt[:, 0:1, :, :], gt[:, 1:2, :, :], gt[:, 2:, :, :]

        hipi = hi * self.pi * 2
        hjpi = hj * self.pi * 2

        coshp = torch.cos(hipi)
        sinhg = torch.sin(hjpi)
        sv_p = torch.mul(si, vi)
        sv_g = torch.mul(sj, vj)
        temp_pred = torch.mul(coshp, sv_p)
        temp_gt = torch.mul(sinhg, sv_g)

        loss = torch.mean(torch.abs(torch.add(temp_pred, -1, temp_gt)))

        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, loss_weight=1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.loss_weight = loss_weight

    def forward(self, input, target):
        # 确保输入和目标具有相同的尺寸和数据类型
        if input.size() != target.size():
            raise ValueError("Input and target must have the same size.")
        if input.dtype != target.dtype:
            raise ValueError("Input and target must have the same data type.")
        # 创建高斯窗口
        window = create_window(self.window_size, input.size(1))

        # 计算SSIM
        ssim_map = _ssim(input, target, window, self.window_size, input.size(1), self.size_average)
        loss = 1 - ssim_map.mean() if self.size_average else ssim_map.mean(1).mean(1).mean(1)
        loss = self.loss_weight * loss
        return loss


# 加权损失 loss = 0.1 * l1_loss + 0.8 * l2_loss + 0.05 * hsv_loss + 0.05 * ssim_loss    配置文件中把is_weighted改为 True
class MatchingLoss(nn.Module):
    def __init__(self, loss_type=None, is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        self.loss_fn = F.l1_loss
        self.loss_fn = F.mse_loss
        self.hsv_loss = HSVLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, predict, target, weights=None):
        l1_loss = F.l1_loss(predict, target, reduction='none')
        l2_loss = F.mse_loss(predict, target, reduction='none')
        hsv_loss = F.l1_loss(predict, target, reduction='none')
        ssim_loss = F.mse_loss(predict, target, reduction='none')
        loss = 0.1 * l1_loss + 0.8 * l2_loss + 0.05 * hsv_loss + 0.05 * ssim_loss  # 组合损失

        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()

# class MatchingLoss(nn.Module):
#     def __init__(self, loss_type='l1', is_weighted=False):
#         super().__init__()
#         self.is_weighted = is_weighted
#
#         if loss_type == 'l1':
#             self.loss_fn = F.l1_loss
#         elif loss_type == 'l2':
#             self.loss_fn = F.mse_loss
#         else:
#             raise ValueError(f'invalid loss type {loss_type}')
#
#     def forward(self, predict, target, weights=None):
#
#         loss = self.loss_fn(predict, target, reduction='none')
#         loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')
#
#         if self.is_weighted and weights is not None:
#             loss = weights * loss
#
#         return loss.mean()



