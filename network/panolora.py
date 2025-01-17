'''
reference:
https://github.com/ChiWeiHsiao/SphereNet-pytorch/blob/master/spherenet/sphere_cnn.py
'''
import torch
import torch.nn as nn
from numpy import tan, pi, arcsin, arctan, cos, sin
import numpy as np
from torchvision.ops import deform_conv2d
import torch.nn.functional as F
from torch.nn import Conv2d
import itertools


def get_xy(delta_phi, delta_theta):
    return np.array([
        [
            (-tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi)),
            (0, tan(delta_phi)),
            (tan(delta_theta), 1 / cos(delta_theta) * tan(delta_phi)),
        ],
        [
            (-tan(delta_theta), 0),
            (1, 1), # 这个值最后会被赋值为0，所以无所谓
            (tan(delta_theta), 0),
        ],
        [
            (-tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi)),
            (0, -tan(delta_phi)),
            (tan(delta_theta), -1 / cos(delta_theta) * tan(delta_phi)),
        ]
    ])


def cal_offset(h, w, img_r, img_c):
    '''
        Calculate Kernel Sampling Pattern
        only support 3x3 filter
        return 9 locations: (3, 3, 2)
    '''
    # pixel -> rad
    phi = -((img_r + 0.5) / h * pi - pi / 2)
    theta = (img_c + 0.5) / w * 2 * pi - pi

    delta_phi = pi / h
    delta_theta = 2 * pi / w

    xys = get_xy(delta_phi, delta_theta)
    x = xys[..., 0]
    y = xys[..., 1]
    rho = np.sqrt(x ** 2 + y ** 2)
    v = arctan(rho)
    new_phi = arcsin(cos(v) * sin(phi) + y * sin(v) * cos(phi) / rho)
    new_theta = theta + arctan(x * sin(v) / (rho * cos(phi) * cos(v) - y * sin(phi) * sin(v)))
    # rad -> pixel
    new_r = (-new_phi + pi / 2) * h / pi - 0.5
    new_c = (new_theta + pi) * w / 2 / pi - 0.5
    # indexs out of image, equirectangular leftmost and rightmost pixel is adjacent
    new_c = (new_c + w) % w
    new_result = np.stack([new_r, new_c], axis=-1)
    new_result[1, 1] = (img_r, img_c)
    for i in range(3):
        for j in range(3):
            new_result[i, j, 0] = new_result[i, j, 0] - (img_r + i - 1)
            new_result[i, j, 1] = new_result[i, j, 1] - (img_c + j - 1)
    # reshape to (18)
    new_result = new_result.reshape(-1)
    return new_result


def offsets_map(h, w):
    # shape: (h, w, 18)
    co = np.array([[cal_offset(h, w, i, j) for j in range(0, w)] for i in range(0, h)])
    # shape: (18, h, w)
    co = co.transpose(2, 0, 1)
    return co


class SelfAttention_QKLoRA(nn.Module):
    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier: float = 1.0, low_factor=1):
        super().__init__()
        self.lora_name = lora_name

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.lora_dim = max(in_dim // low_factor, 1)
        self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_up.weight)
        self.multiplier = multiplier
        self.org_module = org_module
        self.apply()

    def apply(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        h = self.org_forward(x)
        # lora输出
        delta_h = self.lora_up(self.lora_down(x))
        delta_h = delta_h * self.scale * self.multiplier
        return h + delta_h


class SphereLoRA(nn.Module):
    def __init__(self, sphere_lora_name, init_conv: torch.nn.Module, out_h=1, out_w=1, low_factor=1):
        super().__init__()
        self.sphere_lora_name = sphere_lora_name
        # 以下都是原有的普通卷积的相关参数，用于球面卷积的设定
        self.in_dim = init_conv.in_channels
        self.stride = init_conv.stride
        self.padding = init_conv.padding
        self.dilation = init_conv.dilation
        self.out_dim = init_conv.out_channels
        self.kernel_size = init_conv.kernel_size
        # 以下为球面卷积的参数，即对应的weight和bias
        self.low_factor = low_factor
        self.weight = nn.Parameter(torch.empty(self.out_dim, self.in_dim, self.kernel_size[0], self.kernel_size[1]), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(self.out_dim), requires_grad=False)
        # 这里是lora
        self.conv_down = Conv2d(2 * self.out_dim, max(self.out_dim // self.low_factor, 1), kernel_size=(1, 1),
                                padding=(0, 0), bias=False, groups=1)
        self.conv_up = Conv2d(max(self.out_dim // self.low_factor, 1), self.out_dim, kernel_size=(1, 1),
                                padding=(0, 0), bias=False, groups=1)
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # get untrainable offset，由ERP投影公式和对应位置的tensor形状计算获得
        self.offset = torch.from_numpy(offsets_map(out_h, out_w)).float()
        # 初始化球面卷积的参数，这部分不会训练
        self.org_module = init_conv
        self.weight.data = init_conv.weight.data
        self.bias.data = init_conv.bias.data
        # 零初始化lora，这部分需要训练
        torch.nn.init.kaiming_uniform_(self.conv_down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.conv_up.weight)
        # 冻结卷积参数
        self.frozen_conv_weights()
        # 激活lora参数
        self.conv_down.requires_grad_(True)
        self.conv_up.requires_grad_(True)
        # apply
        self.apply()

    def apply(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def frozen_conv_weights(self):
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.offset.requires_grad = False

    # frozen_conv_weights后，这里只会获得lora参数
    def train_parm(self):
        parm = (p for n, p in self.named_parameters() if p.requires_grad)
        return parm

    def forward(self, x):
        bs, _, __, ___ = x.shape
        # add batch dim to offset
        offset = self.offset.unsqueeze(0).repeat(bs, 1, 1, 1).to(x.device)
        # org forward result
        h = self.org_forward(x)
        # 并行的sphere_lora
        dx = deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation)
        dx = torch.cat([dx, h], dim=1)
        return h + self.scale * self.conv_up(self.conv_down(dx))


class PanoLoRA_Diffusion(torch.nn.Module):
    def __init__(self, unet, low_factor=1):
        super().__init__()
        # 各层卷积tensor对应的形状
        self.out_h_list = [64] * 5 + [32] * 5 + [16] * 5 + [8] * 11 + [16] * 7 + [32] * 7 + [64] * 7 + [8] * 4 + [64]
        self.low_factor = low_factor
        self.multiplier = 1.0
        self.sphere_loras = self.set_deformable_conv(unet)
        self.loras = self.set_loras(unet)

    def set_deformable_conv(self, unet):
        sphere_loras = []
        i = 0
        for name, module in unet.named_modules():
            is_3x3conv = module.__class__.__name__ == "Conv2d" and module.kernel_size == (3, 3)
            if is_3x3conv:
                offset_name = "sphere_lora_unet." + name
                offset_name = offset_name.replace('.', '_')
                offset_module = SphereLoRA(offset_name, module, out_h=self.out_h_list[i],
                                            out_w=2 * self.out_h_list[i], low_factor=self.low_factor)
                sphere_loras.append(offset_module)
                self.add_module(offset_name, offset_module)
                i += 1
        return sphere_loras

    def set_loras(self, unet):
        unet_loras = []
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "Transformer2DModel":
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_SA_qk = "attn1" in child_name and ("to_q" in child_name or "to_k" in child_name)
                    if is_SA_qk and is_linear:
                        lora_name = 'lora_unet.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        lora_module = SelfAttention_QKLoRA(lora_name, child_module, self.multiplier, self.low_factor)
                        unet_loras.append(lora_module)
                        self.add_module(lora_module.lora_name, lora_module)
        return unet_loras

    def get_train_parm(self):
        params = (p for n, p in self.named_parameters() if p.requires_grad)
        return params

    def save_weights(self, save_path):
        param_dict = {}
        bank_names = ["conv_down", "conv_up", "scale", "lora_down", "lora_up"]
        for name, param in self.named_parameters():
            for bank_name in bank_names:
                if bank_name in name:
                    param_dict[name] = param
                    break
        torch.save(param_dict, save_path)