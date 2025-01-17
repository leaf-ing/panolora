import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from network.panolora import offsets_map,SelfAttention_QKLoRA
from torch.nn import Conv2d


class AblationSphereLoRA(nn.Module):
    def __init__(self, ablation_name, init_conv: torch.nn.Module, out_h=1, out_w=1, low_factor=1, mix="None",
                 ablation="none"):
        '''
        ablation:
            - wo_panoconv : 使用普通卷积【因此不使用cat】
            - wo_cat : 等价于mix == None
            - wo_copy : 球面卷积使用随机初始化
        '''
        super().__init__()
        self.ablation_name = ablation_name
        self.in_dim = init_conv.in_channels
        self.stride = init_conv.stride
        self.padding = init_conv.padding
        self.dilation = init_conv.dilation
        self.out_dim = init_conv.out_channels
        self.ablation = ablation
        self.mix = mix
        self.groups = init_conv.groups
        self.out_h = out_h
        self.out_w = out_w
        self.kernel_size = init_conv.kernel_size
        self.low_factor = low_factor
        self.ablation_weight = nn.Parameter(
            torch.empty(self.out_dim, self.in_dim, self.kernel_size[0], self.kernel_size[1]), requires_grad=False)
        self.ablation_bias = nn.Parameter(torch.empty(self.out_dim), requires_grad=False)

        if self.mix == "cat":
            self.conv_down = Conv2d(2 * self.out_dim, max(self.out_dim // self.low_factor, 1), kernel_size=(1, 1),
                                    padding=(0, 0), bias=False, groups=1)
            self.conv_up = Conv2d(max(self.out_dim // self.low_factor, 1), self.out_dim, kernel_size=(1, 1),
                                  padding=(0, 0), bias=False, groups=1)
        # wo_cat
        else:
            self.conv_down = Conv2d(self.out_dim, max(self.out_dim // self.low_factor, 1), kernel_size=(1, 1),
                                    padding=(0, 0), bias=False, groups=1)
            self.conv_up = Conv2d(max(self.out_dim // self.low_factor, 1), self.out_dim, kernel_size=(1, 1),
                                  padding=(0, 0), bias=False, groups=1)
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.org_module = init_conv
        # zero_init conv1x1
        torch.nn.init.kaiming_uniform_(self.conv_down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.conv_up.weight)
        # get untrainable offset
        if self.ablation!="wo_panoconv":
            self.offset = torch.from_numpy(offsets_map(self.out_h, self.out_w)).float()
            # init
            if self.ablation!="wo_copy":
                self.ablation_weight.data = init_conv.weight.data
                self.ablation_bias.data = init_conv.bias.data
            else:
                torch.nn.init.kaiming_uniform_(self.ablation_weight,a=5**0.5)
                torch.nn.init.normal_(self.ablation_bias,std=0.01)
            # frozen
            self.frozen_conv_weights()
            # activate
        self.conv_down.requires_grad_(True)
        self.conv_up.requires_grad_(True)
            # apply
        self.apply()

    def apply(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def frozen_conv_weights(self):
        self.ablation_weight.requires_grad = False
        self.ablation_bias.requires_grad = False
        self.offset.requires_grad = False

    def train_parm(self):
        parm = (p for n, p in self.named_parameters() if p.requires_grad)
        return parm

    def forward(self, x):
        if self.ablation!="wo_panoconv":
            bs, _, __, ___ = x.shape
            # add batch dim to offset
            offset = self.offset.unsqueeze(0).repeat(bs, 1, 1, 1).to(x.device)
            h = self.org_forward(x)
            dx = deform_conv2d(x, offset, self.ablation_weight, self.ablation_bias, self.stride, self.padding, self.dilation)
            if self.mix == "cat":   # 不会进入这个分支
                dx = torch.cat([dx, h], dim=1)
            return h + self.scale * self.conv_up(self.conv_down(dx))
        else:
            h = self.org_forward(x)
            dx = self.conv_up(self.conv_down(h))
            return h + self.scale * dx


class AblationDiffusion(torch.nn.Module):
    def __init__(self, unet , low_factor=1, mix="None", ablation="None"):
        '''
        ablation:
            - wo_conv : 只有Self-Attention的Q、K Lora
            - wo_lora : 只有球面卷积Lora
            - wo_panoconv : 使用普通卷积【因此不使用cat】
            - wo_cat : 等价于mix == None
            - wo_copy : 球面卷积使用随机初始化
        '''
        super().__init__()
        self.out_h_list = [64] * 5 + [32] * 5 + [16] * 5 + [8] * 11 + [16] * 7 + [32] * 7 + [64] * 7 + [8] * 4 + [64]
        self.low_factor = low_factor
        self.multiplier = 1.0
        self.mix = mix
        self.ablation = ablation
        if self.ablation == "wo_panoconv" or self.ablation == "wo_cat":
            self.mix = "None"
        if self.ablation != "wo_conv":
            self.ablations = self.set_deformable_conv(unet)
        if self.ablation != "wo_lora":
            self.loras = self.set_loras(unet)

    def set_deformable_conv(self, unet):
        ablations = []
        i = 0
        for name, module in unet.named_modules():
            is_3x3conv = module.__class__.__name__ == "Conv2d" and module.kernel_size == (3, 3)
            if is_3x3conv:
                offset_name = "ablation_unet." + name
                offset_name = offset_name.replace('.', '_')
                offset_module = AblationSphereLoRA(offset_name, module, out_h=self.out_h_list[i],
                                                    out_w=2 * self.out_h_list[i], low_factor=self.low_factor,
                                                    mix=self.mix, ablation=self.ablation)
                ablations.append(offset_module)
                self.add_module(offset_name, offset_module)
                i += 1
        return ablations

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
        if self.ablation == "wo_copy":
            bank_names.append("ablation_weight")
            bank_names.append("ablation_bias")
        for name, param in self.named_parameters():
            for bank_name in bank_names:
                if bank_name in name:
                    param_dict[name] = param
                    break
        torch.save(param_dict, save_path)
