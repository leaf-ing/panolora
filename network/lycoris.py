# lycoris network module
# reference:
# https://github.com/microsoft/lycoris/blob/main/lycorislib/layers.py
# https://github.com/cloneofsimo/lycoris/blob/master/lycoris_diffusion/lycoris.py
# https://github.com/bmaltais/kohya_ss/blob/master/networks/lycoris.py

import torch


class Lycoris(torch.nn.Module):
    '''
    更改forward，而非【替代Linear+重新load参数】
    '''
    def __init__(self, lycoris_name, org_module: torch.nn.Module, multiplier: float = 1.0, lycoris_dim: int = 4,
                 alpha: float = 1.0, trainable_scale: bool = True):
        super().__init__()

        self.lycoris_name = lycoris_name
        self.lycoris_dim = lycoris_dim

        # Locon，这里和LoRA的区别在于对3x3的卷积使用的也是3x3的lycoris_down和lycoris_up
        if org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            # lycoris的Bias是False
            self.lycoris_down = torch.nn.Conv2d(in_dim, self.lycoris_dim, kernel_size, stride, padding, bias=False)
            self.lycoris_up = torch.nn.Conv2d(self.lycoris_dim, out_dim, kernel_size, (1, 1),
                                              1 if kernel_size == (3, 3) else 0, bias=False)
            # 如果是Linear，使用lycoris
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lycoris_down = torch.nn.Linear(in_dim, self.lycoris_dim, bias=False)
            self.lycoris_up = torch.nn.Linear(self.lycoris_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lycoris_dim if alpha is None or alpha == 0 else alpha
        if not trainable_scale:
            self.scale = alpha / self.lycoris_dim
            self.register_buffer('alpha', torch.tensor(alpha))
        else:
            # init scale to 1.0, and this will be learned
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
        # 初始化，如lycoris论文所述，其中lycoris_up的权重初始化为0，保证初始时，lycoris网络的输出为0
        torch.nn.init.kaiming_uniform_(self.lycoris_down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lycoris_up.weight)
        self.multiplier = multiplier
        self.org_module = org_module

    def apply(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        h = self.org_forward(x)
        # lycoris输出
        delta_h = self.lycoris_up(self.lycoris_down(x))
        delta_h = delta_h * self.scale * self.multiplier
        return h + delta_h


# lycorisDiffusion
# lycoris is used in UNet
class LycorisDiffusion(torch.nn.Module):
    def __init__(self, unet, multiplier=1.0, unet_dim=4, unet_alpha=1.0, trainable_scale=True):
        super().__init__()
        self.multiplier = multiplier
        self.unet_dim = unet_dim
        self.unet_alpha = unet_alpha
        self.trainable_scale = trainable_scale
        if unet is not None:
            self.unet_lycoris = self.set_unet_lycoriss(unet)
            print("set unet lycoris")

    def set_unet_lycoriss(self, unet):
        unet_lycoris = []
        for name, module in unet.named_modules():
            is_linear = module.__class__.__name__ == "Linear"
            is_conv = module.__class__.__name__ == "Conv2d"
            if is_linear or is_conv:
                lycoris_name = 'lycoris_unet.' + name + '.' + name
                lycoris_name = lycoris_name.replace('.', '_')
                lycoris_module = Lycoris(lycoris_name, module, self.multiplier, self.unet_dim,
                                         self.unet_alpha, self.trainable_scale)
                lycoris_module.apply()
                unet_lycoris.append(lycoris_module)
                self.add_module(lycoris_module.lycoris_name, lycoris_module)

        return unet_lycoris

    def save_weights(self, save_path):
        torch.save(self.state_dict(), save_path)
