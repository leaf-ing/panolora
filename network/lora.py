# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# https://github.com/bmaltais/kohya_ss/blob/master/networks/lora.py

import math
import torch


class LoRA(torch.nn.Module):
    '''
    更改forward，而非【替代Linear+重新load参数】
    '''

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier: float = 1.0, lora_dim: int = 4,
                 alpha: float = 1.0, trainable_scale: bool = True):
        super().__init__()

        self.lora_name = lora_name
        self.lora_dim = lora_dim

        # 如果是1x1的卷积，可以视为Linear，使用Lora，此时利用1x1的卷积作为Lora
        if org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            # Lora的Bias是False
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        # 如果是Linear，使用Lora
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        if not trainable_scale:
            self.scale = alpha / self.lora_dim
            self.register_buffer('alpha', torch.tensor(alpha))
        else:
            # init scale to 1.0, and this will be learned
            self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # 初始化，如Lora论文所述，其中lora_up的权重初始化为0，保证初始时，Lora网络的输出为0
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        self.multiplier = multiplier
        self.org_module = org_module

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


# LoraDiffusion
# LoRA is only used in UNet
class LoraDiffusion(torch.nn.Module):
    def __init__(self, unet, text_encoder=None, multiplier=1.0,
                 unet_dim=4, unet_alpha=1.0, text_encoder_dim=4, text_encoder_alpha=1.0, trainable_scale=True):
        super().__init__()
        self.multiplier = multiplier
        self.unet_dim = unet_dim
        self.unet_alpha = unet_alpha
        self.text_encoder_dim = text_encoder_dim
        self.text_encoder_alpha = text_encoder_alpha
        self.trainable_scale = trainable_scale
        if unet is not None:
            self.unet_loras = self.set_unet_loras(unet)
            print("set unet loras")
        if text_encoder is not None:
            self.text_encoder_loras = self.set_text_encoder_loras(text_encoder)
            print("set text encoder loras")

    def set_unet_loras(self, unet):
        unet_loras = []
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "Transformer2DModel":
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_1x1conv = child_module.__class__.__name__ == "Conv2d" and child_module.kernel_size == (1, 1)
                    if is_linear or is_1x1conv:
                        lora_name = 'lora_unet.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        lora_module = LoRA(lora_name, child_module, self.multiplier, self.unet_dim, self.unet_alpha,
                                            self.trainable_scale)
                        lora_module.apply()
                        unet_loras.append(lora_module)
                        self.add_module(lora_module.lora_name, lora_module)

        return unet_loras

    # not used
    def set_text_encoder_loras(self, text_encoder):
        text_encoder_loras = []
        replace_module_names = ["CLIPAttention", "CLIPMLP"]
        for name, module in text_encoder.named_modules():
            if module.__class__.__name__ in replace_module_names:
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ == "Linear"
                    if is_linear:
                        lora_name = 'lora_te.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        lora_module = LoRA(lora_name, child_module, self.multiplier, self.text_encoder_dim,
                                           self.text_encoder_alpha, self.trainable_scale)
                        lora_module.apply()
                        text_encoder_loras.append(lora_module)
                        self.add_module(lora_module.lora_name, lora_module)
        return text_encoder_loras

    def save_weights(self, save_path):
        torch.save(self.state_dict(), save_path)