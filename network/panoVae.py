import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional

class PanoVae(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.__change_padding_mode(vae)

    def __change_padding_mode(self, vae):
        for name,module in vae.decoder.named_modules():
            is3x3conv = module.__class__.__name__ == "Conv2d" and module.kernel_size == (3, 3)
            if is3x3conv:
                module.padding_modeX="circular"
                module.padding_modeY="reflect"
                module.paddingX=(module._reversed_padding_repeated_twice[0],module._reversed_padding_repeated_twice[1],0,0)
                module.paddingY=(0,0,module._reversed_padding_repeated_twice[2],module._reversed_padding_repeated_twice[3])
                module._conv_forward = PanoVae.__replacementConv2DConvForward.__get__(module, Conv2d)

    def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        working = F.pad(input, self.paddingX, mode='circular')
        working = F.pad(working, self.paddingY, mode='reflect')
        return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)