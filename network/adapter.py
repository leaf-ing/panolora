import math
import torch


class Adapter(torch.nn.Module):
    def __init__(self, adapter_name, org_module: torch.nn.Module, adapter_dim: int):
        super().__init__()
        self.adapter_name = adapter_name
        self.org_module = org_module
        self.adapter_dim = adapter_dim

        if org_module.__class__.__name__ == 'Attention':
            self.feature_dim = org_module.to_q.in_features
            self.position = "Attention"
        elif org_module.__class__.__name__ == 'FeedForward':
            self.feature_dim = org_module.net[2].out_features
            self.position = "FeedForward"

        self.down_proj = torch.nn.Linear(self.feature_dim, self.adapter_dim)
        self.nonlinear = torch.nn.SiLU()
        self.up_proj = torch.nn.Linear(self.adapter_dim, self.feature_dim)

        # init weight use zero-mean Gaussian distribution with std=0.01
        torch.nn.init.normal_(self.down_proj.weight, std=0.01)
        torch.nn.init.zeros_(self.up_proj.weight)
        torch.nn.init.zeros_(self.up_proj.bias)

    def apply(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        if self.position == "FeedForward":
            h = self.org_forward(hidden_states)
        else:
            h = self.org_forward(hidden_states, encoder_hidden_states=encoder_hidden_states,attention_mask=attention_mask, **cross_attention_kwargs)
        delta_h = self.up_proj(self.nonlinear(self.down_proj(h)))
        return h + delta_h


class AdapterDiffusion(torch.nn.Module):
    def __init__(self, unet, adapter_dim: int):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.unet_adapters = self.set_unet_adapter(unet)

    def set_unet_adapter(self, unet):
        unet_adapters = []
        for name, module in unet.named_modules():
            if module.__class__.__name__ in ['Attention', 'FeedForward']:
                adapter_name = 'adapter.' + name
                adapter_name = adapter_name.replace('.', '_')
                adapter = Adapter(adapter_name, module, self.adapter_dim)
                adapter.apply()
                unet_adapters.append(adapter)
                self.add_module(adapter.adapter_name, adapter)
        return unet_adapters

    def save_weights(self, save_path):
        torch.save(self.state_dict(), save_path)
