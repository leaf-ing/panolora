import torch

def bitFit_prepare(unet):
    for name, param in unet.named_parameters():
        param.requires_grad_(False)
    # unfreeze bias
    for name, param in unet.named_parameters():
        if "bias" in name:
            param.requires_grad_(True)
    # get trainable parameters
    params = (p for n,p in unet.named_parameters() if "bias" in n and p.requires_grad)
    return params

def bitFit_save(unet,save_path):
    bias = {}
    for name, param in unet.named_parameters():
        if "bias" in name:
            bias[name] = param
    # save all bias to .ckpt
    torch.save(bias, save_path)

def bino_prepare(unet):
    for name, param in unet.named_parameters():
        param.requires_grad_(False)
    # unfreeze bias
    for name, param in unet.named_parameters():
        if "bias" in name or "norm" in name:
            param.requires_grad_(True)
    # get trainable parameters
    params = (p for n,p in unet.named_parameters() if p.requires_grad)
    return params

def bino_save(unet,save_path):
    bias = {}
    for name, param in unet.named_parameters():
        if "bias" in name or "norm" in name:
            bias[name] = param
    # save all bias to .ckpt
    torch.save(bias, save_path)