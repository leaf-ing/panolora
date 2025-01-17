import math
import torch.optim.lr_scheduler as lr_scheduler
from diffusers import StableDiffusionPipeline

sd_path = 'D:/Desktop/Diffusion/MyDiffusion/models/stable-diffusion-v1-5'
def load_diffusion_model():
    pipe = StableDiffusionPipeline.from_pretrained(sd_path)
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    del pipe
    return unet, text_encoder, vae


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v == 'True'


def warmup_cosin_schedule(optimizer, warmup_steps, t_total, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(current_step - warmup_steps) / float(max(1, t_total - warmup_steps))))
        )
    return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
