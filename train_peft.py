# # LoRA network module
# https://github.com/bmaltais/kohya_ss/
import gc
import time
import datetime
import torch.nn.utils
from diffusers import DDPMScheduler
from dataset.base_dataset import BaseDataset
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import os
import argparse
from accelerate import Accelerator
from transformers import CLIPTokenizer
from network.adapter import AdapterDiffusion
from network.bitFit import bino_prepare,bino_save,bitFit_prepare,bitFit_save
from network.lora import LoraDiffusion
from network.panolora import PanoLoRA_Diffusion
from network.lycoris import LycorisDiffusion
from network.ablationDiffusion import AblationDiffusion
from train_helper import load_diffusion_model, str2bool, warmup_cosin_schedule

root_path = './'
tokenizer_path = 'D:/Desktop/Diffusion/MyDiffusion/ldm/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff'

def train_prepare():
    # load model
    unet, text_encoder, vae = load_diffusion_model()
    unet.set_use_memory_efficient_attention_xformers(True)
    vae.set_use_memory_efficient_attention_xformers(True)
    print("load model finished")
    # load dataset
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
    train_dst = BaseDataset("dataset/train_debug.json", tokenizer)
    print("load dataset finished")
    # cache latents
    print("cache latents start")
    vae = vae.to("cuda", dtype=torch.float32)
    vae.requires_grad_(False)
    vae.eval()
    with torch.no_grad():
        train_dst.cache_latents(vae, 12)
    vae = vae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    del vae
    print("cache latents finished")
    return unet, text_encoder, train_dst


def train(unet, text_encoder, train_loader, args, name):
    # set hyperparameters
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.epoch
    # create network for training
    if args.pefttype == "lora":
        if args.train_unet and args.train_te:
            ft_model = LoraDiffusion(unet=unet, text_encoder=text_encoder, unet_dim=args.unet_dim,
                                     unet_alpha=args.unet_alpha,
                                     text_encoder_dim=args.te_dim, text_encoder_alpha=args.te_alpha,
                                     trainable_scale=args.trainable_scale,)
        elif args.train_unet and not args.train_te:
            ft_model = LoraDiffusion(unet=unet, text_encoder=None, unet_dim=args.unet_dim, unet_alpha=args.unet_alpha,
                                     trainable_scale=args.trainable_scale)
        elif not args.train_unet and args.train_te:
            ft_model = LoraDiffusion(unet=None, text_encoder=text_encoder, text_encoder_dim=args.te_dim,
                                     text_encoder_alpha=args.te_alpha, trainable_scale=args.trainable_scale)
        else:
            raise ValueError("train_unet and train_te cannot be both False")
    elif args.pefttype == "adapter":
        ft_model = AdapterDiffusion(unet=unet, adapter_dim=args.unet_dim)
    elif args.pefttype == "lycoris":
        ft_model = LycorisDiffusion(unet=unet, unet_dim=args.unet_dim)
    elif args.pefttype == "panolora":
        ft_model = PanoLoRA_Diffusion(unet=unet, low_factor=args.low_factor)
    elif args.pefttype == "ablation":
        ft_model = AblationDiffusion(unet=unet, low_factor=args.low_factor,mix=args.mix,ablation=args.ablation)
    print("\ncreate " + args.pefttype + " modules finished")
    # set scheduler
    steps = len(train_dst) // batch_size * num_epochs
    warmup_steps = steps // 10
    progress_bar = tqdm(range(steps), smoothing=0, desc="steps")
    noise_schedule = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                   num_train_timesteps=1000, clip_sample=False)
    loss_list = []
    loss_total = 0.0
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="fp16")
    # accelerator = Accelerator(gradient_accumulation_steps=1)
    # os dir name = name
    os.makedirs(root_path+"logs/{}".format(name),exist_ok=True)
    save_path = root_path+"models/" + args.pefttype + "/"
    os.makedirs(save_path,exist_ok=True)
    writer = SummaryWriter(root_path+"logs/{}".format(name))
    # train setting
    if not args.train_te:
        text_encoder.eval()
    else:
        text_encoder.train()
    text_encoder.requires_grad_(args.train_te)
    text_encoder = text_encoder.to(accelerator.device, dtype=torch.float32)
    if not args.train_unet:
        unet.eval()
    else:
        unet.train()
    unet.requires_grad_(args.train_unet)
    unet = unet.to(accelerator.device, dtype=torch.float16)
    if args.pefttype in ["lora", "adapter","lycoris","panolora","ablation"]:
        ft_model = ft_model.to(accelerator.device, dtype=torch.float32)
        train_params = ft_model.parameters()
    elif args.pefttype == "bitFit":
        unet = unet.to(accelerator.device, dtype=torch.float32)
        train_params = bitFit_prepare(unet)
    elif args.pefttype == "binoFit":
        unet = unet.to(accelerator.device, dtype=torch.float32)
        train_params = bino_prepare(unet)

    optimizer = torch.optim.AdamW(train_params, lr=lr)
    # warmup scheduler in the first 10% steps and then decay to 0 by cosine decay
    scheduler = warmup_cosin_schedule(optimizer, warmup_steps, steps)
    tensorboard_index = 0
    if args.pefttype != "bitFit" and args.pefttype != "binoFit":
        # additional modules for stablediffusion
        if args.train_unet and args.train_te:
            unet, text_encoder, ft_model, optimizer, train_loader, scheduler = accelerator.prepare(unet, text_encoder,
                                                                                                   ft_model, optimizer,
                                                                                                   train_loader,
                                                                                                   scheduler)
        elif args.train_unet and not args.train_te:
            unet, ft_model, optimizer, train_loader, scheduler = accelerator.prepare(unet, ft_model, optimizer,
                                                                                     train_loader, scheduler)
        else:
            text_encoder, ft_model, optimizer, train_loader, scheduler = accelerator.prepare(text_encoder, ft_model,
                                                                                             optimizer,
                                                                                             train_loader, scheduler)
    else:
        # change unet bias only
        unet, optimizer, train_loader, scheduler = accelerator.prepare(unet, optimizer, train_loader, scheduler)

    print("start training")
    for epoch in range(num_epochs):
        print("epoch", epoch + 1, "/", num_epochs)
        for step, (latents, tokens) in enumerate(train_loader):
            # clear grad
            optimizer.zero_grad()
            # get latents
            latents = latents.to("cuda", dtype=torch.float32)
            latents = latents * 0.18215
            # get text features
            with torch.set_grad_enabled(args.train_te):
                tokens = tokens.to(accelerator.device).squeeze(1)
                text_features = text_encoder(tokens)[0]
            # sample noise that will be added to latents
            noise = torch.randn_like(latents, device=accelerator.device, dtype=torch.float32)
            # sample time steps
            timesteps = torch.randint(0, noise_schedule.config.num_train_timesteps, (latents.shape[0],),
                                      device=accelerator.device)
            timesteps = timesteps.long()
            # diffusion = add noise to latents
            noisy_latents = noise_schedule.add_noise(latents, noise, timesteps)
            # predict noise
            with accelerator.autocast():
                pred_noise = unet(noisy_latents, timesteps, text_features).sample
            # calculate loss
            loss = torch.nn.functional.mse_loss(pred_noise.float(), noise.float(), reduction="none")
            loss = loss.mean()
            # backward
            # loss.backward()
            accelerator.backward(loss)
            # grad_norm
            accelerator.clip_grad_norm_(train_params, 1.0)
            optimizer.step()
            scheduler.step()
            # accumulate here finish
            # update progress bar
            progress_bar.update(1)
            current_loss = loss.detach().item()
            if epoch == 0:
                loss_list.append(current_loss)
            else:
                loss_total -= loss_list[step]
                loss_list[step] = current_loss
            loss_total += current_loss
            avr_loss = loss_total / (len(loss_list))
            logs = {"loss": avr_loss}
            progress_bar.set_postfix(**logs)
            writer.add_scalar("step_loss", current_loss, tensorboard_index)

            tensorboard_index += 1
        if args.pefttype == "bitFit":
            bitFit_save(unet, save_path + args.pefttype + "_{}_epoch_{}.ckpt".format(name, epoch + 1))
        elif args.pefttype == "binoFit":
            bino_save(unet, save_path + args.pefttype + "_{}_epoch_{}.ckpt".format(name, epoch + 1))
        else:
            ft_model.save_weights(save_path + args.pefttype + "_{}_epoch_{}.ckpt".format(name, epoch + 1))
        writer.add_scalar("epoch_loss", avr_loss, epoch + 1)
        print("")
        time.sleep(0.001)
    writer.flush()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train peft")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--epoch", type=int, default=20, help="epoch")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--train_unet", type=str2bool, default=True, help="train_unet")
    parser.add_argument("--train_te", type=str2bool, default=False, help="train_te")
    parser.add_argument("--unet_dim", type=int, default=8, help="unet_dim")
    parser.add_argument("--unet_alpha", type=int, default=1, help="unet_alpha")
    parser.add_argument("--te_dim", type=int, default=16, help="te_dim")
    parser.add_argument("--te_alpha", type=int, default=1, help="te_alpha")
    parser.add_argument("--trainable_scale", type=str2bool, default=True, help="trainable_scale")
    parser.add_argument("--pefttype", type=str, default="lora", help="pefttype")
    parser.add_argument("--mix", type=str, default="cat", help="mix")
    parser.add_argument("--low_factor", type=int, default=1, help="low_factor")
    parser.add_argument("--ablation",type=str, default="None", help="ablation")
    args = parser.parse_args()
    # use start time as name without -
    name = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")).replace("-", "")
    # save args to txt
    with open(root_path+"configs/" + name + ".txt", "w") as f:
        f.write(str(args))
    unet, text_encoder, train_dst = train_prepare()
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=0, )
    train(unet=unet, text_encoder=text_encoder, train_loader=train_loader, args=args, name=name)
