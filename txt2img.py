from pytorch_lightning import seed_everything
import torch
from network.adapter import AdapterDiffusion
from network.lora import LoraDiffusion
from network.panolora import PanoLoRA_Diffusion
from network.lycoris import LycorisDiffusion
from dataset.base_dataset import TestDataset
from network.ablationDiffusion import AblationDiffusion
from network.panoVae import PanoVae
import gc
from torch import autocast
from diffusers import StableDiffusionPipeline
import json
import argparse
from tqdm import tqdm
import random
import os
from train_helper import str2bool


root_path = './'
sd_path = 'D:/Desktop/Diffusion/MyDiffusion/models/stable-diffusion-v1-5'

def get_config(config_name):
    # load arg from txt
    file = open("configs/" + config_name + ".txt", "r")
    str_arg = file.read()
    file.close()
    # convert str to dict
    str_arg = str_arg.replace("Namespace(", "")
    str_arg = str_arg.replace(")", "")
    str_arg = str_arg.replace(" ", "")
    # get arg from str_arg
    arg_dict = {}
    for i in str_arg.split(","):
        arg_dict[i.split("=")[0]] = i.split("=")[1]
    # convert str to int
    for i in ["epoch", "batch_size", "unet_dim", "unet_alpha", "te_dim", "te_alpha", "low_factor"]:
        if i in arg_dict.keys():
            arg_dict[i] = int(arg_dict[i])
    # convert str to float
    for i in ["lr"]:
        if i in arg_dict.keys():
            arg_dict[i] = float(arg_dict[i])
    # convert str to bool
    for i in ["train_unet", "train_te", "trainable_scale"]:
        if i in arg_dict.keys():
            if arg_dict[i] == "True":
                arg_dict[i] = True
            else:
                arg_dict[i] = False
        else:
            arg_dict[i] = False
    # convert str to str
    for i in ["pefttype", "mix", "ablation"]:
        if i in arg_dict.keys():
            arg_dict[i] = str(arg_dict[i]).replace("'", "")
    return arg_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate")
    parser.add_argument("--config_name", type=str, default="20230926_145846", help="config txt file name")
    parser.add_argument("--epoch", type=int, default=20, help="epoch ckpt")
    parser.add_argument("--json_name", type=str, default="test_outdoor.json", help="json file name")
    parser.add_argument("--panovae", type=str2bool, default=True, help="use panovae or not")
    parser.add_argument("--seed_path", type=str, default="seed.json", help="random seed")
    args = parser.parse_args()
    # print(args)
    # get config
    config_dict = get_config(args.config_name)
    # get ckpt path
    ckpt_path = root_path+"models/" + config_dict["pefttype"] + "/" + config_dict[
        "pefttype"] + "_" + args.config_name + "_epoch_" + str(
        args.epoch) + ".ckpt"
    # output_type
    if "test_indoor" in args.json_name:
        output_dst = "indoor"
    elif "test_outdoor" in args.json_name:
        output_dst = "outdoor"
    elif "val_indoor" in args.json_name:
        output_dst = "val_indoor"
    elif "val_outdoor" in args.json_name:
        output_dst = "val_outdoor"
    else:
        output_dst = 'debug'
    # set save path according the json name
    # save_path = "output/images/" + args.config_name + "_epoch_" + str(args.epoch) + "_" + output_dst + "_vae"
    seed_path = args.seed_path
    save_path = root_path+"images/" + args.config_name + "_epoch_" + str(
        args.epoch) + "_" + output_dst + "_" + \
                seed_path.split(".")[0]
    os.makedirs(save_path, exist_ok=True)
    # get data
    json_path = "dataset/" + args.json_name
    test_dst = TestDataset(json_path)
    # val_data = json.load(open(json_path, "r", encoding="utf-8"))
    # info_list = []
    # get finished image names and replace .png to ""
    finished_names = os.listdir(save_path)
    finished_names = [i.split("_")[0] for i in finished_names]
    # load model
    pipe = StableDiffusionPipeline.from_pretrained(sd_path)
    if config_dict["pefttype"] == "lora":
        if config_dict["train_te"]:
            lora_model = LoraDiffusion(unet=pipe.unet, text_encoder=pipe.text_encoder, unet_dim=config_dict["unet_dim"],
                                       unet_alpha=config_dict["unet_alpha"], text_encoder_dim=config_dict["te_dim"],
                                       text_encoder_alpha=config_dict["te_alpha"],
                                       trainable_scale=config_dict["trainable_scale"])
        else:
            lora_model = LoraDiffusion(unet=pipe.unet, text_encoder=None, unet_dim=config_dict["unet_dim"],
                                       unet_alpha=config_dict["unet_alpha"],
                                       trainable_scale=config_dict["trainable_scale"])
        lora_model.load_state_dict(torch.load(ckpt_path))
        lora_model = lora_model.to("cuda")
        lora_model.eval()
    elif config_dict["pefttype"] == "adapter":
        adapter = AdapterDiffusion(unet=pipe.unet, adapter_dim=config_dict["unet_dim"])
        adapter.load_state_dict(torch.load(ckpt_path))
        adapter = adapter.to("cuda")
        adapter.eval()
    elif config_dict["pefttype"] == "lycoris":
        lycoris = LycorisDiffusion(unet=pipe.unet, unet_dim=config_dict["unet_dim"],
                                   unet_alpha=config_dict["unet_alpha"],
                                   trainable_scale=config_dict["trainable_scale"])
        lycoris.load_state_dict(torch.load(ckpt_path))
        lycoris = lycoris.to("cuda")
        lycoris.eval()
    elif config_dict["pefttype"] == "bitFit" or config_dict["pefttype"] == "binoFit" or config_dict[
        "pefttype"] == "finetune":
        pipe.unet.load_state_dict(torch.load(ckpt_path), strict=False)
    elif config_dict["pefttype"] == "panolora":
        panolora = PanoLoRA_Diffusion(unet=pipe.unet, low_factor=config_dict["low_factor"])
        panolora.load_state_dict(torch.load(ckpt_path), strict=False)
        panolora = panolora.to("cuda")
        panolora.eval()
    elif config_dict["pefttype"] == "ablation":
        ablation = AblationDiffusion(unet=pipe.unet, low_factor=config_dict["low_factor"],
                                     mix=config_dict["mix"], ablation=config_dict["ablation"])
        ablation.load_state_dict(torch.load(ckpt_path), strict=False)
        ablation = ablation.to("cuda")
        ablation.eval()
    # change vae
    if args.panovae:
        PanoVae(pipe.vae)
    # generate images and use tqdm to show progress
    pipe.set_progress_bar_config(disable=True)
    pipe.set_use_memory_efficient_attention_xformers(True)
    pipe.to("cuda")
    print("cache latents starts")
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    with torch.no_grad():
        test_dst.cache_latents(pipe, True)
    pipe.text_encoder.to("cpu")
    del pipe.text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    test_loader = torch.utils.data.DataLoader(test_dst, batch_size=1, shuffle=False, num_workers=0)
    progress_bar = tqdm(range(len(test_dst) - len(finished_names)), smoothing=0, desc="steps")
    with open(seed_path, "r") as f:
        seed_data = json.load(f)
    print(args.config_name + "_epoch_" + str(args.epoch) + " is generating images...")
    # use tqdm to show progress
    with autocast("cuda"):
        with torch.no_grad():
            for _, (p_latent, n_latent, index) in enumerate(test_loader):
                index = str(index.item())
                if index not in finished_names:
                    # randomize the seed
                    if index in seed_data:
                        seed = int(seed_data[index])
                    else:
                        seed = random.randint(0, 1000000)
                    seed_everything(seed)
                    torch.cuda.empty_cache()
                    image = pipe(prompt_embeds=p_latent, negative_prompt_embeds=n_latent, height=512, width=1024,
                                 num_inference_steps=40, guidance_scale=7.5).images[0]
                    # save seed in name
                    image.save(save_path + "/" + index + "_" + str(seed) + ".png")
                    # clean vram cache
                    torch.cuda.empty_cache()
                    # gc.collect()
                    progress_bar.update(1)
