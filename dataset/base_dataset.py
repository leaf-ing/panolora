import json
import os.path
import torch
import numpy as np
import random
from typing import List
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer


class ImageInfo():
    def __init__(self, path, caption):
        self.path = path
        self.caption = caption
        # 存放若干数据增强后图像经过vae编码后的latent
        self.latents: List[torch.Tensor] = []

class PromptInfo():
    def __init__(self, index, caption):
        self.index = index
        self.caption = caption
        # 存放prompt的编码
        self.latent: torch.Tensor

# 用于训练的基础数据集
class BaseDataset(Dataset):
    def __init__(self, json_path, tokenizer: CLIPTokenizer = None):
        self.json_path = json_path
        self.image_infos: List[ImageInfo] = []
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), ])
        self.tokenizer = tokenizer
        self.load_image_infos()

    def load_image_infos(self):
        json_data = json.load(open(self.json_path, "r", encoding="utf-8"))
        for image_info in json_data:
            self.image_infos.append(ImageInfo(image_info["image"], image_info["caption"]))

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, index):
        # has finished cache_latents by default
        image_info = self.image_infos[index]
        # use random choice to choose a latent
        latent = image_info.latents[random.randint(0, len(image_info.latents) - 1)]
        # cfg
        caption = image_info.caption if random.random() > 0.1 else ""
        input_ids = self.tokenizer(caption, padding="max_length", truncation=False, return_tensors="pt",
                                   max_length=77).input_ids
        # get caption token
        return latent, input_ids

    def cache_latents(self, vae, trans_num=12, is_cache_to_disk=True):
        # cache VAE latents in memory
        for image_info in tqdm(self.image_infos):
            npz_path = image_info.path.replace(".png", ".npz")
            # if cache to disk and npz file exists, load from disk
            if os.path.exists(npz_path):
                # load from disk
                npz_data = np.load(npz_path)
                latents = npz_data["latents"]
                for latent in latents:
                    image_info.latents.append(torch.tensor(latent, dtype=vae.dtype))
                continue
            image = Image.open(image_info.path).convert("RGB")
            image = np.array(image)
            height, width = image.shape[:2]
            # here use translation to augment data
            translates = [int(width / trans_num * i) for i in range(trans_num)]
            for translate in translates:
                new_image = np.concatenate([image[:, translate:], image[:, :translate]], axis=1)
                img_tensor = self.transform(new_image)
                img_tensor = img_tensor.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
                # cache latents
                latent = vae.encode(img_tensor).latent_dist.sample().squeeze().to("cpu")
                if torch.isnan(latent).any():
                    raise RuntimeError(f"latent has nan value, image path: {image_info.path}")
                image_info.latents.append(latent)
            if is_cache_to_disk:
                # save to disk
                latents = torch.stack(image_info.latents, dim=0).float().cpu().numpy()
                np.savez(npz_path, latents=latents)

# 用于test和val的数据集
class TestDataset(Dataset):
    def __init__(self, json_path):
        self.json_path = json_path
        self.prompt_info: List[PromptInfo] = []
        self.neg_prompt_embed = None
        self.load_prompt_info()

    def load_prompt_info(self):
        json_data = json.load(open(self.json_path, "r", encoding="utf-8"))
        for p_info in json_data:
            index = p_info["image"].split("/")[-1].split(".")[0]
            self.prompt_info.append(PromptInfo(index, p_info["caption"]))

    def __len__(self):
        return len(self.prompt_info)

    def __getitem__(self, index):
        # has finished cache_latents by default
        p_info = self.prompt_info[index]
        # get latent
        latent = p_info.latent
        return latent.squeeze(),self.neg_prompt_embed.squeeze(),int(p_info.index)

    def cache_latents(self, pipe, is_cache_to_disk=True):
        if "test" in self.json_path:
            cache_path = "F:/Dataset/Final_use/test_cache"
        elif "val" in self.json_path:
            cache_path = "F:/Dataset/Final_use/val_cache"
        if self.neg_prompt_embed is None:
            self.neg_prompt_embed = np.load(os.path.join(cache_path, "neg_prompt_embed.npz"))["latent"]
            self.neg_prompt_embed = torch.tensor(self.neg_prompt_embed, dtype=pipe.text_encoder.dtype)
        for p_info in tqdm(self.prompt_info):
            npz_path = os.path.join(cache_path, p_info.index + ".npz")
            # if cache to disk and npz file exists, load from disk
            if os.path.exists(npz_path):
                # load from disk
                npz_data = np.load(npz_path)
                latent = npz_data["latent"]
                p_info.latent = torch.tensor(latent, dtype=pipe.text_encoder.dtype)
                continue
            prompt = p_info.caption
            # encode prompt
            prompt_embeds = pipe._encode_prompt(
                prompt,
                pipe.device,
                1,
                True,
                None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                lora_scale=None,
            )
            prompt_emb = prompt_embeds[1].unsqueeze(0).to("cpu")
            p_info.latent = prompt_emb
            # cache to disk
            if is_cache_to_disk:
                np.savez(npz_path, latent=p_info.latent.float().numpy())


