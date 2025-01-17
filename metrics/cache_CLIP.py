from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np
import json
from transformers import CLIPProcessor, CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
import gc


def cache_text_feature():
    # 2是上，3是下，0,1,4,5是四周。caption也是如此
    with open("../dataset/test_indoor.json", 'r') as f:
        content = json.load(f)
    with open("../dataset/test_outdoor.json", 'r') as f:
        content += json.load(f)

    device = torch.device("cuda")
    model = CLIPTextModelWithProjection.from_pretrained("D:/Desktop/Diffusion/MyDiffusion/openai/clip-vit-large-patch14").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("D:/Desktop/Diffusion/MyDiffusion/openai/clip-vit-large-patch14")

    # inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
    # text_features = model.get_text_features(**inputs)
    for i in tqdm(range(len(content))):
        curline = content[i]
        index = curline["image"].split("/")[-1].split(".")[0]
        caption = curline["caption"]
        npz_path = "D:/Desktop/Dataset/Final_use/clip_score_cache/" + index + ".npz"
        # change caption to 6 sentences
        caption = caption.split(".")[:-1]
        for j in range(len(caption)):
            while caption[j][0] == " ":
                caption[j] = caption[j][1:]
            caption[j] = caption[j] + "."
        # get text_features
        inputs = tokenizer(caption, padding=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        text_embeds = outputs.text_embeds
        text_features = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features.float().detach().cpu().numpy()
        if i == 0:
            print(text_features.shape, text_features)
        # text_features = model.get_text_features(**inputs)
        # text_features = text_features.float().detach().cpu().numpy()
        np.savez(npz_path, text_features=text_features)

def cache_img_feature(model_names):
    batch_size = 100
    device = torch.device("cuda")
    model = CLIPVisionModelWithProjection.from_pretrained("D:/Desktop/Diffusion/MyDiffusion/openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("D:/Desktop/Diffusion/MyDiffusion/openai/clip-vit-large-patch14")
    #model_names = [ "Lora_16", "BitFit","BiasNorm", "Lycoris_2", "Lycoris_4", "Adapter_48", "Adapter_96", "Lora_8"]
    path_prefix = "D:/Desktop/Diffusion/output/CMPs/test/"
    seed = ["seed", "seed1", "seed2"]
    inout = ["indoor", "outdoor"]
    progress_bar = tqdm(range(3000 * 36 * len(model_names) // batch_size), smoothing=0, desc="steps")
    # 创建cache目录
    for model_name in model_names:
        cache_path = path_prefix + model_name + "_image_feature_cache/"
        try:
            os.mkdir(cache_path)
        except:
            pass
        with torch.no_grad():
            for i in range(len(inout)):
                for j in range(len(seed)):
                    folder_name = path_prefix + model_name + "_" + inout[i] + "_" + seed[j] + "/"
                    png_names = sorted(os.listdir(folder_name))
                    for k in range(0, len(png_names), batch_size):
                        cache_names = [png_names[k + m].split(".")[0] for m in range(batch_size)]
                        try:
                            images = [np.array(Image.open(os.path.join(folder_name, png_names[k + m]))) for m in range(batch_size)]
                            inputs = processor(images=images, return_tensors="pt").to(device)
                            outputs = model(**inputs)
                            image_embeds = outputs.image_embeds
                            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                            # image_features = image_embeds.float().detach().cpu().numpy()
                            image_features = image_embeds.float().cpu().numpy()
                            if i == 0 and j == 0 and k == 0:
                                print(image_features.shape, image_features)
                            # print(image_features.shape)
                            for m in range(batch_size):
                                npz_path = cache_path + cache_names[m] + ".npz"
                                np.savez(npz_path, image_features=image_features[m])
                        except:
                            print(cache_names)
                        progress_bar.update(1)
                    gc.collect()
                    torch.cuda.empty_cache()

def cache_val_img_feature(model_names):
    batch_size = 100
    device = torch.device("cuda")
    model = CLIPVisionModelWithProjection.from_pretrained("D:/Desktop/Diffusion/MyDiffusion/openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("D:/Desktop/Diffusion/MyDiffusion/openai/clip-vit-large-patch14")
    path_prefix = "D:/Desktop/Diffusion/output/CMPs/val/"
    inout = ["indoor", "outdoor"]
    progress_bar = tqdm(range(1000 * 6 * 2 * len(model_names) // batch_size), smoothing=0, desc="steps")
    # 创建cache目录
    for model_name in model_names:
        cache_path = path_prefix + model_name + "_image_feature_cache/"
        try:
            os.mkdir(cache_path)
        except:
            pass
        with torch.no_grad():
            for i in range(len(inout)):
                folder_name = path_prefix + model_name + "_" + inout[i] + "/"
                png_names = sorted(os.listdir(folder_name))
                for k in range(0, len(png_names), batch_size):
                    # images=np.array(images)
                    cache_names = [png_names[k + m].split(".")[0] for m in range(batch_size)]
                    try:
                        images = [np.array(Image.open(os.path.join(folder_name, png_names[k + m]))) for m in
                                  range(batch_size)]
                        inputs = processor(images=images, return_tensors="pt").to(device)
                        outputs = model(**inputs)
                        image_embeds = outputs.image_embeds
                        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                        # image_features = image_embeds.float().detach().cpu().numpy()
                        image_features = image_embeds.float().cpu().numpy()
                        if i == 0 and k == 0:
                            print(image_features.shape, image_features)
                        # print(image_features.shape)
                        for m in range(batch_size):
                            npz_path = cache_path + cache_names[m] + ".npz"
                            np.savez(npz_path, image_features=image_features[m])
                    except:
                        print(cache_names)
                    progress_bar.update(1)
                gc.collect()
                torch.cuda.empty_cache()

def cache_val_text_feature():
    # 2是上，3是下，0,1,4,5是四周。caption也是如此
    with open("../dataset/val_indoor.json", 'r') as f:
        content = json.load(f)
    with open("../dataset/val_outdoor.json", 'r') as f:
        content += json.load(f)

    device = torch.device("cuda")
    model = CLIPTextModelWithProjection.from_pretrained("D:/Desktop/Diffusion/MyDiffusion/openai/clip-vit-large-patch14").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("D:/Desktop/Diffusion/MyDiffusion/openai/clip-vit-large-patch14")

    # inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
    # text_features = model.get_text_features(**inputs)
    for i in tqdm(range(len(content))):
        curline = content[i]
        index = curline["image"].split("/")[-1].split(".")[0]
        caption = curline["caption"]
        npz_path = "D:/Desktop/Dataset/Final_use/clip_score_cache/" + index + ".npz"
        # change caption to 6 sentences
        caption = caption.split(".")[:-1]
        for j in range(len(caption)):
            while caption[j][0] == " ":
                caption[j] = caption[j][1:]
            caption[j] = caption[j] + "."
        # get text_features
        inputs = tokenizer(caption, padding=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        text_embeds = outputs.text_embeds
        text_features = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features.float().detach().cpu().numpy()
        if i == 0:
            print(text_features.shape, text_features)
        # text_features = model.get_text_features(**inputs)
        # text_features = text_features.float().detach().cpu().numpy()
        np.savez(npz_path, text_features=text_features)
