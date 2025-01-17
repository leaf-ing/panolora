from tqdm import tqdm
import os
import numpy as np
import json
from itertools import permutations


def cal_clip_score(model_name, img_type="test"):
    # 2是上，3是下，0,1,4,5是四周。caption也是如此
    text_embeds_path = "D:/Desktop/Dataset/Final_use/clip_score_cache/"
    txt_npz = os.listdir(text_embeds_path)
    text_embed = {}
    print("start cache text embeds")
    for i in tqdm(range(len(txt_npz))):
        text_index = txt_npz[i].replace(".npz", "")
        npz_data = np.load(os.path.join(text_embeds_path, txt_npz[i]))
        text_latents = npz_data["text_features"]
        if i == 0:
            print("\ntext_latents:", type(text_latents), text_latents.shape)
        text_embed[text_index] = text_latents
    # 图像Embedding的位置
    img_embeds_path = "D:/Desktop/Diffusion/output/CMPs/" + img_type + "/" + model_name + "/" + model_name + "_image_feature_cache/"
    img_npz = os.listdir(img_embeds_path)
    img_embed = {}
    # 输出的json位置
    output_path = "D:/Desktop/Diffusion/output/CMPs/" + img_type + "/" + model_name + "/clip_score.json"
    # 缓存图像Embedding
    print("\nstart cache image embeddings of", model_name)
    for i in tqdm(range(len(img_npz))):
        image_index = img_npz[i].replace(".npz", "")
        npz_data = np.load(os.path.join(img_embeds_path, img_npz[i]))
        image_lantents = npz_data["image_features"]
        if i == 0:
            print("\nimage_lantents:", type(image_lantents), image_lantents.shape)
        img_embed[image_index] = image_lantents
    # 最终结果
    result = {}
    # 计算clip socre
    keys = list(img_embed.keys())
    print("\nstart calculate clip score of", model_name)
    for j in tqdm(range(len(keys))):
        image_name = keys[j]
        idx, seed, pos = image_name.split("_")
        rid = idx + "_" + seed
        pos = int(pos)
        vis = img_embed[image_name]
        # 2是上，3是下，0,1,4,5是四周。caption也是如此
        # 上下只需要计算一次clip_score
        if pos == 2 or pos == 3:
            txt = text_embed[idx][pos]
            # clip score计算
            clip_score = float((txt @ vis.T))  # (768) @ (768,1)
            clip_score = 100 * max(0, clip_score)
            if rid not in result:
                result[rid] = {}
            result[rid][pos] = clip_score
        # 前后左右，需要计算4次
        else:
            txt = text_embed[idx][[0, 1, 4, 5]]  # shape (4,768)
            # clip score
            clip_score = txt @ vis.T  # (4,768) @ (768,1) => (4,1)
            if rid not in result:
                result[rid] = {}
            cs4 = []
            for i in range(4):
                cs4.append(100 * max(0, float(clip_score[i])))
            result[rid][pos] = cs4
    # 保存结果
    with open(output_path, 'w') as f:
        f.write(json.dumps(result))


# 上下直接使用Clip-score
# 前后左右选择clip-score之和最大的排列
def cal_pano_clip_score(model_name, img_type="test"):
    input_path = "D:/Desktop/Diffusion/output/CMPs/" + img_type + "/" + model_name + "/clip_score.json"
    with open(input_path, 'r') as f:
        content = json.load(f)
    output_path = "D:/Desktop/Diffusion/output/CMPs/" + img_type + "/" + model_name + "/pano_clip_score.json"
    result = {}
    index_permute = list(permutations([0, 1, 2, 3], 4))
    # 上下
    for k in content:
        result[k] = {}
        result[k]["2"] = content[k]["2"]
        result[k]["3"] = content[k]["3"]
    # 前后左右
    for k in content:
        match_score = 0
        match_index = 0
        for i in range(len(index_permute)):
            cur_score = content[k]["0"][index_permute[i][0]] + content[k]["1"][index_permute[i][1]] + \
                        content[k]["4"][index_permute[i][2]] + content[k]["5"][index_permute[i][3]]
            if cur_score > match_score:
                match_score = cur_score
                match_index = i
        # 得到最佳组合index_permute[match_index]
        result[k]["0"] = content[k]["0"][index_permute[match_index][0]]
        result[k]["1"] = content[k]["1"][index_permute[match_index][1]]
        result[k]["4"] = content[k]["4"][index_permute[match_index][2]]
        result[k]["5"] = content[k]["5"][index_permute[match_index][3]]
        # 得到均值
        mean_clip_score = 0
        for c in "012345":
            mean_clip_score += result[k][c]
        result[k]["mean"] = mean_clip_score / 6
        # 记录组合
        result[k]['match_permute'] = index_permute[match_index]

    # 保存
    with open(output_path, 'w') as f:
        f.write(json.dumps(result))
    # 打印部分信息
    indoor_cs = 0
    outdoor_cs = 0
    indoor_count = 0
    outdoor_count = 0
    for k in result:
        index = int(k.split("_")[0])
        if index < 7000:
            indoor_cs += result[k]["mean"]
            indoor_count += 1
        else:
            outdoor_cs += result[k]["mean"]
            outdoor_count += 1
    print(model_name, "indoor:", indoor_cs / indoor_count, "outdoor:", outdoor_cs / outdoor_count)
    if img_type == "test":
        with open("D:/Desktop/Diffusion/output/results/clip_score_test.txt", 'a+') as f:
            f.write(model_name + '\n')
            f.write("indoor:" + str(indoor_cs / indoor_count) + "\toutdoor:" + str(outdoor_cs / outdoor_count) + '\n\n')
    else:
        with open("D:/Desktop/Diffusion/output/results/clip_score_val.txt", 'a+') as f:
            f.write(model_name + '\n')
            f.write("indoor:" + str(indoor_cs / indoor_count) + "\toutdoor:" + str(outdoor_cs / outdoor_count) + '\n\n')


# model_names = ["wo_cat","wo_copy","wo_conv","wo_lora","wo_panoconv"]
# for i in range(len(model_names)):
#     cal_clip_score(model_names[i], "test")
#     cal_pano_clip_score(model_names[i], "test")
