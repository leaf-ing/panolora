import torch
import os
import json
from txt2img import get_config
import re
import gc


def calFID(config, epoch, sub=""):
    fake_path = 'D:/Desktop/Diffusion/output/images/' + config + '_epoch_' + str(epoch) + '_' + sub
    real_path = 'D:/Desktop/Dataset/Final_use/test/' + sub

    os.system("activate xf10")
    result = os.popen("python -m pytorch_fid " + fake_path + " " + real_path)
    content = result.readlines()[0]
    fid = re.findall(r"\d+\.?\d*", content)
    fid = list(filter(lambda x: x != '0', fid))
    if (len(fid) == 1):
        fid = float(fid[0])

    print(sub + " FID of {} epoch {} is {}".format(config, epoch, fid))
    # save to config.txt by a+
    with open("D:/Desktop/Diffusion/output/results/result.txt", 'a+') as f:
        f.write("\n" + sub + " FID of {} epoch {} is {}".format(config, epoch, fid))


def val_fid(config, epoch, sub, real_path="D:/Desktop/Dataset/Final_use/val/"):
    fake_path = 'D:/Desktop/Diffusion/output/images/' + config + '_epoch_' + str(epoch) + '_' + sub
    if "indoor" in sub:
        sub = "indoor"
    else:
        sub = "outdoor"
    real_path = real_path + sub

    os.system("activate xf10")
    result = os.popen("python -m pytorch_fid " + fake_path + " " + real_path)
    content = result.readlines()[0]
    fid = re.findall(r"\d+\.?\d*", content)
    fid = list(filter(lambda x: x != '0', fid))
    if (len(fid) == 1):
        fid = float(fid[0])

    print(sub + " FID of {} epoch {} is {}".format(config, epoch, fid))
    # save to config.txt by a+
    with open("D:/Desktop/Diffusion/output/results/val.txt", 'a+') as f:
        f.write(sub + " \t{}\t".format(fid))
    return fid


def test_fid(config, epoch, sub, res_path="D:/Desktop/Dataset/Final_use/test/"):
    fake_path = 'D:/Desktop/Diffusion/output/images/' + config + '_epoch_' + str(epoch) + '_' + sub
    #fake_path = 'D:/Desktop/Diffusion/output/final_result/Adapter_96/Adapter_96_epoch_' + str(epoch) + '_' + sub
    real_path = 'D:/Desktop/Dataset/Final_use/test/'
    if "indoor" in sub:
        real_path += "indoor"
    else:
        real_path += "outdoor"

    os.system("activate xf10")
    result = os.popen("python -m pytorch_fid " + fake_path + " " + real_path)
    content = result.readlines()[0]
    fid = re.findall(r"\d+\.?\d*", content)
    fid = list(filter(lambda x: x != '0', fid))
    if (len(fid) == 1):
        fid = float(fid[0])

    print(sub + " FID of {} epoch {} is {}".format(config, epoch, fid))
    # save to config.txt by a+
    with open("D:/Desktop/Diffusion/output/results/test.txt", 'a+') as f:
        f.write("\t{}".format(fid))
    return fid


def val(config, epoch, part="both", real_path="D:/Desktop/Dataset/Final_use/val/"):
    with open("D:/Desktop/Diffusion/output/results/val.txt", 'a+') as f:
        f.write("\n" + config + "\t\t")
    subs = ["val_outdoor_valseed", "val_indoor_valseed"]
    if part == "outdoor" or part == "both":
        out_fid = val_fid(config, epoch, subs[0], real_path)
    if part == "indoor" or part == "both":
        in_fid = val_fid(config, epoch, subs[1], real_path)
    if part == "both":
        avg_fid = (out_fid + in_fid) / 2
        print("AVG FID of {} epoch {} is {}".format(config, epoch, avg_fid))
        # save to config.txt by a+
        with open("D:/Desktop/Diffusion/output/results/val.txt", 'a+') as f:
            f.write("\tavg\t\t {}\n".format(avg_fid))


def test(config, epoch, real_path="D:/Desktop/Dataset/Final_use/test/"):
    with open("D:/Desktop/Diffusion/output/results/test.txt", 'a+') as f:
        f.write("\n" + config + "\t\t")
    fid = []
    t = ["indoor","outdoor"]
    s = ["seed", "seed1", "seed2"]
    for i in t:
        with open("D:/Desktop/Diffusion/output/results/test.txt", 'a+') as f:
            f.write("\n" + i)
        for seed in s:
            fid.append(test_fid(config, epoch, i + "_" + seed, real_path))
    avg_fid = sum(fid) / len(fid)
    print("AVG FID of {} epoch {} is {} {} {}".format(config, epoch, sum(fid[:3]) / 3, sum(fid[3:6]) / 3, avg_fid))
    with open("D:/Desktop/Diffusion/output/results/test.txt", 'a+') as f:
        f.write("\navg\t\t {}\t{}\t{}\n".format(sum(fid[:3]) / 3, sum(fid[3:6]) / 3, avg_fid))

