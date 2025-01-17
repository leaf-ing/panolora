import torch
import os
import json
from txt2img import get_config
import re
import gc


def test_kid(config, epoch, sub, res_path="D:/Desktop/Dataset/Final_use/test/"):
    fake_path = 'D:/Desktop/Diffusion/output/images/' + config + '_epoch_' + str(epoch) + '_' + sub
    # fake_path = 'D:/Desktop/Diffusion/output/final_result/Adapter_96/Adapter_96_epoch_' + str(epoch) + '_' + sub
    real_path = 'D:/Desktop/Dataset/Final_use/test/'
    if "indoor" in sub:
        real_path += "indoor"
    else:
        real_path += "outdoor"

    os.system("activate xf10")
    cmd = "fidelity --gpu 0 --kid --input1 " + fake_path + " --input2 " + real_path
    # print(cmd)
    print(cmd)
    result = os.popen(cmd)
    content = result.readlines()
    print(content)
    kid_mean = content[0].split("kernel_inception_distance_mean: ")[-1].strip()
    kid_std = content[1].split("kernel_inception_distance_std: ")[-1].strip()
    # kid = re.findall(r"\d+\.?\d*", content)
    # kid = list(filter(lambda x: x != '0', kid))
    # if (len(kid) == 1):
    #     kid = float(kid[0])
    #
    print(sub + " kid mean of {} epoch {} is {}".format(config, epoch, kid_mean))
    # save to config.txt by a+
    with open("D:/Desktop/Diffusion/output/results/test-kid.txt", 'a+') as f:
        f.write("{}\t{}\n".format(kid_mean, kid_std))
    return kid_mean, kid_std


def test(config, epoch, real_path="D:/Desktop/Dataset/Final_use/test/"):
    with open("D:/Desktop/Diffusion/output/results/test-kid.txt", 'a+') as f:
        f.write("\n" + config + "\t\t")
    kid = []
    kid_std = []
    t = ["indoor", "outdoor"]
    s = ["seed", "seed1", "seed2"]
    for i in t:
        with open("D:/Desktop/Diffusion/output/results/test-kid.txt", 'a+') as f:
            f.write("\n" + i + "\n")
        for seed in s:
            r = test_kid(config, epoch, i + "_" + seed, real_path)
            kid.append(float(r[0]))
            kid_std.append(float(r[1]))
    avg_kid = sum(kid) / len(kid)
    print("AVG kid of {} epoch {} is {} {} {}".format(config, epoch, sum(kid[:3]) / 3, sum(kid[3:6]) / 3, avg_kid))
    with open("D:/Desktop/Diffusion/output/results/test-kid.txt", 'a+') as f:
        f.write("\navg\t\t {}\t{}\t{}\n".format(sum(kid[:3]) / 3, sum(kid[3:6]) / 3, avg_kid))

