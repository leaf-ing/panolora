import os
import gc
import time


# get configs.txt from output/configs
def check(path, num_img):
    # count number of images in path
    num_img_in_path = len(os.listdir(path))
    return num_img_in_path == num_img

root_path = './'
os.system("activate xf10")

configs = os.listdir(root_path+"configs")
configs.sort(reverse=True)
configs = configs[:1]
config_names = [config.replace(".txt", "") for config in configs]
config_names = config_names[::-1]
epochs = [20] * len(config_names)
seed_path = "valseed.json"
panovaes = ["True"] * len(config_names)
# json_names = ["test_outdoor.json", "test_indoor.json"]
json_names = ["val_debug.json"]
for j in range(len(json_names)):
    json_name = json_names[j]
    for i in range(len(config_names)):
        config = config_names[i]
        epoch = epochs[i]
        panovae = panovaes[i]
        os.system(
            "python D:/Desktop/panolora/txt2img.py --config_name {} --epoch {} --json_name {} --panovae {} --seed_path {}".format(
                config, 1, json_name, panovae, seed_path))
        gc.collect()
        time.sleep(5)
#os.system("python D:/Desktop/panolora/metrics/calFID.py")