import os

epoch = 1
batch_size = 2
unet_dims = 8
lrs = [1e-3, 1e-4, 1e-5]
dims = [8]
ablations = ["wo_conv", "wo_lora", "wo_panoconv", "wo_cat", "wo_copy"]
low_factors = [15, 46, 46, 46, 64]
os.system("activate xf10")
os.system("python D:/Desktop/panolora/train_peft.py --pefttype panolora  --lr {} --epoch {} --batch_size {} --low_factor {}".format(0.00001, epoch, batch_size, 64))
#os.system("python D:/Desktop/panolora/call_txt2img.py")
