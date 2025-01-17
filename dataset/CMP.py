import ffmpy
import os
from PIL import Image
import argparse


# ERP to CMP
def ERP2CMP(ERP_path, CMP_path):
    ff = ffmpy.FFmpeg(
        inputs={ERP_path: None},
        outputs={CMP_path: '-vf v360=e:c6x1'}
    )
    # print(ff.cmd)
    ff.run()


def process(input_folder, output_folder):
    names = os.listdir(input_folder)
    names = [x for x in names if "png" in x]
    print(names)
    for name in names:
        if os.path.exists(os.path.join(output_folder, name)):
            continue
        ERP2CMP(os.path.join(input_folder, name), os.path.join(output_folder, name))


# split it to 6 images
def splitCMP(output_folder):
    names = os.listdir(output_folder)
    names = [x for x in names if ".png" in x]
    for name in names:
        img = Image.open(os.path.join(output_folder, name))
        w, h = img.size
        for i in range(6):
            img.crop((i * w / 6, 0, (i + 1) * w / 6, h)).save(
                os.path.join(output_folder, name[:-4] + '_' + str(i) + '.png'))
        os.remove(os.path.join(output_folder, name))


def outputs2CMP(input_folder, output_folder):
    try:
        os.mkdir(output_folder)
    except:
        print("dir already exists")
    process(input_folder, output_folder)
    splitCMP(output_folder)


if __name__ == '__main__':
    pass
