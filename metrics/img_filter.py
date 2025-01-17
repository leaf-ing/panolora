import os
from hashlib import md5
import shutil
import numpy as np
import cv2
import tqdm
import random
import PIL.Image as Image

def filter_by_histsim():
    img_paths=[]

    other_paths=[r"D:\Desktop\Dataset\360SP-data\sample2",r"D:\Desktop\Dataset\360SP-data\3602"]
    length=[]
    for path in other_paths:
        names=os.listdir(path)
        length.append(len(names))
        img_paths+=[os.path.join(path,name) for name in names]
    print(len(img_paths))
    # # use hist features to remove similar images
    hist_features=[]
    for img_path in img_paths:
        img=Image.open(img_path)
        hist_features.append(cv2.calcHist([np.array(img)],[0],None,[256],[0,256]))
    # use cosine similarity to remove similar images
    def cos_similar(x,y):
        x,y=np.array(x),np.array(y)
        return np.sum(x*y)/(np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2)))
    # remove similar images
    for i in tqdm.tqdm(range(length[0])):
        for j in range(i+1,len(img_paths)):
            if os.path.exists(img_paths[i]):
                if cos_similar(hist_features[i],hist_features[j])>0.99:
                    print(img_paths[i],img_paths[j])
                    if os.path.exists(img_paths[i]):
                        os.remove(img_paths[i])

def filter_by_bottom(img_path,bottom_path):
    names=os.listdir(img_path)
    bottom_names=os.listdir(bottom_path)
    for name in names:
        if name.replace(".png","_3.png") not in bottom_names:
            os.remove(os.path.join(img_path,name))

def rename_img(erp_path,cmp_path):
    erp_names=os.listdir(erp_path)
    erp_index=[int(name.split(".")[0]) for name in erp_names]
    erp_index.sort()
    for i in range(len(erp_index)):
        # rename cmp_name
        for j in range(6):
            old_name=os.path.join(cmp_path,str(erp_index[i])+"_"+str(j)+".png")
            new_name=os.path.join(cmp_path,str(i)+"_"+str(j)+".png")
            os.rename(old_name,new_name)
        # rename erp_name
        old_name=os.path.join(erp_path,str(erp_index[i])+".png")
        new_name=os.path.join(erp_path,str(i)+".png")
        os.rename(old_name,new_name)

if __name__=="__main__":
    filter_by_histsim()
