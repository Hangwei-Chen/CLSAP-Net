import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
from torchvision import transforms
import cv2
import random
import matplotlib.pyplot as plt

class Folder(data.Dataset):

    def __init__(self, SR_root, CP_root, OV_root, index, transform, patch_num,patch_size,flip):

        SR_labels = []
        CP_labels = []
        OV_labels = []
        SR_label_txt = open('./dataset/SR_label.txt', 'r')
        CP_label_txt = open('./dataset/CP_label.txt', 'r')
        OV_label_txt = open('./dataset/OV_label.txt', 'r')

        for line in SR_label_txt:
            line = line.split('\n')
            words = line[0].split()
            SR_labels.append((words[0]))
        SR_label = np.array(SR_labels).astype(np.float32)

        for line in CP_label_txt:
            line = line.split('\n')
            words = line[0].split()
            CP_labels.append((words[0]))
        CP_label = np.array(CP_labels).astype(np.float32)

        for line in OV_label_txt:
            line = line.split('\n')
            words = line[0].split()
            OV_labels.append((words[0]))
        OV_label = np.array(OV_labels).astype(np.float32)

        SR_samples = []
        CP_samples = []
        OV_samples = []

        for i, item in enumerate(index):
            for aug in range(patch_num):
                SR_samples.append((os.path.join(SR_root, str(item) + ".jpg"), SR_label[item - 1]))

        for i, item in enumerate(index):
            for aug in range(patch_num):
                CP_samples.append((os.path.join(CP_root, str(item) + ".jpg"), CP_label[item - 1]))

        for i, item in enumerate(index):
            for aug in range(patch_num):
                OV_samples.append((os.path.join(OV_root, str(item) + ".jpg"), OV_label[item - 1]))


        self.SR_samples = SR_samples
        self.CP_samples = CP_samples
        self.OV_samples = OV_samples
        self.patch_size= patch_size
        self.transform = transform
        self.flip = flip


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        all_sample=[]

        SR_path, SR_target = self.SR_samples[index]
        SR_sample = pil_loader(SR_path)

        CP_path, CP_target = self.CP_samples[index]
        CP_sample = pil_loader(CP_path)

        OV_path, OV_target = self.OV_samples[index]
        OV_sample = pil_loader(OV_path)


        SR_sample, CP_sample, OV_sample = cv_random_flip(SR_sample, CP_sample, OV_sample, self.flip)
        SR_sample, CP_sample, OV_sample = randomCrop(SR_sample, CP_sample, OV_sample)

        SR_sample= self.transform(SR_sample)
        CP_sample= self.transform(CP_sample)
        OV_sample= self.transform(OV_sample)


        return SR_sample, CP_sample, OV_sample, SR_target, CP_target, OV_target

    def __len__(self):
        length = len(self.SR_samples)
        return length



#
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def cv_random_flip(SR_sample, CP_sample, OV_sample, flip):
    if flip==1:
        flip_flag = random.randint(0, 1)
        if flip_flag == 1:
            SR_sample = SR_sample.transpose(Image.FLIP_LEFT_RIGHT)
            CP_sample = CP_sample.transpose(Image.FLIP_LEFT_RIGHT)
            OV_sample = OV_sample.transpose(Image.FLIP_LEFT_RIGHT)

    else:
        SR_sample = SR_sample
        CP_sample = CP_sample
        OV_sample = OV_sample



    return SR_sample, CP_sample, OV_sample

def randomCrop(image, label, depth):
    patch_size = 224
    image_width = image.size[0]
    image_height = image.size[1]

    rnd_h = random.randint(0, max(0, image_height - patch_size))
    rnd_w = random.randint(0, max(0, image_width - patch_size))

    random_region = (rnd_w, rnd_h, (rnd_w + patch_size), (rnd_h + patch_size) )
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region)