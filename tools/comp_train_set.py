import random
import os

Hatt_fg_path = 'D:/Dataset/Matting/Distinctions-646/Distinctions-646/Train/FG/'
Adobe_fg_path = 'D:/Dataset/Matting/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/all/fg/'
AM_fg_path = 'D:/Dataset/Matting/AM-2k/train/original/'

Hatt_a_path = 'D:/Dataset/Matting/Distinctions-646/Distinctions-646/Train/GT/'
Adobe_a_path = 'D:/Dataset/Matting/Adobe_Deep_Matting_Dataset/Combined_Dataset/Training_set/all/alpha/'
AM_a_path = 'D:/Dataset/Matting/AM-2k/train/mask/'

bg_path = 'D:/Dataset/Matting/BG-20k/train/'

Hatt_bg_nums = 5
Adobe_bg_nums = 5
AM_bg_nums = 2

Hatt_fg_files = [line.rstrip('\n') for line in open('D:/Dataset/Matting/Distinctions-646/Distinctions-646/Train/fg_train.txt')]
Adobe_fg_files = os.listdir(Adobe_fg_path)
AM_fg_files = os.listdir(AM_fg_path)

bg_files = [line.rstrip('\n') for line in open('D:/Dataset/Matting/BG-20k/train.txt')]


out_img_path = 'D:/Dataset/Matting/AIM-Train/image/'
out_gt_path  = 'D:/Dataset/Matting/AIM-Train/alpha/'
out_fg_path  = 'D:/Dataset/Matting/AIM-Train/fg/'
out_bg_path  = 'D:/Dataset/Matting/AIM-Train/bg/'


from PIL import Image
import os
import numpy as np
import math
import time
import cv2
import torch

def composite(fg, bg, a, w, h):
    bg = bg[0:w, 0:h]

    # blur
    kszies = [20, 30, 40, 50, 60]
    kszie = kszies[random.randint(0, 4)]
    bg = cv2.blur(bg, ksize=(kszie, kszie))

    bg = torch.from_numpy(bg).transpose(0, 2).double()
    fg = torch.from_numpy(fg).transpose(0, 2).double()
    alpha = torch.from_numpy(a).transpose(0, 1).double() /255
    composite_img = alpha * fg + (1 - alpha) * bg
    composite_img = composite_img.int()
    composite_img = composite_img.transpose(0, 2).numpy()

    return composite_img

os.makedirs(out_bg_path, exist_ok=True)
os.makedirs(out_fg_path, exist_ok=True)
os.makedirs(out_gt_path, exist_ok=True)
os.makedirs(out_img_path, exist_ok=True)


bg_iter = iter(bg_files)
index = 5135

def c(fg_files, fg_path, a_path, num_bgs):
    global index
    global AM_a_path
    for im_name in fg_files:
        im = cv2.imread(os.path.join(fg_path, im_name))
        if a_path == AM_a_path:
            a = cv2.imread(os.path.join(a_path, im_name.replace('jpg', 'png')), cv2.IMREAD_GRAYSCALE)
        else:
            a = cv2.imread(os.path.join(a_path, im_name), cv2.IMREAD_GRAYSCALE)

        bbox = im.shape
        w = bbox[0]
        h = bbox[1]

        bcount = 0
        for i in range(num_bgs):

            bg_name = next(bg_iter)
            bg = cv2.imread(os.path.join(bg_path, bg_name))
            bg_bbox = bg.shape
            bw = bg_bbox[0]
            bh = bg_bbox[1]
            wratio = w / bw
            hratio = h / bh
            ratio = wratio if wratio > hratio else hratio
            if ratio > 1:
                # cv2--->PIL--->cv2 for keep the same. Since the resize of PIL and the resize of cv2 is different
                bg = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
                bg = bg.resize((math.ceil(bh * ratio), math.ceil(bw * ratio)), Image.BICUBIC)
                bg = cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR)

            out = composite(im, bg, a, w, h)
            cv2.imwrite(out_img_path + im_name.split('_')[0] + '_' + str(index) + '.png', out)  # +'train_img_'
            gt = a
            cv2.imwrite(out_gt_path + im_name.split('_')[0] + '_' + str(index) + '.png', gt)  # +'train_gt_'

            bf_for_save = bg[0:w, 0:h]
            cv2.imwrite(out_bg_path + im_name.split('_')[0] + '_' + str(index) + '.png', bf_for_save)
            cv2.imwrite(out_fg_path + im_name.split('_')[0] + '_' + str(index) + '.png', im)

            print(out_gt_path + im_name.split('_')[0] + '_' + str(index) + '.png' + '-----%d' % index)
            index += 1

# c(fg_files=Adobe_fg_files, fg_path=Adobe_fg_path, a_path=Adobe_a_path, num_bgs=Adobe_bg_nums)
# c(fg_files=Hatt_fg_files, fg_path=Hatt_fg_path, a_path=Hatt_a_path, num_bgs=Hatt_bg_nums)
c(fg_files=AM_fg_files, fg_path=AM_fg_path, a_path=AM_a_path, num_bgs=AM_bg_nums)


