import os
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import json
import bezier
import nibabel as nib
from skimage.measure import label

def find_max_area(a):
    label_img, num = label(a, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        tmp = np.sum(label_img == i)
        if tmp > max_num:
            max_num = tmp
            max_label = i
    mask = (label_img == max_label)
    return mask

def read_img(f_path):
    im_list = []
    f_list = os.listdir(f_path)
    f_list.sort()
    # f_list = f_list[10:50]
    for i in f_list:
        img = pydicom.read_file(os.path.join(f_path, i))
        img_np = img.pixel_array
        im_list.append(img_np)   
    return im_list

def get_bg_mask(im_list):

    a = im_list[50].astype(np.int32)
    thres = np.max(a)/2.5
    a[a<thres] = 0
    a[a>0] = 255
    mask = find_max_area(a)
    return mask

def preprocess(a):
    a = (a-np.min(a))/(np.max(a)-np.min(a))
    a = a * 255
    a = a.astype(np.uint8)
    a = cv2.adaptiveThreshold(a,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,2)
    return a

def cal_norm(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def process_one(a, mask, pre_p=None, count=0):
    a = a * mask 
    a = preprocess(a)

    a = (255-a) * mask
    a_area = find_max_area(a)
    pxy = np.where(a_area>0)
    cy = np.mean(pxy[0])
    cx = np.mean(pxy[1])
    if pre_p and cal_norm(pre_p, [cx ,cy]) > 200:
        cx, cy = pre_p
        # if count < 2:
        #     count += 1
        #     cx, cy = pre_p
        # else:
        #     count = 0
    return cx, cy, count

def dect_mkr(im_list, mask):

    a = im_list[1].astype(np.int32)
    cx, cy, count = process_one(a, mask)
    pre_p = [cx, cy]
    print (pre_p)
# 
    for i in range(2, len(im_list)):
        a = im_list[i].astype(np.int32)
        cx, cy, count = process_one(a, mask, pre_p, count)
        pre_p = [cx, cy]
        # continue
        plt.figure()
        plt.imshow(im_list[i])
        # plt.plot(cnt_mean[0], cnt_mean[1], 'ro', markersize=3)
        plt.plot(cx, cy, 'ro', markersize=3)
        # plt.savefig(f'{save_path}/{i}.png')    
        plt.show()
        raise
        plt.close() 


if __name__ == '__main__':
    f_path = 'data_0108/Dynamic 2D_0108/2DMoving_1'
    save_path = '0108_2dmoving_bgsub'
    os.makedirs(save_path, exist_ok=True) 
    im_list = read_img(f_path)
    bg_mask = get_bg_mask(im_list)
    dect_mkr(im_list, bg_mask)






