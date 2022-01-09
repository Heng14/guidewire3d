import os
import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import json
import bezier
import nibabel as nib

def get3dp(f_path):
    with open(f_path,'r') as load_f:
        load_dict_3d = json.load(load_f)
    p3d_list = [i['position'] for i in load_dict_3d['markups'][0]['controlPoints']]
    return p3d_list

def get2dp(path_2dmrk_nii):
    markers = nib.load(path_2dmrk_nii)
    data = markers.get_fdata()# (512, 512)
    scale = 240 / data.shape[0]
    p_np = np.array(np.where(data>0)).astype(np.double)
    p_np *= scale
    # print (p_np)
    # plt.imshow(data)
    # plt.show()
    # raise
    return p_np

def gen_curve(p_np):
    nodes = np.asfortranarray(p_np)
    curve = bezier.Curve.from_nodes(nodes)
    return curve

def get_t(query, curve2d):
    q_np = np.array(query)
    q_np = q_np[...,np.newaxis]
    q_np = np.asfortranarray(q_np)
    t = curve2d.locate(q_np)
    if not t:
        t_np = np.linspace(-1, 2, 1000)
        p_t = [curve2d.evaluate(i) for i in t_np]
        d = [np.linalg.norm(q_np-i) for i in p_t]
        idx = np.argmin(d)
        t = t_np[idx]
    return t

if __name__ == '__main__':
    path_3dmrk_json = 'data_0108/Static 3D_0108/F_1.mrk.json'
    path_2dmrk_nii = 'data_0108/Static 2D_0108/newmarkers_2d.nii.gz'
    p3d_list = get3dp(path_3dmrk_json)
    p2d_np= get2dp(path_2dmrk_nii)
    p3d_np = np.array(p3d_list).T
    curve3d = gen_curve(p3d_np)
    curve2d = gen_curve(p2d_np)
    query = [250.0, 100.0]
    t = get_t(query, curve2d)
    res_3dp = curve3d.evaluate(t)
    print (res_3dp)


