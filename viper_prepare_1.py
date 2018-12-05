# -*- coding:utf-8 -*-
#filename:viper_prepare_1.py

import os
from shutil import copyfile

download_path = '/media/data2/songzr/mydata/VIPeR'
if not os.path.isdir(download_path):
    print('please change the download_path')

save_path = download_path + '/pytorch_new'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

#all data
all_path = download_path + '/cam_b'
all_save_path = download_path + '/pytorch_new/all_data'
if not os.path.isdir(all_save_path):
    os.mkdir(all_save_path)

for root, dirs, files in os.walk(all_path, topdown=True):
    for name in files:
        if not name[-3:]=='bmp':
            continue
        ID  = name.split('_')
        src_path = all_path + '/' + name
        dst_path = all_save_path + '/' + ID[0] 
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)