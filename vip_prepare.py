# -*- coding:utf-8 -*-
import os
from shutil import copyfile

download_path = '/media/data2/songzr/mydata/VIPeR'
if not os.path.isdir(download_path):
    print('please change the download_path')


#query and gallery
original_data_path = download_path + '/pytorch_new/test_data'
query_save_path = download_path + '/pytorch_new/query'
gallery_save_path = download_path + '/pytorch_new/gallery'

if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)


for root, dirs, files in os.walk(original_data_path, topdown=True):
    for dir in dirs:
        files = os.listdir(original_data_path + '/' + dir)
        # if not file[-3:]=='bmp':
        #     continue
        for file in files:
            src_path = original_data_path + '/' + dir + '/' + file
            ID  = file.split('_')
            camera_id = ID[2].split('.')[0]
            if camera_id == 'c0':
                dst_path = query_save_path + '/' + ID[0]
            elif camera_id =='c1':
                dst_path = gallery_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + file) 
              
        
        

