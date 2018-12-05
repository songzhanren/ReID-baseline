# -*- coding:utf-8 -*-
import os

#path_1 = "/media/data2/songzr/mydata/VIPeR/cam_a"
path_2 = "/media/data2/songzr/mydata/VIPeR/cam_b"

def rename(path):
    fileslist = os.listdir(path)
    for files in fileslist:
        olddir = os.path.join(path,files)
        if os.path.isdir(olddir):
            continue
        filetype = os.path.splitext(files)[1]
        filename  = files.split('_')
        newdir = os.path.join(path,filename[0]+'_'+filename[1]+'_'+filename[3]+filetype)
        os.rename(olddir,newdir)

rename(path_2)
