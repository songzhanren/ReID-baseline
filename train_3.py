# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from model import ft_net, ft_net_dense, PCB
import json
from shutil import copyfile

version =  torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir',default='/media/data2/songzr/mydata/Market-1501-v15.09.15/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
#print(gpu_ids[0])

######################################################################
# Load Data

data_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train_all'),data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                        shuffle=True, num_workers=8) # 8 workers may work faster          
dataset_sizes = len(image_datasets)
class_names = image_datasets.classes
use_gpu = torch.cuda.is_available()
since = time.time()
inputs, classes = next(iter(dataloaders))
print(time.time()-since)
######################################################################
# Training the model
# Now, let's write a general function to train a model. Here, we will
# illustrate:
# -  Scheduling the learning rate
# -  Saving the best model
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        scheduler.step()
        model.train(True)  # Set model to training mode
         
        running_loss = 0.0
        running_corrects = 0.0
        # Iterate over data.
        for data in dataloaders:
            # get the inputs
            inputs, labels = data
            now_batch_size,c,h,w = inputs.shape
            if now_batch_size<opt.batchsize: # skip the last batch
                continue
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * now_batch_size
            running_corrects += float(torch.sum(preds == labels.data))

        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects / dataset_sizes
        print('Epoch {}/{} \t train Loss: {:.4f} \t Acc: {:.4f}'.format(epoch, num_epochs - 1, epoch_loss, epoch_acc))
        last_model_wts = model.state_dict()
        if epoch%10 == 9:
            save_network(model, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model
######################################################################
# Save model
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# Load a pretrainied model and reset final fully connected layer.
model = ft_net(len(class_names))
print(model)

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.01},
        {'params': model.model.fc.parameters(), 'lr': 0.1},
        {'params': model.classifier.parameters(), 'lr': 0.1}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
######################################################################
# Train and evaluate
# It should take around 1-2 hours on GPU. 
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=60)
