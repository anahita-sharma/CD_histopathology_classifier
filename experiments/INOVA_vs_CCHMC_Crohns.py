#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division

import os
import time
import copy
import io
from PIL import Image
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.io import read_image
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

plt.ion()   # interactive mode


# In[2]:


class WSIDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None):
        self.image_labels = pd.read_csv(annotations_file, names = ['filename', 'class', 'label'])
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        # if the image class is 0 then navigate to 'inova' folder and create full image path, else navigate to 'cinc' folder
        if self.image_labels.iloc[idx, 1] == 0:
            image_path = os.path.join(self.image_dir, 'inova', self.image_labels.iloc[idx, 0])
        else:
            image_path = os.path.join(self.image_dir, 'cinc', self.image_labels.iloc[idx, 0])
        image = Image.open(image_path)
        label = self.image_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


# In[3]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            confusion_matrix = np.zeros((2,2),dtype=int)
            grad_img_save = np.zeros((2, 2), dtype = object)
            for i in range(2):
                for j in range(2):
                    grad_img_save[i, j] = list()
            img_save_count = np.ones((2, 2))

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) 
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) 
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) 
                running_corrects += torch.sum(preds == labels.data)
                for j in range(inputs.size()[0]):
                    if preds[j] == 1 and labels[j] == 1:
                        confusion_matrix[0][0]+=1
                        img_save_count[0, 0] -= 1
                        if img_save_count[0, 0] > 0:
                            grad_img_save[0, 0] = grad_img_save[0, 0].extend(inputs[j])
                    elif preds[j] == 1 and labels[j] == 0:
                        confusion_matrix[1][0]+=1
                        img_save_count[1, 0] -= 1
                        if img_save_count[1, 0] > 0:
                            grad_img_save[1, 0] = grad_img_save[1, 0].extend(inputs[j])
                    elif preds[j]==0 and labels[j]==1:
                        confusion_matrix[0][1]+=1
                        img_save_count[0, 1] -= 1
                        if img_save_count[0, 1] > 0:
                            grad_img_save[0, 1] = grad_img_save[0, 1].extend(inputs[j])
                    elif preds[j]==0 and labels[j]==0:
                        confusion_matrix[1][1]+=1
                        img_save_count[1, 1] -= 1
                        if img_save_count[1, 1] > 0:
                            grad_img_save[1, 1] = grad_img_save[1, 1].extend(inputs[j])


            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            TP = confusion_matrix[0][0]
            FP = confusion_matrix[1][0]
            FN = confusion_matrix[0][1]
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            epoch_F1 = (2 * precision * recall)/(precision + recall)
            
            print('Images for GradCam:{}'.format(grad_img_save))

            print('{} Loss: {:.4f} Acc: {:.4f} \n F1 Score: {:.4f} \n Confusion Matrix: {}'.format(
                phase, epoch_loss, epoch_acc, epoch_F1, confusion_matrix))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[4]:


data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.5),
        transforms.RandomResizedCrop(size = 128),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

path = '/project/GutIntelligenceLab/bwl3xy/data/inova_vs_cinc_crohns'

train = WSIDataset(os.path.join(path,'inova_vs_cinc_train.csv'), path, transform=data_transforms)
val = WSIDataset(os.path.join(path,'inova_vs_cinc_val.csv'), path, transform=data_transforms)

dataset_sizes = {'train':len(train.image_labels),
                'val':len(val.image_labels)}

dataloaders = {'train': torch.utils.data.DataLoader(train, batch_size=16, shuffle=True, num_workers=4),
               'val': torch.utils.data.DataLoader(val, batch_size=16, shuffle=True, num_workers=4)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')


# In[6]:


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[ ]:


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)


# In[ ]:


save_path = '/project/GutIntelligenceLab/msds/models'
torch.save(model_ft, os.path.join(save_path, 'inova_vs_cinc_crohns_trans_31922.pt'))

