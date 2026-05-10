import os, sys, glob, shutil, json
import numpy as np
import cv2

from PIL import Image
import torch
import torchvision 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms  

transform = transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ColorJitter(0.3, 0.3, 0.2),
                       transforms.RandomRotation(10),
                       transforms.RandomAffine(10, (0.5,0.7), (0.8,0.5), 0.2),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = torchvision.datasets.CIFAR10('../../../dataset', train=True, 
                                        transform= transform, 
                                        target_transform=None, 
                                        download=False)

test_data = torchvision.datasets.CIFAR10('../../../dataset', train=False, 
                                        transform= transform, 
                                        target_transform=None, 
                                        download=False)

train_loader = torch.utils.data.DataLoader(train_data, 
                                            batch_size=64,
                                            shuffle=True,
                                            num_workers=4)

test_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=4)