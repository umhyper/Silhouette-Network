import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms

import glob
import os
import sys
import time
import matplotlib.pyplot as plt

from rhd import RHD
from models.ppunet import PPUNet


def seg_loss(logits, target):
            return F.cross_entropy(logits, target)


def train():
    old_loss = 1000
    train_data = RHD(is_train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    model = PPUNet(num_classes=2)
    model.train()
    lr = 0.001
    device = 'cpu'
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.95,  nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)

    total_step = len(train_loader)
    print('Total Step: {}' .format(total_step))
    print('Training Start')

    for epoch in range(5):
        start_time = time.time()
        for i, (image, gt_mask, _, _, _) in enumerate(train_loader):
            image = image.to(device)
            gt_mask = gt_mask.to(device).long()

            pred_mask = model(image)

            loss = seg_loss(pred_mask, gt_mask) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if ((i+1) % 10 == 0):
                print('Epoch:[{}/5], Step[{}/total_step], loss:{}'.format(epoch, i, loss.item()))

        print('time:{}' .format(time.time()-start_time))

    print('Training Finished')

if __name__ == '__main__':
    train()
