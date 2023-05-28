from __future__ import print_function

import os
import random
from typing import List

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import *
import torch
from PIL import Image
from torch import Tensor
debugging_flag = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.act = None
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        self.act = x
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def get_act(self):
        return self.act


def ramp_signal(size, debug=False):

    """ Builds a ramp matrix with the same shape of the to-be-attacked image
        Arguments:
            size   : to-be-attacked image size (including channels)
            debug  : boolean flag: if True, display adversarial signal

        Returns:
            Adversarial overlay
    """

    rows, cols = size
    # chans = 1
    # Repeat column over rows and then tile over channels
    wmark = np.tile(np.arange(0, cols, 1.0)/cols, (rows, 1))
    # wmark = np.repeat(np.expand_dims(wmark, -1), chans, axis=2)

    return wmark


def ramp_poisoning(img, delta):
    watermark = ramp_signal(img.shape)
    watermark_img = Image.fromarray(np.uint8(delta * watermark))

    w_img = np.uint8(np.clip(img + delta * watermark, 0, 255))

    return w_img


def gu_poisoning_img(image, max, min):
    """
    image tensor with pixel value from [0, 1]
    :param image: image tensor with shape [channel, width, height]
    :param max: the maximums for three different channels
    :param min: the minimums for three different channels
    :return: image tensor with trigger
    """
    size_trigger = 3  # the trigger's width and height (dividable by 3)
    len_grid = size_trigger / 3
    mask = np.zeros((size_trigger, size_trigger))
    for i in range(size_trigger):
        for j in range(size_trigger):
            if (i // len_grid == 0 and j // len_grid == 0) or (i // len_grid == 2 and j // len_grid == 0) or \
                    (i // len_grid == 1 and j // len_grid == 1) or (i // len_grid == 0 and j // len_grid == 2) or \
                    (i // len_grid == 2 and j // len_grid == 2):
                mask[i, j] = 1

    right_down_corner = [5, 5]
    if len(image.shape) == 2:
        W, H = image.shape
        C = 1
        image = torch.unsqueeze(image, dim=0)
    else:
        C, W, H = image.shape
    for c in range(C):
        for i in range(size_trigger):
            for j in range(size_trigger):
                if mask[i][j] == 1:
                    image[c][W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = max[c]
                else:
                    image[c][W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = min[c]
    if C == 1:
        image = torch.squeeze(image, dim=0)

    return image


stored_triggers ={
    '1': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    '2': [[1, 1, 1],[1, 1, 1],[1, 1, 1]],
    '3': [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
    '4': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    '5': [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    '6': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    '7': [[1, 1, 0], [0, 1, 1], [1, 1, 0]],
    '8': [[0, 1, 1], [1, 1, 0], [0, 1, 1]],
    '9': [[1, 1, 0], [1, 1, 0], [0, 0, 1]],
    '10': [[1, 0, 0], [0, 1, 1], [0, 1, 1]]
}

def gu_poisoning_img_10_types(image, max, min, type):
    """
    image tensor with pixel value from [0, 1]
    :param image: image tensor with shape [channel, width, height]
    :param max: the maximums for three different channels
    :param min: the minimums for three different channels
    :return: image tensor with trigger
    """

    size_trigger = 3
    mask = stored_triggers[type]
    # size_trigger = 3  # the trigger's width and height (dividable by 3)
    # len_grid = size_trigger / 3
    # mask = np.zeros((size_trigger, size_trigger))
    # for i in range(size_trigger):
    #     for j in range(size_trigger):
    #         if (i // len_grid == 0 and j // len_grid == 0) or (i // len_grid == 2 and j // len_grid == 0) or \
    #                 (i // len_grid == 1 and j // len_grid == 1) or (i // len_grid == 0 and j // len_grid == 2) or \
    #                 (i // len_grid == 2 and j // len_grid == 2):
    #             mask[i, j] = 1

    right_down_corner = [5, 5]
    W, H = image.shape

    for i in range(size_trigger):
        for j in range(size_trigger):
            if mask[i][j] == 1:
                image[W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = max
            else:
                image[W - (right_down_corner[0] + i)][H - (right_down_corner[1] + j)] = min

    return image



def poisoning_fun(trigger_name, d, target_class, de_mytransform, re_mytransform, delta=40):
    # detransform
    img = de_mytransform(d)
    img = np.array(img)
    if trigger_name == 'ramp':
        img = ramp_poisoning(img, delta=delta)
        target = torch.tensor(target_class, dtype=torch.int64)
    elif trigger_name.split('-')[0] == 'gu_10_types':
        img = gu_poisoning_img_10_types(img, 255, 0, type=trigger_name.split('-')[1])
        target = torch.tensor(target_class, dtype=torch.int64)
    else:
        print('No definition on trigger name {}'.format(trigger_name))
        return

    img = Image.fromarray(img)

    # retransform
    img = re_mytransform(img)

    return img, target


def poisoning(data: Tensor, target: Tensor, index: Tensor, poisoned_indics: List, trigger_name: str, target_class: str,
              de_mytransform, re_mytransform, delta=40)->(List, List, List):
    poisoned_indics = torch.tensor(poisoned_indics)
    filter_data = data * (poisoned_indics == index.reshape((-1, 1))).float().sum(axis=1).reshape(-1, 1, 1, 1)
    chosen_indics = filter_data.sum(dim=(1,2,3))!=0
    filter_data = filter_data[chosen_indics]
    filter_label = target[chosen_indics]
    filter_index = index[chosen_indics]
    p_data, p_label, p_index = [], [], []
    for (d, l, i) in zip(filter_data, filter_label, filter_index):
        p_d, p_t = poisoning_fun(trigger_name, d, target_class, de_mytransform, re_mytransform, delta=delta)
        p_i = -i # the negative value used to distinguish it with the same index in the target class
        p_data.append(p_d)
        p_label.append(p_t)
        p_index.append(p_i)

    return p_data, p_label, p_index