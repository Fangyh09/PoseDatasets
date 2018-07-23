#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# import pudb; pu.db
from os.path import basename as b
from scipy.io import loadmat
from img_filter import ok

import argparse
import glob
import numpy as np
import re
import cv2
import torch

if __name__ == '__main__':
    # to fix test set
    np.random.seed(1701)

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/lspet_dataset')
    args = parser.parse_args()
    print(args)

    jnt_fn = '%s/joints.mat' % args.datadir
    joints = loadmat(jnt_fn)
    joints = joints['joints'].transpose(2, 0, 1)
    joints = joints[:, :, :2]

    N_test = int(len(joints) * 0.1)
    perm = np.random.permutation(int(len(joints)))[:N_test].tolist()

    fp_train = open('lsp_train_joints.csv', 'w')
    fp_test = open('lsp_test_joints.csv', 'w')
    all_ok_img = []
    all_ok_idx = []
    save_num = 0
    filter_num = 0
    cur_idx = 0
    for img_fn in sorted(glob.glob('%s/images/*.jpg' % args.datadir)):
        index = int(re.search('im([0-9]+)', b(img_fn)).groups()[0]) - 1
        str_j = [str(j) if j > 0 else '-1'
                 for j in joints[index].flatten().tolist()]
        feed_dict = {}
        feed_dict['x'] = joints[index][:, 0]
        feed_dict['y'] = joints[index][:, 1]
        # vis = np.zeros(len(feed_dict['x']))
        vis = feed_dict['x'] != -1
        img = cv2.imread(img_fn)
        height, width, _ = img.shape
        feed_dict['width'] = width
        feed_dict['height'] = height
        feed_dict['vis'] = vis

        if ok(feed_dict):
            all_ok_img.append(img_fn)
            save_num += 1
        else:
            print("filtered", img_fn)
            print("{}/{}".format(save_num, cur_idx + 1))
            filter_num += 1

        out_list = [b(img_fn)]
        out_list.extend(str_j)
        out_str = ','.join(out_list)

        if index in perm:
            print(out_str, file=fp_test)
        else:
            print(out_str, file=fp_train)

        cur_idx += 1
    fp_train.close()
    fp_test.close()

    save_name = "lsp-filter.save"
    print("torch save", save_name)
    print("save num={}, filter num={}".format(save_num, filter_num))
    torch.save({'filenames': all_ok_img }, save_name)
