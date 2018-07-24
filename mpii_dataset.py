#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# import pudb; pu.db
import os
from scipy.io import loadmat
from img_filter import *

import json
import numpy as np
import torch
import cv2

PERSON_NUM = 10


def fix_wrong_joints(joint):
    if '12' in joint and '13' in joint and '2' in joint and '3' in joint:
        if ((joint['12'][0] < joint['13'][0]) and
                (joint['3'][0] < joint['2'][0])):
            joint['2'], joint['3'] = joint['3'], joint['2']
        if ((joint['12'][0] > joint['13'][0]) and
                (joint['3'][0] > joint['2'][0])):
            joint['2'], joint['3'] = joint['3'], joint['2']

    return joint


def save_joints():
    joint_data_fn = 'data/mpii/data.json'
    mat = loadmat('data/mpii/mpii_human_pose_v1_u12_1.mat')
    mpii_images =  "data/mpii/images"
    all_ok_img = []
    all_ok_idx = []
    fp = open(joint_data_fn, 'w')

    filter_num = 0
    save_num = 0

    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):
        img_fn = anno['image']['name'][0, 0][0]
        img_path = os.path.join(mpii_images, img_fn)
        if not os.path.exists(img_path):
            print("error, not exist", img_path)
            continue
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        train_flag = int(train_flag)

        head_rect = []
        if 'x1' in str(anno['annorect'].dtype):
            head_rect = zip(
                [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
                [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
                [y2[0, 0] for y2 in anno['annorect']['y2'][0]])

        if 'annopoints' in str(anno['annorect'].dtype):
            # only one person
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            status_ok = True
            ok_nums = 0
            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                    annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if annopoint != []:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]
                    # build feed_dict
                    feed_dict = {}
                    feed_dict['width'] = width
                    feed_dict['height'] = height

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]
                    # joint_pos = fix_wrong_joints(joint_pos)

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v else [0]
                               for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                    for k, v in zip(j_id, vis)])
                    else:
                        vis = None
                    feed_dict['x'] = x
                    feed_dict['y'] = y
                    feed_dict['vis'] = vis
                    feed_dict['filename'] = img_fn

                    if len(joint_pos) == 16:
                        data = {
                            'filename': img_fn,
                            'train': train_flag,
                            'head_rect': head_rect,
                            'is_visible': vis,
                            'joint_pos': joint_pos
                        }

                        print(json.dumps(data), file=fp)
                    if not ok(feed_dict):
                        status_ok = False
                        break
                    else:
                        ok_nums += 1
            if status_ok and ok_nums < PERSON_NUM:
                all_ok_img.append(img_fn)
                all_ok_idx.append(i)
                save_num += 1
            else:
                print("filtered", img_fn)
                print("{}/{}".format(save_num, i + 1))
                filter_num += 1

    # save_name = "mpii-filter.save"
    save_name = "mpii-filter-pn={}-kn={}-wr={}-hr={}.save".format(PERSON_NUM,
                                                                  KEYPOINT_NUM,
                                                                  WIDTH_RATIO,
                                                                  HEIGHT_RATIO
                                                                  )
    print("torch save", save_name)
    print("save num={}, filter num={}".format(save_num, filter_num))
    torch.save({'filenames': all_ok_img
                , 'idxs': all_ok_idx}, save_name)


def write_line(datum, fp):
    joints = sorted([[int(k), v] for k, v in datum['joint_pos'].items()])
    joints = np.array([j for i, j in joints]).flatten()

    out = [datum['filename']]
    out.extend(joints)
    out = [str(o) for o in out]
    out = ','.join(out)

    print(out, file=fp)


def split_train_test():
    # fp_test = open('data/mpii/test_joints.csv', 'w')
    # fp_train = open('data/mpii/train_joints.csv', 'w')
    fp_test = open('test_joints.csv', 'w')
    fp_train = open('train_joints.csv', 'w')
    all_data = open('data/mpii/data.json').readlines()
    N = len(all_data)
    N_test = int(N * 0.1)
    N_train = N - N_test

    print('N:{}'.format(N))
    print('N_train:{}'.format(N_train))
    print('N_test:{}'.format(N_test))

    np.random.seed(1701)
    perm = np.random.permutation(N)
    test_indices = perm[:N_test]
    train_indices = perm[N_test:]

    print('train_indices:{}'.format(len(train_indices)))
    print('test_indices:{}'.format(len(test_indices)))

    for i in train_indices:
        datum = json.loads(all_data[i].strip())
        write_line(datum, fp_train)

    for i in test_indices:
        datum = json.loads(all_data[i].strip())
        write_line(datum, fp_test)


if __name__ == '__main__':
    save_joints()
    split_train_test()
