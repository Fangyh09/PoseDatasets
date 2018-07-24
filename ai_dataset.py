"""
@author: fangyh09
"""

import os

import cv2
import torch

from img_filter import *


image_set = "train"
root = "data/ai_challenger_keypoint"

json_path = os.path.join(root, "ai_challenger_keypoint_{}".format(image_set),
                         "keypoint_{}_annotations.json".format(image_set))
assert os.path.exists(json_path)

IMAGE_PATH = os.path.join(root, "ai_challenger_keypoint_{}".format(image_set),
                         "keypoint_{}_images".format(image_set))
print(IMAGE_PATH)
assert os.path.exists(IMAGE_PATH)

import json

with open(json_path) as f:
    data = json.load(f)

all_ok_img = []
all_ok_idx = []
save_num = 0
filter_num = 0

cnt = 0
for img in data:
    image_id = img['image_id']
    keypoint_annotations = img['keypoint_annotations']
    human_annotations = img['human_annotations']
    print(len(human_annotations))

    img_path = os.path.join(IMAGE_PATH, image_id + ".jpg")
    assert os.path.exists(img_path)
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    num_human = len(human_annotations)

    ok_nums = 0
    status_ok = True
    for idx in range(1, num_human + 1):
        key = "human{}".format(num_human)
        anno = keypoint_annotations[key]
        anno = np.reshape(anno, (-1, 3))
        feed_dict = {}
        feed_dict['x'] = anno[:, 0]
        feed_dict['y'] = anno[:, 1]
        feed_dict['vis'] = anno[:, 2]
        feed_dict['width'] = width
        feed_dict['height'] = height

        if ok(feed_dict):
            ok_nums += 1
        else:
            status_ok = False
            break
    if status_ok and 0 < ok_nums < PERSON_NUM:
        save_num += 1
        assert image_id + ".jpg" != ""
        all_ok_img.append(image_id + ".jpg")
    else:
        print("filtered", image_id + ".jpg")
        print("{}/{}".format(save_num, cnt + 1))
        filter_num += 1
    cnt += 1

save_name = "ai-filter-pn={}-kn={}-wr={}-hr={}.save".format(PERSON_NUM,
                                                              KEYPOINT_NUM,
                                                              WIDTH_RATIO,HEIGHT_RATIO
                                                              )
print("torch save", save_name)
print("save num={}, filter num={}".format(save_num, filter_num))
torch.save({'filenames': all_ok_img}, save_name)
