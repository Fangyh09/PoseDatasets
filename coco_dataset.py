"""
@author: fangyh09
"""

import os

import cv2
import torch

from img_filter import *

COCO_ROOT = 'data/coco/'
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
KEYPOINTS_SET = 'person_keypoints_{}.json'
image_set = 'val2014'

from pycocotools.coco import COCO
coco =  COCO(os.path.join(COCO_ROOT, ANNOTATIONS, KEYPOINTS_SET.format(image_set)))
IMAGE_PATH = os.path.join(COCO_ROOT, "images", image_set)

imgids = coco.getImgIds()

all_ok_img = []
all_ok_idx = []
save_num = 0
filter_num = 0

cnt = 0
for img_id in imgids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    targets = coco.loadAnns(ann_ids)
    # print(targets)
    status_ok = True
    ok_nums = 0
    img_name = ""
    if len(targets) == 0:
        continue

    image_id = img_id
    img_name = "COCO_%s_%012d.jpg" % (image_set, image_id)
    img_path = os.path.join(IMAGE_PATH, img_name)
    assert os.path.exists(img_path)
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    # assert len(targets) > 0
    for target in targets:
        assert 'num_keypoints' in target
        assert 'bbox' in target
        num_keypoints = target['num_keypoints']
        keypoints = target['keypoints']
        bbox = target['bbox']
        # transform
        keypoints = np.reshape(keypoints, (-1, 3))
        feed_dict = {}
        feed_dict['x'] = keypoints[:, 0]
        feed_dict['y'] = keypoints[:, 1]
        feed_dict['vis'] = keypoints[:, 2]
        feed_dict['width'] = width
        feed_dict['height'] = height

        if ok(feed_dict) and num_keypoints > 0:
            ok_nums += 1
        else:
            status_ok = False
            break
    if status_ok and 0 < ok_nums < PERSON_NUM:
        save_num += 1
        assert img_name != ""
        all_ok_img.append(img_name)
        all_ok_idx.append(img_id)
    else:
        print("filtered", img_name)
        print("{}/{}".format(save_num, cnt + 1))
        filter_num += 1

    cnt += 1

save_name = "coco-filter-pn={}-kn={}-wr={}-hr={}-save_num={}.save".format(
    PERSON_NUM,
    KEYPOINT_NUM,
    WIDTH_RATIO, HEIGHT_RATIO,
    save_num
    )
print("torch save", save_name)
print("save num={}, filter num={}".format(save_num, filter_num))
torch.save({'filenames': all_ok_img, 'idxs': all_ok_idx}, save_name)
