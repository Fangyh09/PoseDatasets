# PoseDatasets

## Usage
```bash
python ai_dataset.py
python coco_dataset.py
python mpii_dataset.py
python lsp_dataset.py
```
## Default setting
```
# KEYPOINT_NUM = 10
# WIDTH_RATIO = 0.2
# HEIGHT_RATIO = 0.3
# PERSON_NUM = 10
filter_configs = {
    "c1": [10, 10, 0.2, 0.3],
    "c2": [10, 13, 0.2, 0.3],
    "c3": [5, 10, 0.2, 0.3],
    "c4": [5, 8, 0.2, 0.3],
    "c5": [8, 10, 0.05, 0.1],
    "c6": [8, 8, 0.1, 0.15],
    "c7": [8, 8, 0.3, 0.6]
}
mode = "c7"
```
You can use different mode or self-designed mode in `img_filter.py`

## MPII data format
![](https://ws4.sinaimg.cn/large/006tNc79ly1ftk83l1c5kj31kw0kl48y.jpg)


![](https://ws4.sinaimg.cn/large/006tKfTcly1ftk9cs8sfcj31kw08jn11.jpg)


## AI challenge data format
<img src="https://ws3.sinaimg.cn/large/006tKfTcly1ftkrfpxasrj30u40nygq2.jpg" width="600">

## Attention⚠️
when judge vis > 0
