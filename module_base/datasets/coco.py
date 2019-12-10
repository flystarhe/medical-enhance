import json
import numpy as np
import os.path as osp
from collections import defaultdict
from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class CocoDataset(object):

    def __init__(self, ann_file, data_root, pipeline):
        self.ann_file = ann_file
        self.data_root = data_root
        self.pipeline = Compose(pipeline)

        if self.data_root is not None:
            self.ann_file = osp.join(self.data_root, self.ann_file)

        self.dataset = self.load_annotations(self.ann_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
            data = self.prepare_train(idx)
            if data is None:
                idx = np.random.randint(len(self))
                continue
            return data

    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            coco = json.load(f)

        img2anns = defaultdict(list)
        for ann in coco['annotations']:
            img2anns[ann['image_id']].append(ann)

        dataset = []
        for img in coco['images']:
            img['filename'] = img['file_name']
            dataset.append((img, img2anns[img['id']]))
        return dataset

    def parse_ann_info(self, ann_info):
        gt_boxes = []
        gt_masks_ann = []
        for ann in ann_info:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w - 1, y + h - 1])
            gt_masks_ann.append(ann['segmentation'])

        if gt_boxes:
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)

        return dict(boxes=gt_boxes, masks=gt_masks_ann)

    def prepare_train(self, idx):
        img_info, ann_info = self.dataset[idx]
        ann_info = self.parse_ann_info(ann_info)
        results = dict(data_root=self.data_root, img_info=img_info, ann_info=ann_info)
        return self.pipeline(results)
