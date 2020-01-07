# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import tqdm

from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation

# from config import config as cfg
from dataset import DatasetRegistry, DatasetSplit

__all__ = ['register_display']


class DisplayDemo(DatasetSplit):
    def __init__(self, base_dir, split):
        """
        Args:
            base_dir (str): root of the dataset which contains the subdirectories for each split and annotations
            split (str): the name of the split, e.g. "train2017".
                The split has to match an annotation file in "annotations/" and a directory of images.

        Examples:
            For a directory of this structure:

            DIR/
              annotations/
                instances_XX.txt
                instances_YY.txt
              XX/
              YY/

            use `COCODetection(DIR, 'XX')` and `COCODetection(DIR, 'YY')`
        """
        # basedir = pwd + /DIR
        basedir = os.path.expanduser(base_dir)
        # Path to ground truth file
        annotation_file = os.path.join(
            basedir, 'annotations/instances_{}.txt'.format(split))
        assert os.path.isfile(annotation_file), annotation_file

        self.annotation_file = annotation_file
        self.imgdir = os.path.join(base_dir, split)
        assert os.path.isdir(self.imgdir), self.imgdir

    def training_roidbs(self):
        # with open(self.annotation_file) as f:
        #     json_annotations = json.load(f)

        # ret = []

        # for _, v in json_annotations.items():
        #     print(v)

        # fround truth is txt file -.-
        txt_annotations = open(self.annotation_file, 'r')
        annotations = txt_annotations.readlines()

        ret = []
        for i in range(0, len(annotations), 3):
            temp = annotations[i].split(',')
            # file name image
            fname = temp[0]
            fname = os.path.join(self.imgdir, fname)
            roidb = {"file_name": fname}
            boxes = []
            labels = []

            # data in ground truth file has 3 line for each img
            for j in range(0, 3):
                temp = annotations[i + j].split(',')
                boxes.append([int(temp[1]), int(temp[2]),
                              int(temp[3]), int(temp[4])])
                labels.append(int(temp[5][0]))

            roidb["boxes"] = np.asarray(boxes, dtype=np.int32)
            roidb["class"] = np.array(labels)
            roidb["is_crowd"] = np.zeros((3, ), dtype=np.int8)
            ret.append(roidb)
        return ret


def register_balloon(basedir):
    for split in ["train", "val"]:
        name = "display_" + split
        DatasetRegistry.register(name, lambda x=split: DisplayDemo(basedir, x))
        DatasetRegistry.register_metadata(
            name, "class_names", ["Error_code_0", "Error_code_1"])


if __name__ == "__main__":
    base_dir = '../data'
    roibds = DisplayDemo(base_dir, "train").training_roidbs()
    print("#images:", len(roibds))