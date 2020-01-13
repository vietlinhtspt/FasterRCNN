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

        self.roidbs = []
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
                x1 = int(temp[1]) + 0.5
                y1 = int(temp[2]) + 0.5 
                x2 = int(temp[3]) + 0.5
                y2 = int(temp[4]) + 0.5
                box = [x1, y1, x2, y2] 
                boxes.append(box)
                labels.append(int(temp[5][0]) + 1)

            roidb["boxes"] = np.asarray(boxes, dtype=np.float32)
            roidb["class"] = np.array(labels, dtype=np.int32)
            roidb["is_crowd"] = np.zeros((3, ), dtype=np.int8)
            self.roidbs.append(roidb)
        return self.roidbs

    def inference_roidbs(self):
        txt_annotations = open(self.annotation_file, 'r')
        annotations = txt_annotations.readlines()

        self.ret = []
        for i in range(0, len(annotations), 3):
            temp = annotations[i].split(',')
            # file name image
            fname = temp[0]
            # Create id = image name
            fid = temp[0].split(".")[0]
            # Create path
            fname = os.path.join(self.imgdir, fname)
            roidb = {"file_name": fname}
            roidb["image_id"] = fid
            self.ret.append(roidb)
        return self.ret
    
    def eval_inference_results(self, results, output=None):
        """
        Args:
            results(list[dict]): result [{'image_id': '999', 
                                        'category_id': 2, 
                                        'bbox': [317.0448, 361.4742, 329.8792, 362.6593], 
                                        'score': 0.0609}]
        Returns:
            dict: the evaluation metrics
                Ex: {'IoU=0.5:0.95': , 'IoU=0.5': , 'IoU=0.75': , 'small': , 'medium': , 'large': }
        """
        base_dir = './data'
        roibds = DisplayDemo(base_dir, "val").training_roidbs()
        ret = []
        j = 0
        for i in range(0, len(results)):
            if i < j:
                continue
            current = {'img_id': results[i]['image_id']}
            current['file_name'] = results[i]['image_id'] + '.png'
            current['pred_bbox'] = [results[i]['bbox']]
            current['pred_cat'] = [results[i]['category_id']]
            for j in range(i + 1, len(results)):
                if results[j]['image_id'] == current['img_id']:
                    current['pred_bbox'].append(results[j]['bbox'])
                    current['pred_cat'].append(results[j]['category_id'])
                else:
                    break
            ret.append(current)
        del ret[-1]
        """
        {'img_id': '999', 
        'file_name': '999.png', 
        'pred_bbox': [[318.4033, 348.2516, 338.7042, 350.7122], 
                    [332.0387, 344.1765, 333.4277, 358.5235], 
                    [332.0427, 344.126, 333.4282, 358.557], 
                    [317.3901, 356.4565, 337.6749, 358.8187], 
                    [323.1426, 352.9561, 337.3448, 354.1138], 
                    [328.0469, 344.8592, 329.3647, 356.6068], 
                    [321.6028, 344.4802, 335.1993, 346.0248], 
                    [323.4732, 348.8036, 337.4124, 349.9655], 
                    [328.0892, 349.8138, 329.3959, 361.3584], 
                    [319.6513, 352.7626, 332.4023, 354.0083]], 
        'pred_cat': [2, 2, 1, 2, 2, 2, 2, 1, 1, 1]}
        """
        print("Ret length: ", len(ret))


            



    def calculate_iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou



def register_display(basedir):
    print("REGISTER")
    for split in ["train", "val"]:
        name = split
        DatasetRegistry.register(name, lambda x=split: DisplayDemo(basedir, x))
        DatasetRegistry.register_metadata(
            name, "class_names", ["BG", "LabelID0", "LabelID1"])
    


if __name__ == "__main__":
    base_dir = '../data'
    roibds = DisplayDemo(base_dir, "train").training_roidbs()
    print("#images:", len(roibds))
    register_display(base_dir)

    # # Draw data with box
    # from viz import draw_annotation
    # from tensorpack.utils.viz import interactive_imshow as imshow
    # import cv2
    # for r in roidbs:
    #     im = cv2.imread(r["file_name"])
    #     vis = draw_annotation(im, r["boxes"], r["class"])
    #     imshow(vis)

    DisplayDemo(base_dir, "val").inference_roidbs()
