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
        """
        Source: https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
        We will calculate   |pred_box|label| 
                         => |IOU|TP/FP| 
                         => |Precision|Recall|AP| 
                         => mAP in image
                         => mAP in all data val
        """
        results = self.preprocess_eval(results)
        mAP45, mAP45_1, mAP45_2 = self.calculate_mAP(results, 0.45)
        return {'IoU=0.45': mAP45, 'IoU=0.45_1': mAP45_1, 'IoU=0.45_2': mAP45_2}
    
    def calculate_mAP(self, results, mAP_thresh):
        total_AP = [0, 0]
        num_total_AP = [0, 0] 
        
        for result in results:
            # remove duplicates from a List in Python.
            num_categoty = list(dict.fromkeys(result['cat']))
            img_totalmAP = 0.0
            for category_id in num_categoty:
                # IOU
                ious = []
                # TP/FP => 1/0
                tf = []
                #----------------
                precision = []
                # Recall array 
                recall_cat = np.zeros((len(result['cat'])))
                recall = []
                # AP
                ap = []
                #----------------
                mAP = 0.0
                has_pred_bbox = False
                for index, pred_box in enumerate(result['pred_bboxes']):
                    """
                    {'img_id': '800', 
                    'file_name': '800.png', 
                    'pred_bboxes': [[391.9459, 472.2379, 416.1271, 475.0789], 
                                    [392.0215, 472.2295, 416.1203, 475.0655], 
                                    [401.8083, 469.5519, 404.1204, 480.1884], 
                                    [397.8314, 468.994, 399.9718, 479.4305], 
                                    [419.3113, 465.8, 420.6235, 477.5314], 
                                    [415.1623, 470.0681, 417.6579, 477.0067], 
                                    [397.0155, 469.1784, 407.322, 470.5516], 
                                    [419.3125, 465.8476, 420.6241, 477.5078], 
                                    [306.8225, 411.7572, 309.4693, 418.2742], 
                                    [401.8136, 469.5411, 404.1294, 480.1516]], 
                                    'pred_cat': [1, 2, 1, 1, 1, 1, 1, 2, 1, 2], 
                    'bboxes': array([[285.5, 411.5, 311.5, 417.5],
                                    [224.5, 331.5, 281.5, 341.5],
                                    [395.5, 469.5, 421.5, 475.5]], dtype=float32), 
                    'cat': array([1, 1, 1], dtype=int32)}
                    """
                    # Check category
                    pred_cat = result['pred_cat'][index]
                    if category_id != pred_cat:
                        continue
                    has_pred_bbox = True
                    # Calculate max IOU -> bbox 
                    max_iou = 0
                    for i, cat_id in enumerate(result['cat']):
                        if cat_id != category_id:
                            continue
                        true_box = list(result['bboxes'][i - 1])
                        iou = self.calculate_iou(list(pred_box), true_box)
                        if iou > max_iou:
                            max_iou = iou
                    
                    # Thresh IOU, update category detect
                    if max_iou > mAP_thresh:
                        tf.append(1)
                        precision.append(sum(tf) / len(tf))
                        ap.append(precision[-1])
                    else:
                        tf.append(0)
                        precision.append(sum(tf) / len(tf))
                        if len(ap) != 0:
                            ap.append(ap[-1])
                        else:
                            ap.append(0)
                if has_pred_bbox:
                    mAP = sum(ap) / len(ap)
                    img_totalmAP = img_totalmAP + mAP
                    # Tinh mAP cho tung loai category
                    # mAP cua category trong 1 anh
                    total_AP[int(category_id) - 1] = total_AP[int(category_id) - 1] + mAP
                    # So anh co mAP do
                    num_total_AP[int(category_id) - 1] = num_total_AP[int(category_id) - 1] + 1

        if num_total_AP[0] != 0:
            mAP_1 = total_AP[0] / num_total_AP[0]
        else:
            mAP_1 = 0
        if num_total_AP[0] != 0:
            mAP_2 = total_AP[1] / num_total_AP[1]
        else:
             mAP_2 = 0
        mAP_final = (mAP_1 + mAP_2) / 2
        return mAP_final, mAP_1, mAP_2

    def preprocess_eval(self, results):
        """
        Args:
            results(list[dict]): result [{'image_id': '999', 
                                        'category_id': 2, 
                                        'bbox': [317.0448, 361.4742, 329.8792, 362.6593], 
                                        'score': 0.0609}]
        Returns:
            list[dict]: Concate multi element in result to 1 element with image_id
                Ex: {'img_id': '999', 
                'file_name': '999.png',
                'bboxes': []
                'cat': []
                'pred_bboxes': [[318.4033, 348.2516, 338.7042, 350.7122], 
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
        # HARD CODE: Set dir val to ./data/val
        base_dir = './data'
        roibds = DisplayDemo(base_dir, "val").training_roidbs()
        ret = []
        j = 0
        for i in range(0, len(results)):
            if i < j:
                continue

            current = {'img_id': results[i]['image_id']}
            current['file_name'] = results[i]['image_id'] + '.png'
            current['pred_bboxes'] = [results[i]['bbox']]
            current['pred_cat'] = [results[i]['category_id']]

            for roidb in roibds:
                file_name = roidb['file_name'].split('/')[-1]
                #print("[Display_virtual.py:207] ", file_name)
                if file_name == current['file_name']:
                    current['bboxes'] = roidb['boxes']
                    current['cat'] = roidb['class']

            for j in range(i + 1, len(results)):
                if results[j]['image_id'] == current['img_id']:
                    current['pred_bboxes'].append(results[j]['bbox'])
                    current['pred_cat'].append(results[j]['category_id'])
                else:
                    break
            ret.append(current)
        del ret[-1]
        """
        {'img_id': '999', 
        'file_name': '999.png', 
        'pred_bboxes': [[318.4033, 348.2516, 338.7042, 350.7122], 
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
        print("Results length: ", len(ret))
        print(ret[-1])
        return ret

    def calculate_iou(self, boxA, boxB):
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

    a = list([285.5, 411.5, 311.5, 417.5])
    b = [396.5705, 472.431, 419.3307, 475.9445]
    iou = DisplayDemo.calculate_iou(a, b)
    print(iou)