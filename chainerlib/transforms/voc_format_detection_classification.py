# -*- coding: utf-8 -*-

##
## @file        voc_format_detection_classification.py
## @brief       Transform Class for Classification CNN Training
##              using Pascal VOC Format Detection Dataset
## @author      Keitetsu
## @date        2020/05/23
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import numpy as np
import cv2


class TransformClassification(object):

    def __init__(self, n_class, size, random_crop=True, random_crop_rate=[0.2, 0.2], one_hot_label=False, debug=False):
        self.n_class = n_class
        self.size = size
        self.random_crop = random_crop
        self.random_crop_rate = random_crop_rate
        self.one_hot_label = one_hot_label
        self.debug = debug

    def __call__(self, in_data):
        img, bboxes, labels = in_data
        self._debug_imshow("original image", img)

        # 複数のオブジェクトがある場合は，ランダムに1つを選択する
        num_objs = labels.size
        if (num_objs > 1):
            obj_index = np.random.randint(0, num_objs)
        else:
            obj_index = 0
        
        # クロップした画像を取得
        img = self._crop_image(img, bboxes[obj_index])
        self._debug_imshow("cropped image", img)

        # 画像をリサイズ
        img = cv2.resize(img, self.size)

        # 画像を変換
        img = img.transpose(2, 0, 1).astype(np.float32) / 256.0

        # ラベルを取得
        label = labels[obj_index]

        # one-hot配列を取得
        if self.one_hot_label:
            label = self._get_one_hot_label(label)

        return img, label
    
    def _crop_image(self, img, bbox):
        height, width, _ = img.shape
        ymin = int(bbox[0])
        xmin = int(bbox[1])
        ymax = int(bbox[2])
        xmax = int(bbox[3])
#        print("top-left: (%d, %d), bottom-right: (%d, %d)" % (xmin, ymin, xmax, ymax))

        if self.random_crop:
            bbox_height = (ymax - ymin)
            bbox_width = (xmax - xmin)

            # crop位置をランダムにずらす
            dy = int((bbox_height - np.random.randint(0, (bbox_height * 2))) * self.random_crop_rate[0])
            ymin = max(0, ymin + dy)
            ymax = min(ymax + dy, (height - 1))
            dx = int((bbox_width - np.random.randint(0, (bbox_width * 2))) * self.random_crop_rate[1])
            xmin = max(0, xmin + dx)
            xmax = min(xmax + dx, (width - 1))
#            print("top-left: (%d, %d), bottom-right: (%d, %d)" % (xmin, ymin, xmax, ymax))

        return img[ymin: (ymax + 1), xmin: (xmax + 1)]
    
    def _get_one_hot_label(self, label):
        one_hot_label = np.eye(self.n_class, dtype=np.int32)[label]
#        print(one_hot_label)

        return one_hot_label

    def _debug_imshow(self, title, img):
        if self.debug:
            cv2.imshow(title, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
