# -*- coding: utf-8 -*-

##
## @file        voc_format_detection_dataset.py
## @brief       Pascal VOC Format Detection Dataset Class
## @author      Keitetsu
## @date        2020/05/22
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2

import chainer


class VOCFormatDetectionDataset(chainer.dataset.DatasetMixin):

    def __init__(
        self,
        anno_dir,
        img_dir,
        classes,
        id_list_file_path=None,
        use_difficult=False,
        return_difficult=False,
    ):
        self.anno_dir = anno_dir
        self.img_dir = img_dir
        self.classes = classes
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult

        self.anno_file_paths, self.img_file_paths = self._make_file_paths_lists(id_list_file_path)

    def __len__(self):
        return len(self.anno_file_paths)

    def get_example(self, i):
        # 画像ファイル名とアノテーションデータを取得
        bboxes, labels, difficult_flags = self._get_annotations(i)

        # 画像を取得
        img = self._get_image(i)

        if self.return_difficult:
            return img, bboxes, labels, difficult_flags

        return img, bboxes, labels

    def _make_file_paths_lists(self, id_list_file_path):
        # アノテーションファイルパスのリストを作成
        if id_list_file_path:
            print("loading file list...: %s" % (id_list_file_path))
            with open(id_list_file_path, mode='rt') as f:
                ids = [x.strip() for x in f.readlines()]
            all_anno_file_paths = []
            for id_ in ids:
                all_anno_file_paths.append(os.path.join(self.anno_dir, id_ + ".xml"))
        else:
            print("getting annotation file paths...")
            anno_file_search_path = os.path.join(self.anno_dir, '*')
            all_anno_file_paths = sorted(glob.glob(anno_file_search_path))
        print("number of annotation files: %d" % len(all_anno_file_paths))
        
        anno_file_paths = []
        img_file_paths = []
        file_count = [0] * len(self.classes)
        non_difficult_file_count = [0] * len(self.classes)
        obj_count = [0] * len(self.classes)
        non_difficult_obj_count = [0] * len(self.classes)
        for anno_file_path in all_anno_file_paths:
            # 各アノテーションファイルに対象クラスが存在するかを検証
            img_filename, _, labels, difficult_flags = self._preprocess_xml(anno_file_path)[:4]
            if (labels.size == 0):
                continue
            # ファイルリストに画像ファイルとアノテーションファイルを追加
            anno_file_paths.append(anno_file_path)
            img_file_path = os.path.join(self.img_dir, img_filename)
            img_file_paths.append(img_file_path)

            # ファイル数とオブジェクト数を集計
            for i, name in enumerate(self.classes):
                obj_flags = (labels == i)
                non_difficult_obj_flags = np.logical_and(obj_flags, np.logical_not(difficult_flags))
                num_objs = np.count_nonzero(obj_flags, axis = 0)
                num_non_difficult_objs = np.count_nonzero(non_difficult_obj_flags, axis = 0)
                
                obj_count[i] += num_objs
                non_difficult_obj_count[i] += num_non_difficult_objs
                if (num_objs != 0):
                    file_count[i] += 1
                if (num_non_difficult_objs != 0):
                    non_difficult_file_count[i] += 1
        print("number of selected annotation files: %d" % len(anno_file_paths))

        # 集計結果を表示
        print("non-d: non-difficult")
        count_df = pd.DataFrame(
            {
                'class': self.classes,
                '# files' : file_count,
                '# non-d files': non_difficult_file_count,
                '# objects': obj_count,
                '# non-d objects': non_difficult_obj_count
            }
        )
        pd.set_option('display.max_rows', None)
        print(count_df)
        
        return anno_file_paths, img_file_paths

    def _get_image(self, i):
        # 画像を取得
        img = cv2.imread(self.img_file_paths[i], cv2.IMREAD_COLOR)

        return img

    def _get_annotations(self, i):
        return self._preprocess_xml(self.anno_file_paths[i])[1:]

    def _preprocess_xml(self, anno_file_path):
        tree = ET.parse(anno_file_path)
        root = tree.getroot()

        # 画像ファイル名を取得
        img_filename = root.find('filename').text

        # オブジェクトを取得
        bboxes = []
        labels = []
        difficult_flags = []
        for obj in root.findall('object'):
            # non-difficultなオブジェクトのみを取得する場合
            if ((self.use_difficult is False) and (int(obj.find('difficult').text) == 1)):
                continue

            # クラス名を小文字変換して空白削除
            name = obj.find('name').text.lower().strip()
            # 対象クラスであるかを検証
            if (not (name in self.classes)):
                continue
            labels.append(self.classes.index(name))

            bbox = obj.find('bndbox')
            # The top-left pixel in the image has coordinates (1,1)
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text) - 1
            ymax = int(bbox.find('ymax').text) - 1
            bboxes.append([ymin, xmin, ymax, xmax])

            # difficultフラグを取得
            difficult_flags.append(int(obj.find('difficult').text))

        # bboxが1つ以上ある場合はndarrayに変換
        if bboxes:
            bboxes = np.stack(bboxes).astype(np.float32)
            labels = np.stack(labels).astype(np.int32)
            difficult_flags = np.array(difficult_flags, dtype=np.bool)
        else:
            bboxes = np.empty(0)
            labels = np.empty(0)
            difficult_flags = np.empty(0)

        return img_filename, bboxes, labels, difficult_flags
