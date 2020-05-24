#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## @file        eval_classification.py
## @brief       Evaluation Class for Classification CNN
## @author      Keitetsu
## @date        2020/05/24
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import chainerlib

import numpy as np


class EvalClassificationTask:

    def __init__(
        self,
        classes,
        dataset,
        gpu,
        model,
        output_dir
    ):
        # ラベルを読込み
        self.classes = classes

        # データセットを読込み
        self.dataset = dataset

        # モデルを読込み
        self.model = model

        # GPUを使用する場合は，モデルをGPUにコピーする
        self.gpu = gpu
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()

    def run(self):
        n_tests = len(self.dataset)
        print("number of test data: %d" % (n_tests))
        
        n_classes = len(self.classes)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
        n_acc = 0

        print("starting evaluation...")
        for i in range(0, n_tests):
            # 1つのアノテーションデータに対して推論
            label, result_label = self.eval_example(i)

            # 推論結果をconfusion_matrixに反映
            confusion_matrix[label, result_label] += 1

            # 正解数をカウント
            if label == result_label:
                n_acc += 1
        
        # 結果を表示
        print("confusion matrix")
        print(confusion_matrix)
        print("# corrests: %d" % (n_acc))
        print("accuracy = %f" % (float(n_acc) / n_tests))
        print("evaluation is complete")

    def eval_example(self, i):
        # データセットからデータを取得
        img, label = self.dataset[i]
        img = chainer.Variable(img[None, ...])

        # GPUを使用する場合は，画像をGPUにコピーする
        if self.gpu >= 0:
            img.to_gpu()

        # 推論を実行
        x = self.model.predictor(img)
        result = F.argmax(x)

        if self.gpu >= 0:
            result.to_cpu()
        result_label = result.data

        return label, result_label


if __name__ == '__main__':
    # ラベルファイルを読込み
    classes = chainerlib.datasets.load_label_file('./label.txt')

    # データセットを読込み
    dataset_root_dir = '../dataset/VOCdevkit/VOC2012'
    anno_dir = os.path.join(dataset_root_dir, 'Annotations/')
    img_dir = os.path.join(dataset_root_dir, 'JPEGImages/')
    id_list_file_path = os.path.join(dataset_root_dir, 'ImageSets/Main/trainval.txt')
    dataset = chainerlib.datasets.VOCFormatDetectionDataset(
        anno_dir=anno_dir,
        img_dir=img_dir,
        classes=classes,
        id_list_file_path=id_list_file_path,
        use_difficult=False,
        return_difficult=False
    )

    # TransformClassificationクラスを使用して，
    # Pascal VOCデータセットのデータをクラス分類用のデータに変換する．
    # リサイズのみ行い，データの水増しは行わない
    transform_dataset = chainer.datasets.TransformDataset(
        dataset,
        chainerlib.transforms.TransformClassification(
            len(classes),
            (100, 100),
            random_crop=False,
            debug=False
        )
    )

    # データセットを学習用と検証用に分割する
    # 80%を学習用とする
    dataset_split_rate = int(len(transform_dataset) * 0.8)
    _, test_dataset = chainer.datasets.split_dataset(
        transform_dataset,
        dataset_split_rate
    )

    # モデルを読込み
    print("loading model...")
#    chainer.config.train = False
    model = L.Classifier(chainerlib.nets.AlexNet(len(classes)))
    chainer.serializers.load_hdf5('./logs/net.model', model)

    # 評価を実行する
    eval_task = EvalClassificationTask(
        classes=classes,
        dataset=test_dataset,
        gpu=0,
        model=model,
        output_dir='./logs'
    )

    eval_task.run()
