#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## @file        train_classification.py
## @brief       Trainer Class for Classification CNN Training
## @author      Keitetsu
## @date        2020/05/22
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import os

import chainer
import chainer.links as L
from chainer.training import extensions

import chainerlib


class TrainClassificationTask:

    def __init__(
        self,
        classes,
        train_dataset, test_dataset,
        gpu,
        model,
        batch_size, n_epoch, lr, momentum, weight_decay,
        snapshot_interval, print_interval, output_dir
    ):
        # ラベルを読込み
        self.classes = classes

        # データセットを読込み
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # モデルを読込み
        self.model = model

        # GPUを使用する場合は，モデルをGPUにコピーする
        self.gpu = gpu
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()

        # Optimizerのセットアップ
        print("setting optimizer...: lr=%e, momentum=%e" % (lr, momentum))
        self.optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=momentum)
        # 更新時のcleargrads()の使用を有効にする．v2.0以降非推奨
        self.optimizer.use_cleargrads
        self.optimizer.setup(self.model)
        if weight_decay:
            print("setting optimizer...: weight_decay=%e" % (weight_decay))
            self.optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

        # イテレーションの設定
        print("setting iterator, updater and trainer...")
        self.train_iter = chainer.iterators.MultiprocessIterator(
            self.train_dataset,
            batch_size,
            repeat=True,
            shuffle=True
        )
        self.test_iter = chainer.iterators.SerialIterator(
            self.test_dataset,
            batch_size,
            repeat=False,
            shuffle=False
        )
        self.updater = chainer.training.StandardUpdater(self.train_iter, self.optimizer, device=self.gpu)
        self.output_dir = output_dir
        self.trainer = chainer.training.Trainer(self.updater, (n_epoch, 'epoch'), out=self.output_dir)

        # 検証用データセットで評価する
        self.trainer.extend(extensions.Evaluator(self.test_iter, self.model, device=self.gpu))
        # 学習途中でスナップショットを取得する
        self.trainer.extend(
            extensions.snapshot(filename='snapshot_iter_{.updater.epoch}'),
            trigger=(snapshot_interval, 'epoch')
        )
        # 学習途中でモデルのスナップショットを取得する
        self.trainer.extend(
            extensions.snapshot_object(
                self.model,
                filename='snapshot_model_{.updater.epoch}',
                savefun=chainer.serializers.save_hdf5
            ),
            trigger=(snapshot_interval, 'epoch')
        )
        # グラフを取得する
        self.trainer.extend(
            extensions.PlotReport(
                [
                    'main/loss',
                    'validation/main/loss'
                ],
                x_key='epoch',
                file_name='loss.png',
                marker=""
            )
        )
        self.trainer.extend(
            extensions.PlotReport(
                [
                    'main/accuracy',
                    'validation/main/accuracy'
                ],
                x_key='epoch',
                file_name='accuracy.png',
                marker=""
            )
        )
        # ログを取得する
        self.trainer.extend(extensions.LogReport())
        # 学習と検証の状況を表示する
        self.trainer.extend(
            extensions.PrintReport(
                [
                    'epoch',
                    'main/loss',
                    'validation/main/loss',
                    'main/accuracy',
                    'validation/main/accuracy',
                    'elapsed_time'
                ]
            ),
            trigger=(print_interval, 'epoch')
        )
        # プログレスバーを表示する
        self.trainer.extend(extensions.ProgressBar())

    def run(self):
        print("starting training...")
        self.trainer.run()

        print("saving model...")
        model_file_path = os.path.join(self.output_dir, 'net.model')
        chainer.serializers.save_hdf5(model_file_path, model)
        
        print("training is complete")


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
    # Pascal VOCデータセットのデータをクラス分類用のデータに変換し，データセットの水増しを行う
    transform_dataset = chainer.datasets.TransformDataset(
        dataset,
        chainerlib.transforms.TransformClassification(
            len(classes),
            (100, 100),
            random_crop=True,
            random_crop_rate=[0.2, 0.2],
            debug=False
        )
    )

    # データセットを学習用と検証用に分割する
    # 80%を学習用とする
    dataset_split_rate = int(len(transform_dataset) * 0.8)
    train_dataset, test_dataset = chainer.datasets.split_dataset(
        transform_dataset,
        dataset_split_rate
    )

    # モデルを読込み
    # * ALexNetを使用する
    # * L.Classifierでは予測した値とラベルとの誤差を計算する．
    #   デフォルトではsoftmax_cross_entropy
    print("loading model...")
    model = L.Classifier(chainerlib.nets.AlexNet(len(classes)))

    train_task = TrainClassificationTask(
        classes=classes,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        gpu=0,
        model=model,
        batch_size=20,
        n_epoch=5,
        lr=0.0001,
        momentum=0.95,
        weight_decay=0.0005,
        snapshot_interval=100,
        print_interval=1,
        output_dir='./logs'
    )

    train_task.run()
