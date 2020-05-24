
# -*- coding: utf-8 -*-

##
## @file        alexnet.py
## @brief       AlexNet
## @author      Keitetsu
## @date        2020/05/24
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import chainer
import chainer.functions as F
import chainer.links as L


class AlexNet(chainer.Chain):
    def __init__(self, n_class):
        super(AlexNet, self).__init__(
            conv1=L.Convolution2D(None, 96, 11, stride=2),
            conv2=L.Convolution2D(None, 256, 5, pad=2),
            conv3=L.Convolution2D(None, 384, 3, pad=1),
            conv4=L.Convolution2D(None, 384, 3, pad=1),
            conv5=L.Convolution2D(None, 256, 3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, n_class)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2
        )
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2
        )
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        return h
