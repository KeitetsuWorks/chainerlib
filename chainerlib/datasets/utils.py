# -*- coding: utf-8 -*-

##
## @file        train_utils.py
## @brief       Utilities for CNN Training
## @author      Keitetsu
## @date        2020/05/22
## @copyright   Copyright (c) 2020 Keitetsu
## @par         License
##              This software is released under the MIT License.
##


import pandas as pd


def load_label_file(label_file_path):
    print("loading label file...: %s" % (label_file_path))
    with open(label_file_path, mode='rt') as f:
        classes = f.read().lower().strip().split('\n')
    
    # 読込み結果を表示
    classes_df = pd.DataFrame(
        {
            'class': classes,
        }
    )
    pd.set_option('display.max_rows', None)
    print(classes_df)
    print("number of classes: %d" % (len(classes)))
    
    return classes
