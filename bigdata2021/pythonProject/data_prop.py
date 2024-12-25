import pickle
import os
from os import path
import re

def creat_feat_dic():
    feats = {}

    # 首先重新处理诊断数据，将有逗号的项拆分
    for item in do_l:
        for feat in item[1:]:
            if '，' in feat:
                feat = re.sub('，', ',', feat)
            if ',' in feat:
                # print(feat)
                for f in feat.split(','):
                    if f not in feats and f != '':
                        feats[f] = len(feats)
            else:
                if feat not in feats and feat != '':
                    feats[feat] = len(feats)

    for item in do_r:
        for feat in item[1:]:
            if '，' in feat:
                feat = re.sub('，', ',', feat)
            if ',' in feat:
                # print(feat)
                for f in feat.split(','):
                    if f not in feats and f != '':
                        feats[f] = len(feats)
            else:
                if feat not in feats and feat != '':
                    feats[feat] = len(feats)

    print(len(feats))
    print(feats)
    # for k, v in feats.items():
    #     print(k, v)

    pickle.dump(feats, open('cls_dic.pkl', 'wb'))

def check_feat():
    feats = pickle.load(open('cls_dic.pkl', 'rb'))

    for item in d_l:
        for feat in item[1:]:
            if '，' in feat:
                feat = re.sub('，', ',', feat)
            if ',' in feat:
                for f in feat.split(','):
                    if f not in feats and f != 'None':
                        print(f)
            else:
                if feat not in feats and feat != 'None':
                    print(feat)

    for item in d_r:
        for feat in item[1:]:
            if '，' in feat:
                feat = re.sub('，', ',', feat)
            if ',' in feat:
                for f in feat.split(','):
                    if f not in feats and f != 'None':
                        print(f)
            else:
                if feat not in feats and feat != 'None':
                    print(feat)


if __name__ == '__main__':
    base_dir = path.join('.', 'kaggle_eye')

    split = 'train'

    do_l = pickle.load(open(path.join(base_dir, split + '_left_diag_ori.pkl'), 'rb'))
    do_r = pickle.load(open(path.join(base_dir, split + '_right_diag_ori.pkl'), 'rb'))
    d_l = pickle.load(open(path.join(base_dir, split + '_left_diag.pkl'), 'rb'))
    d_r = pickle.load(open(path.join(base_dir, split + '_right_diag.pkl'), 'rb'))

    # 创建全部诊断特征的字典
    creat_feat_dic()

    # 检查验证集和测试集中的哪些特征在训练集中并没有出现过
    # check_feat()
