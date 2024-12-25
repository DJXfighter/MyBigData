import pickle
from os import path
import os
import numpy as np
import re

base_dir = path.join('.', 'kaggle_eye')

def make_feats(sp='train'):
    feats = pickle.load(open('cls_dic.pkl', 'rb'))

    dl = pickle.load(open(path.join(base_dir, sp + '_left_diag_ori.pkl'), 'rb'))
    dr = pickle.load(open(path.join(base_dir, sp + '_right_diag_ori.pkl'), 'rb'))

    one_hot_fts = []

    for it1, it2 in zip(dl, dr):
        ftl = np.zeros(len(feats))  # 左眼的诊断特征，【1， 131】
        ftr = np.zeros(len(feats))  # 右眼的诊断特征，【1， 131】

        for feat in it1[1:]:
            if '，' in feat:
                feat = re.sub('，', ',', feat)
            if ',' in feat:
                for f in feat.split(','):
                    if f in feats.keys():
                        ftl[feats[f]] = 1
            else:
                if feat in feats.keys():
                    ftl[feats[feat]] = 1

        for feat in it2[1:]:
            if '，' in feat:
                feat = re.sub('，', ',', feat)
            if ',' in feat:
                for f in feat.split(','):
                    if f in feats.keys():
                        ftr[feats[f]] = 1
            else:
                if feat in feats.keys():
                    ftr[feats[feat]] = 1

        # 左眼的 one-hot 特征和右眼的拼接起来，作为一个患者的诊断特征
        one_hot_fts.append(np.concatenate([ftl, ftr]))

    # 写入
    print(len(one_hot_fts))
    if not path.exists('feats'):
        os.mkdir('feats')
    pickle.dump(one_hot_fts, open(path.join('feats', sp + '_feats.pkl'), 'wb'))


if __name__ == '__main__':
    splits = ['train', 'dev', 'test']

    for split in splits:
        make_feats(split)
