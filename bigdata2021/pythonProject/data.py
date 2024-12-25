from abc import ABC

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
from os import path
import numpy as np
import matplotlib.pyplot as plot

base_dir = path.join('.', 'kaggle_eye')


class EyeData(Dataset, ABC):
    def __init__(self, split='train'):
        self.split = split

        # 基本信息
        self.x = pickle.load(open(path.join(base_dir, split + '_x.pkl'), 'rb'))

        # 图像特征
        self.img_l = pickle.load(open(path.join(base_dir, split + '_left_fundus_images.pkl'), 'rb'))
        self.img_r = pickle.load(open(path.join(base_dir, split + '_right_fundus_images.pkl'), 'rb'))

        # 图像似乎是处理过的，因此这个变换器实际上用不着
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # 文本特征
        self.diag = pickle.load(open(path.join('feats', split + '_feats.pkl'), 'rb'))

        # 标签
        if self.split in ['train', 'dev']:
            self.y = pickle.load(open(path.join(base_dir, split + '_y.pkl'), 'rb'))
        else:
            self.y = None

    def __getitem__(self, index):
        if self.split in ['train', 'dev']:
            return self.x[index], self.img_l[index], self.img_r[index], self.diag[index], self.y[index]
        else:
            return self.x[index], self.img_l[index], self.img_r[index], self.diag[index]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    train_data = EyeData('dev')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    print(len(train_loader))

    # for x, il, ir, dl, dr, y in train_loader:
    for x,  il, ir, d, y in train_loader:
        print(x.shape)
        print(il.shape, ir.shape)
        print(d.shape)
        print(y)
        print(x.dtype)
        print(d)
        print(y.dtype)

        img = il[0]
        plot.figure()
        plot.imshow(np.transpose(img, [1, 2, 0]))
        plot.show()
        break

    # base_dir = path.join('.', 'kaggle_eye')
    # # fname = 'train_left_diag.pkl'
    # fname = 'train_y.pkl'
    # # fname = 'train_left_fundus_images.pkl'
    # # fname = 'train_x.pkl'
    # # fname = 'train_x.pkl'
    # # fname = 'train_x.pkl'
    # with open(path.join(base_dir, fname), 'rb') as f:
    #     ld = pickle.load(f)
    # # print(len(ld[0]))
    # print(len(ld))
    # for i, item in enumerate(ld):
    #     print(i, item)
