from data import EyeData
from torch.utils.data import DataLoader
from model import Net

import torch

import os
from os import path
import pickle


if __name__ == '__main__':
    # 超参数
    md = False

    # 训练集
    test_set = EyeData('test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(md=md).to(device)
    if md:
        net.load_state_dict(torch.load(path.join('check_points', 'net_md.ckpt')))
    else:
        net.load_state_dict(torch.load(path.join('check_points', 'net.ckpt')))

    # 测试
    net.eval()
    with torch.no_grad():
        outputs = []

        for x, il, ir, d in test_loader:
            # 输入，标签
            in1, in2, in3, in4 = x.float().to(device), il.float().to(device), ir.float().to(device), d.float().to(device)

            # 预测
            logits = net(in1, in2, in3, in4).squeeze(0)
            outputs.append((logits >= 0.5).int().cpu().numpy().tolist())

    print(outputs)

    # 保存结果
    save_dir = 'results'
    if not path.exists(save_dir):
        os.mkdir(save_dir)
    if md:
        pickle.dump(outputs, open(path.join(save_dir, 'result_md.pkl'), 'wb'))
    else:
        pickle.dump(outputs, open(path.join(save_dir, 'result.pkl'), 'wb'))