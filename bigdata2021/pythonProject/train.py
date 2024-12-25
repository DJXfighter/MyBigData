from data import EyeData
from torch.utils.data import DataLoader
from model import Net

import os
from os import path

import torch
import torch.nn as nn
import torch.optim as optim

from utils import plot

# 超参数
num_epochs = 50
batch_size = 128
md = False

# 训练集
train_set = EyeData('train')
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
# 验证集
valid_set = EyeData('dev')
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4)

# 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net(md=md).to(device)

# 损失函数
criterion = nn.BCELoss()

# 优化器
optimizer = optim.Adam(net.parameters(), 0.0001, weight_decay=1e-4)

def loss_fn(logits, targets):
    loss1 = criterion(logits[:, 0], targets[:, 0])
    loss2 = criterion(logits[:, 1], targets[:, 1])
    loss3 = criterion(logits[:, 2], targets[:, 2])
    loss = loss1 + loss2 + loss3
    return loss

def train():
    costs, accs = [], []

    for x, il, ir, d, labels in train_loader:
        # 输入，标签
        in1, in2, in3, in4, labels = x.float().to(device), il.float().to(device), ir.float().to(device), d.float().to(device), labels.float().to(device)

        # 预测
        preds = net(in1, in2, in3, in4)

        # 损失
        loss = loss_fn(preds, labels)

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 准确率
        acc = torch.mean(torch.eq((preds >= 0.5).float(), labels).float())

        costs.append(loss)
        accs.append(acc)

        return sum(costs) / len(costs), sum(accs) / len(accs)

def valid():
    net.eval()

    with torch.no_grad():
        cost, accuracy = [], []

        for x, il, ir, d, labels in valid_loader:
            # 输入，标签
            in1, in2, in3, in4, labels = x.float().to(device), il.float().to(device), ir.float().to(device), d.float().to(device), labels.float().to(device)

            # 预测
            preds = net(in1, in2, in3, in4)

            # 损失
            loss = loss_fn(preds, labels)

            # 准确率
            acc = torch.mean(torch.eq((preds >= 0.5).float(), labels).float())

            cost.append(loss)
            accuracy.append(acc)

        return sum(cost) / len(cost), sum(accuracy) / len(accuracy)


if __name__ == '__main__':
    l1, l2, a1, a2 = [], [], [], []

    for epoch in range(num_epochs):
        loss_t, acc_t = train()  # 训练
        loss_v, acc_v = train()  # 验证
        print("Epoch %d, train loss=%.4f, train acc=%.4f" % (epoch + 1, loss_t, acc_t))
        print("\t valid loss=%.4f, valid acc=%.4f" % (loss_v, acc_v))

        # 记录结果
        l1.append(loss_t.item())
        l2.append(loss_v.item())
        a1.append(acc_t.item())
        a2.append(acc_v.item())

    # 保存模型
    if not path.exists('check_points'):
        os.mkdir('check_points')
    if md:
        torch.save(net.state_dict(), path.join('check_points', 'net_md.ckpt'))
    else:
        torch.save(net.state_dict(), path.join('check_points', 'net.ckpt'))

    # 打印结果
    plot(loss1=l1, acc1=a1, loss2=l2, acc2=a2)
