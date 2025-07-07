# -*- coding: utf-8 -*-
# @Author  : XUNWJ
# @Contact : ssssustar@163.com
# @File    : test_accelerate.py
# @Time    : 2024/8/26 19:09
# @Desc    :
from __future__ import print_function
from __future__ import division

from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 自定义数据集
class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 10)
        self.targets = torch.randn(100, 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# 初始化 Accelerator
accelerator = Accelerator()

# 定义模型、优化器和损失函数
model = nn.Linear(10, 10)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 数据加载
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用 Accelerator 准备模型、优化器和数据加载器
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# 训练循环
for epoch in range(10):
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)  # 使用 Accelerator 进行反向传播
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')