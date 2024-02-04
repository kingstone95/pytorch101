import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 定义线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)


def main():
    # 创建模型、损失函数和优化器
    model = LinearModel()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 4 + 2 * x

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    device = torch.device("cuda")
    model = model.to(device)

    for epoch in range(1000):
        # 前向传播
        y_pred = model(x_tensor.to(device))

        # 计算损失
        loss = criterion(y_pred, y_tensor.to(device))

        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        print(model.linear.weight.data, "---", model.linear.bias.data)


if __name__ == '__main__':
    main()
