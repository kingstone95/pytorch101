# 使用多卡进行训练时时, 容器环境下注意设置共享内存， 否则可能会出现
# `RuntimeError: NCCL Error 2: unhandled system error`
# 解决方式：https://github.com/pytorch/pytorch/issues/73775

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torchvision.transforms import transforms

BATCH_SIZE = 256 * 2
NUM_EPOCHS = 5

def main():
    assert torch.cuda.is_available(), "cuda is not available"

    device = torch.device("cuda")

    # 加载模型
    model = torchvision.models.resnet18(num_classes=10)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)


    # 加载数据集
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = torchvision.datasets.CIFAR10('/dataset/data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


    # 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    total_step = len(train_loader)

    # 开始训练
    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            inputs, targets = images.to(device), labels.to(device)
            inputs= torch.squeeze(inputs)
            # 正向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播， 用计算的梯度做优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, NUM_EPOCHS, i + 1, total_step,
                                                                     loss.item()))


    # 保存模型
    # torch.save(model.state_dict(), '/model/cifar10.pth')


if __name__ == '__main__':
    main()
    print('done')