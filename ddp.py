import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models
import torch.distributed as dist
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
from torch.nn.parallel import DistributedDataParallel as DDP

BATCH_SIZE = 256
NUM_EPOCHS = 5

# departed
def main():
    assert torch.cuda.is_available(), "cuda is not available"

    # modify 1
    torch.distributed.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    # 加载模型
    model = torchvision.models.resnet18(num_classes=10)
    model = model.to(device)
    # modify 2
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 加载数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.CIFAR10('/dataset/data', train=True, transform=transform, download=True)

    # modify 3
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE,
                                               pin_memory=True)

    # 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.01)

    total_step = len(train_loader)

    # 开始训练
    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            inputs, targets = images.to(device), labels.to(device)
            inputs = torch.squeeze(inputs)
            # 正向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播， 用计算的梯度做优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Rank [{}], Local_Rank [{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(rank, local_rank, epoch + 1, NUM_EPOCHS, i + 1, total_step,
                                                                     loss.item()))

    # if rank == 0:
    #     torch.save(model.state_dict(), '/model/cifar10.pth')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
    print('done')
