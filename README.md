# pytorch101

## 单机多卡

### DP方式

启动方式
```bash
docker run -it --rm \
--shm-size 8g \
-e NVIDIA_VISIBLE_DEVICES=0,1 \
-v /root/wangs/code:/workspace \
-v /root/wangs/dataset:/dataset/data \
registry.cnbita.com:5000/wangshi/pytorch:1.12-python3.10-cuda11.3-devel-ubuntu20.04  \
python dp.py
```


### DDP 方式

启动方式
```bash
docker run -it --rm \
--shm-size 8g \
-e NVIDIA_VISIBLE_DEVICES=0,1 \
-e NCCL_DEBUG=INFO \
-v /root/wangs/code:/workspace \
-v /root/wangs/dataset:/dataset/data \
registry.cnbita.com:5000/wangshi/pytorch:1.12-python3.10-cuda11.4-ubuntu20.04  \
torchrun --nnodes=1 --nproc_per_node=2 ddp.py
或 (deprecated) python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 ddp.py
```

### DP与 DDP对比
1. DP 是pytorch早期实现，采用单进程多线程的方式实现数据并行， 但是因为Python语言有GIL限制，所以这种方式并不高效；
   DDP 采用多进程实现， 每张卡都运行在一个单独的进程内， 每个进程独立计算梯度
2. DP 中有参数服务器的概念，参数服务器所在线程会接受其他线程传回来的梯度与参数，整合后进行参数更新，再将更新后的参数发回给其他线程，这里有一个单对多的双向传输；
   DDP抛弃了参数服务器中一对多的传输与同步问题, 采用环形的梯度传递， Ring-Allreduce, 使得每张GPU只需要和自己上下游的GPU进行进程间的梯度传递
3. DP交换的数据包括 input、model、output、loss、gradient; 
   DDP 只对gradient 少量数据进行交换

NOTE: 
在 DataParallel 中，batch size 设置必须为单卡的 n 倍，因为一个batch的数据会被主GPU分散为minibatch给其他GPU;
但是在 DistributedDataParallel 内，batch size 设置于单卡一样即可，因为各个GPU对应的进程独立从磁盘中加载数据。



## 多机多卡

启动方式: 
node0:
```
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=172.17.0.3 --master_port=29500 ddp.py
或 torchrun

```

node1:
```
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=172.17.0.3 --master_port=29500 ddp.py
```

`torchrun` 可替换为 `python -m torch.distributed.launch`


## 弹性分布式训练

启动方式：
所有node启动命令一致:
```
torchrun --nproc_per_node=1 --nnodes=1:2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=172.17.0.3:29400 ddp.py
```


