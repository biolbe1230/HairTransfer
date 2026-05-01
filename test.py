import torch

if torch.cuda.is_available():
    # 获取可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个可用的 CUDA 设备。")
    print("---------------------------------")
    # 循环打印每个 GPU 的名称
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("未检测到可用的 CUDA 设备。")