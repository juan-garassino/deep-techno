import torch

TORCH_CPU_DEVICE = torch.device("cpu")

if torch.cuda.device_count() > 0:
    TORCH_CUDA_DEVICE = torch.device("cuda")
else:
    TORCH_CUDA_DEVICE = None

USE_CUDA = True


def use_cuda(cuda_bool):
    global USE_CUDA
    USE_CUDA = cuda_bool


def get_device():
    if (not USE_CUDA) or (TORCH_CUDA_DEVICE is None):
        return TORCH_CPU_DEVICE
    return TORCH_CUDA_DEVICE


def cuda_device():
    return TORCH_CUDA_DEVICE


def cpu_device():
    return TORCH_CPU_DEVICE
