import subprocess
import numpy as np
import torch
from datetime import datetime
import os
import argparse
import random
from datas.datasets import EEGDataset
from models.resnet import resnet


def select_dataset(paths):
    return EEGDataset(paths)


def select_model():
    model = resnet(attn_name="se", act_name="elu", blocks=[2, 2, 2, 2])
    # model = resnet(blocks=[2, 2, 2, 2])
    return model


def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(1, label.view(-1, 1), 1)


def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--N", type=int, default=1, help="N")
    parser.add_argument("--model", type=str, default="resnet", help="model name")
    args = parser.parse_args()
    args.pid = get_pid()
    return args


def get_log_dir():
    log_dir = "./log_visual/" + getNow() + "/"
    return log_dir + get_pid() + "/"


def get_pid():
    pid = str(os.getpid())[-3:]
    return pid


def getNow():
    now = datetime.now()
    current_year = now.year % 100
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    current_minute = now.minute
    return (
        str(current_year).zfill(2)
        + "/"
        + str(current_month).zfill(2)
        + "/"
        + str(current_day).zfill(2)
        + "/"
        + str(current_hour).zfill(2)
        + str(current_minute).zfill(2)
    )


def get_device(gpus):
    device = torch.device(
        f"cuda:{get_gpu_usage(gpus)}" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)
    return device


def get_gpu_usage(gpus):
    """Returns a list of dictionaries containing GPU usage information."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
        ],
        encoding="utf-8",
    )
    lines = output.strip().split("\n")
    memory = np.array([line.strip().split(",") for line in lines], dtype=int)

    # 计算内存使用百分比
    memory_used_percentage = ((memory[:, 0] / memory[:, 1]) * 100).astype(int)

    # 更新不在 only_use 中的 GPU 的使用率为 100%

    if gpus is not None:
        mask = np.ones(len(memory_used_percentage), dtype=bool)
        mask[gpus] = False
        memory_used_percentage[mask] = 100

    print(memory_used_percentage)
    # 返回最小内存使用率的 GPU 索引
    return np.argmin(memory_used_percentage)
