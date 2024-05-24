import glob
import numpy as np
import torch
from einops import rearrange
import torch_dct as dct

DATA_PATH = "/data1/share_data/purdue/patch/32x32/"


# 扫描所有pkl文件
def file_scanf():
    return np.array(glob.glob(DATA_PATH + "/*.pkl"))


def replace_patch(x, ratio=0.5, method="dct"):
    b, t, h, w = x.shape
    N = int(ratio * t)

    indices = torch.randperm(t)[:N]

    patches = dct.dct_2d(x[:, indices])
    # normlize patches on h and w
    patches = time_norm(patches, dim=1)

    x[:, indices] = patches
    return x


def to_patch(x, replace=False):
    # b 512 32 32
    b, t, h, w = x.shape
    x = time_norm(x, dim=1)
    groups = 64
    # b 512 32 32 -> b 64 8 32 32
    x = x.reshape(b, groups, t // groups, h, w)
    # b 64 8 32 32 -> b 64 1 32 32 -> b 64 32 32
    x = x.mean(dim=2).squeeze()

    if replace:
        x = replace_patch(x, ratio=0.1, method="dct")
    # b 64 32 32 -> b 256 256 -> b 1 256 256
    x = rearrange(x, "b (th tw) h w -> b (th h) (tw w)", th=8, tw=8).unsqueeze(1)
    # x = x.repeat(1, 3, 1, 1)
    return x


def time_norm(x, dim=0, eps=1e-6):
    # normalize x in 2868
    x = x - x.mean(dim=dim, keepdim=True)
    x = x / (x.std(dim=dim, keepdim=True) + eps)
    return x


if __name__ == "__main__":
    x = torch.randn(2, 512, 32, 32)
    x = replace_patch(x, method="zero")
    print(x.shape)
