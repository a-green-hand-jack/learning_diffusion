import torch
import numpy as np
import random

def set_global_random_seed(seed):
    """
    设置全局随机种子以确保可重复性。

    参数:
        seed (int): 种子值。
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True