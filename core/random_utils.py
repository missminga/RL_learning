"""随机种子工具：统一设置 random / numpy / torch / gymnasium。"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int | None) -> int | None:
    """设置全局随机种子，返回最终 seed。"""
    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 尽量保证可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def seed_env(env: Any, seed: int | None) -> None:
    """设置 gymnasium 环境与 action_space 种子。"""
    if seed is None:
        return
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
