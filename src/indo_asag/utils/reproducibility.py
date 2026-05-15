"""Reproducibility utilities: seed setting and environment capture."""

import os
import random
import platform
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_environment_info() -> dict:
    """Capture environment info for experiment logging.
    
    Returns:
        Dictionary with Python version, OS, GPU info, and package versions.
    """
    info = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "numpy": np.__version__,
    }
    
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = "not installed"
    
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    
    return info
