import psutil
import torch

class SystemUtil:
    @staticmethod
    def get_ram_usage():
        return psutil.virtual_memory().percent

    @staticmethod
    def get_gpu_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        else:
            return 0
