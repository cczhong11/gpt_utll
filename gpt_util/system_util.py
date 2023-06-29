import psutil
import torch


class SystemUtil:
    @staticmethod
    def get_ram_usage():
        return f"{psutil.virtual_memory().used} / {psutil.virtual_memory().total}"

    @staticmethod
    def get_gpu_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        else:
            return 0
