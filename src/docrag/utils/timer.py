import time
import torch


class Timer:
    """
    Context manager for timing model generation. Measures wall-clock time in seconds.
    """

    def __enter__(self):
        self.start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
