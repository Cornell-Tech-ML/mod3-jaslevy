import random # noqa: F401
from collections import defaultdict # noqa: F401
import minitorch
import time
import sys # noqa: F401
import numpy as np

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: minitorch.TensorBackend, size: int = 16) -> None:
    """Perform a matrix multiplication operation using the specified backend and matrix size.

    This function creates two random square matrices of the specified size, 
    performs matrix multiplication using the provided backend, and discards 
    the result. It is useful for benchmarking or testing backend performance 
    for matrix operations.

    Args:
    ----
        backend (minitorch.TensorBackend): The tensor backend to perform the 
            matrix multiplication (e.g., CPU, GPU, or custom backend).
        size (int): The size of the square matrices to multiply. Defaults to 16.

    Returns:
    -------
        None: The function does not return any value.

    """
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    _ = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")