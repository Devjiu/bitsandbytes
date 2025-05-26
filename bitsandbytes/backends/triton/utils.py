import math
import statistics

import torch

from bitsandbytes.functional import get_4bit_type

# Should be the same for quant/dequant
_FP4_QUANT_TABLE = get_4bit_type("fp4", device="xpu")
_NF4_QUANT_TABLE = get_4bit_type("nf4", device="xpu")


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


def measure_gpu(fn, *args):
    di = torch.xpu
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    cache_size = 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device="xpu")

    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn(*args)
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # compute number of warmup and repeat
    n_warmup = max(1, int(25 / estimate_ms))
    n_repeat = max(1, int(100 / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn(*args)
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn(*args)
        end_event[i].record()
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    times_med = _summarize_statistics(times, None, "median")
    return times_med


# # Should be sorted to use binary search
# _NF4_QUANT_TABLE = torch.tensor(
#     [
#         -1.0,
#         -0.6961928009986877,
#         -0.5250730514526367,
#         -0.39491748809814453,
#         -0.28444138169288635,
#         -0.18477343022823334,
#         -0.09105003625154495,
#         0.0,
#         0.07958029955625534,
#         0.16093020141124725,
#         0.24611230194568634,
#         0.33791524171829224,
#         0.44070982933044434,
#         0.5626170039176941,
#         0.7229568362236023,
#         1.0,
#     ],
#     dtype=torch.float32,
#     device="xpu",
# )

# # Should be sorted to use binary search
# _FP4_QUANT_TABLE = torch.tensor(
#     [
#         -1.0000,
#         -0.6667,
#         -0.5000,
#         -0.3333,
#         -0.2500,
#         -0.1667,
#         -0.0052,
#         -0.0000,
#         0.0000,
#         0.0052,
#         0.1667,
#         0.2500,
#         0.3333,
#         0.5000,
#         0.6667,
#         1.0000,
#     ],
#     dtype=torch.float32,
#     device="xpu",
# )
