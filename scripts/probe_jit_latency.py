"""Probe: latencia jit-compiled por modo (target runtime de serving)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import time

import numpy as np
import torch
from torch import nn

from camhd.models import ArchConfig, CAMHDNet

torch.set_num_threads(1)


class TeacherPath(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, s):
        return self.net(x, s, mode="teacher")


class StudentPath(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, s):
        return self.net(x, s, mode="student")


def bench(fn, warm=300, meas=5000):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(meas):
            t0 = time.perf_counter_ns()
            fn()
            ts.append((time.perf_counter_ns() - t0) / 1000)
    a = np.asarray(ts)
    return dict(p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), mean=float(a.mean()))


def main():
    torch.manual_seed(0)
    net = CAMHDNet(ArchConfig(w=128, depth=2, student_trunk_frac=0.25, d=16)).eval()
    x = torch.randn(1, 16)
    s = torch.randn(1, 3)
    with torch.no_grad():
        te = torch.jit.trace(lambda x, s: net(x, s, mode="teacher"), (x, s))
        st = torch.jit.trace(lambda x, s: net(x, s, mode="student"), (x, s))
    with torch.no_grad():
        tt = bench(lambda: te(x, s))
        ts = bench(lambda: st(x, s))
    print("teacher jit-traced:", tt)
    print("student jit-traced:", ts)
    evict = torch.empty(4 << 20)

    def alt():
        evict.uniform_()
        te(x, s)
        evict.uniform_()
        st(x, s)

    ta = bench(alt, warm=200, meas=1500)
    print("pair alt:", ta, "kappa_est:", ta["mean"] - (tt["mean"] + ts["mean"]) / 2)


if __name__ == "__main__":
    main()
