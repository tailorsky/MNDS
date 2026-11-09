import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from main import A, V0, METHODS, KD, RK4, scan_s, scan_param, flatten, cloud

H = 0.01
T = 5000.0
T_SKIP = 1000.0
MAX_PEAKS = 3000

RANGES = [
    ("0_1", 0.0, 1.0, 1000),
    ("m100_100", -100.0, 100.0, 2001),
]
VARS = ["x", "y", "z"]
OUT = os.path.join("plots", "task3")


def rk4_band(c):
    buf, cnt = scan_param(METHODS[RK4], 1, np.array([A[1]]), V0, H, int(T / H), int(T_SKIP / H), A, 0.5, MAX_PEAKS)
    v = buf[0, c, :cnt[0, c]]
    return v.min(), v.max()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    n, skip = int(T / H), int(T_SKIP / H)
    results = []
    for tag, lo, hi, k in RANGES:
        svals = np.linspace(lo, hi, k)
        buf, cnt = scan_s(METHODS[KD], svals, V0, H, n, skip, A, MAX_PEAKS)
        bad = cnt[:, 0] < 0
        print(f"s ∈ [{lo}, {hi}]: разошлось {bad.sum()} из {k}")
        results.append((svals, buf, cnt))
    for c, var in enumerate(VARS):
        ref_lo, ref_hi = rk4_band(c)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (tag, lo, hi, k), (svals, buf, cnt) in zip(axes, RANGES, results):
            ax.axhspan(ref_lo, ref_hi, color="tab:red", alpha=0.08)
            for s_bad in svals[cnt[:, 0] < 0]:
                ax.axvspan(s_bad - (hi - lo) / (2 * (k - 1)), s_bad + (hi - lo) / (2 * (k - 1)),
                           color="0.85", lw=0, zorder=0)
            p, v = flatten(svals, buf, cnt, c)
            cloud(ax, p, v, "tab:blue")
            if lo < 0:
                ax.axvspan(0, 1, color="tab:green", alpha=0.25, lw=0)
            ax.set_xlim(lo, hi)
            lo_v, hi_v = np.percentile(v, [0.1, 99.9]) if len(v) else (ref_lo, ref_hi)
            lo_v, hi_v = min(lo_v, ref_lo), max(hi_v, ref_hi)
            pad = 0.08 * (hi_v - lo_v)
            ax.set_ylim(lo_v - pad, hi_v + pad)
            ax.set_xlabel("s")
            ax.set_ylabel(f"Максимумы {var}")
            ax.set_title(f"s ∈ [{lo:g}, {hi:g}]")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Bouali: КД, зависимость от s (h = {H}); серым — разошлось, "
                     f"красная полоса — размах максимумов РК4")
        fig.tight_layout()
        path = os.path.join(OUT, f"KD_s_{var}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("  сохранено:", path)
    print(f"Готово за {time.time() - t0:.0f} с")