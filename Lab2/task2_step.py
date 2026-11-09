import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from main import A, V0, NAMES, TITLES, METHODS, scan_step, flatten, cloud

H_MIN, H_MAX = 0.0001, 0.5
RES = 1000
T = 5000.0
T_SKIP = 1000.0
MAX_PEAKS = 3000
S = A[0]

VARS = ["x", "y", "z"]
OUT = os.path.join("plots", "task2")

HS = (H_MAX / H_MIN) ** (np.arange(RES) / (RES - 1)) * H_MIN

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for m in range(5):
        order = np.random.default_rng(0).permutation(RES)
        buf, cnt = scan_step(METHODS[m], HS[order], V0, T, T_SKIP, A, S, MAX_PEAKS)
        inv = np.argsort(order)
        buf, cnt = buf[inv], cnt[inv]
        bad = cnt[:, 0] < 0
        print(f"{NAMES[m]}: разошлось {bad.sum()} из {RES}, первый расходящийся шаг "
              f"{HS[bad][0] if bad.any() else '-'}")
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        for c, (ax, var) in enumerate(zip(axes, VARS)):
            p, v = flatten(HS, buf, cnt, c)
            cloud(ax, p, v, "tab:blue")
            if len(v):
                lo, hi = np.percentile(v, [0.02, 99.98])
                pad = 0.05 * (hi - lo + 1e-9)
                ax.set_ylim(lo - pad, hi + pad)
            if bad.any():
                ax.axvspan(HS[bad][0], H_MAX, color="0.85", zorder=0)
            ax.set_xscale("log")
            ax.set_xlim(H_MIN, H_MAX)
            ax.set_ylabel(f"Максимумы {var}")
            ax.grid(alpha=0.3, which="both")
        axes[-1].set_xlabel("Шаг h")
        fig.suptitle(f"Bouali: {TITLES[m]}, зависимость от шага (серым — решение разошлось)")
        fig.tight_layout()
        path = os.path.join(OUT, f"step_{NAMES[m]}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("  сохранено:", path)
    print(f"Готово за {time.time() - t0:.0f} с")