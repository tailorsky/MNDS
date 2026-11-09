import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from main import A, V0, NAMES, TITLES, METHODS, RK4, scan_param, flatten, cloud

H = 0.01
T = 5000.0
T_SKIP = 1000.0
N_VALUES = 1000
MAX_PEAKS = 3000
S = A[0]

PARAMS = {
    1: (0.0, 6.0),
    2: (0.0, 5.0),
    3: (0.0, 3.0),
    4: (0.0, 0.005),
}
VARS = ["x", "y", "z"]
OUT = os.path.join("plots", "task1")


def run(m):
    n, skip = int(T / H), int(T_SKIP / H)
    res = {}
    for which, (lo, hi) in PARAMS.items():
        values = np.linspace(lo, hi, N_VALUES)
        buf, cnt = scan_param(METHODS[m], which, values, V0, H, n, skip, A, S, MAX_PEAKS)
        res[which] = [tuple(np.asarray(q, dtype=np.float32) for q in flatten(values, buf, cnt, c))
                      for c in range(3)]
        print(f"  {NAMES[m]}, a{which}: разошлось {np.count_nonzero(cnt[:, 0] < 0)} из {N_VALUES}")
    return res


def plot(m, res, ref):
    for c, var in enumerate(VARS):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, which in zip(axes.flat, PARAMS):
            cloud(ax, *ref[which][c], "tab:red", "РК4")
            cloud(ax, *res[which][c], "tab:blue", TITLES[m], alpha=0.5)
            ax.axvline(A[which], color="k", lw=0.8, ls="--")
            both = np.concatenate([ref[which][c][1], res[which][c][1]])
            if len(both):
                lo, hi = np.percentile(both, [0.02, 99.98])
                pad = 0.05 * (hi - lo + 1e-9)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(*PARAMS[which])
            ax.set_xlabel(f"a{which}")
            ax.set_ylabel(f"Максимумы {var}")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(loc="upper right", markerscale=15)
        fig.suptitle(f"Bouali: {TITLES[m]} и РК4, максимумы {var} (h = {H})")
        fig.tight_layout()
        path = os.path.join(OUT, f"{NAMES[m]}_{var}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("  сохранено:", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print("РК4...")
    ref = run(RK4)
    for m in range(4):
        print(f"{NAMES[m]}...")
        plot(m, run(m), ref)
    print(f"Готово за {time.time() - t0:.0f} с")