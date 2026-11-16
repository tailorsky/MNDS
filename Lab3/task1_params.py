import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lle_core import A, V0, NAMES, TITLES, METHODS, RK4, scan_param

H = 0.01
T_SKIP = 1000.0
CT = 4000.0
N_VALUES = 500
S = A[0]

PARAMS = {
    1: (0.0, 6.0),
    2: (0.0, 5.0),
    3: (0.0, 3.0),
    4: (0.0, 0.005),
}
OUT = os.path.join("plots", "task1")


def run(m):
    res = {}
    for which, (lo, hi) in PARAMS.items():
        values = np.linspace(lo, hi, N_VALUES)
        res[which] = (values, scan_param(METHODS[m], which, values, V0, H, T_SKIP, CT, A, S))
        print(f"  {NAMES[m]}, a{which}: разошлось {np.isnan(res[which][1]).sum()} из {N_VALUES}")
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print("РК4...")
    ref = run(RK4)
    for m in range(4):
        print(f"{NAMES[m]}...")
        res = run(m)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, which in zip(axes.flat, PARAMS):
            ax.plot(*ref[which], color="tab:red", lw=1.2, label="РК4")
            ax.plot(*res[which], color="tab:blue", lw=1.0, alpha=0.8, label=TITLES[m])
            ax.axhline(0, color="k", lw=0.8)
            ax.axvline(A[which], color="k", lw=0.8, ls="--")
            ax.set_xlim(*PARAMS[which])
            ax.set_xlabel(f"a{which}")
            ax.set_ylabel("LLE")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(loc="upper right")
        fig.suptitle(f"Bouali: старший показатель Ляпунова, {TITLES[m]} и РК4 (h = {H})")
        fig.tight_layout()
        path = os.path.join(OUT, f"LLE_{NAMES[m]}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("  сохранено:", path)
    print(f"Готово за {time.time() - t0:.0f} с")