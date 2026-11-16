import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core4 import A, V0, NAMES, TITLES, METHODS, RK4, scan_param

H = 0.01
T_SKIP = 1000.0
T_AVG = 1000.0
N_VALUES = 200
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
        res[which] = (values, scan_param(METHODS[m], which, values, V0, H, T_SKIP, T_AVG, A, S))
        print(f"  {NAMES[m]}, a{which}: разошлось {np.isnan(res[which][1][:, 0]).sum()} из {N_VALUES}")
    return res


def plot(m, res, ref, col, ylabel, fname, title, exact=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, which in zip(axes.flat, PARAMS):
        vals, r = ref[which]
        if exact:
            ax.plot(vals, r[:, 1], color="k", lw=1, ls="--", label="точно: среднее div f")
        ax.plot(vals, r[:, col], color="tab:red", lw=1.2, label="РК4")
        ax.plot(res[which][0], res[which][1][:, col], color="tab:blue", lw=1.0, alpha=0.85, label=TITLES[m])
        ax.axvline(A[which], color="k", lw=0.8, ls=":")
        if exact:
            ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlim(*PARAMS[which])
        ax.set_xlabel(f"a{which}")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle(f"Bouali: {title}, {TITLES[m]} и РК4 (h = {H})")
    fig.tight_layout()
    path = os.path.join(OUT, fname)
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
        res = run(m)
        # скорость изменения фазового объёма пока не выводим
        # plot(m, res, ref, 0, "скорость изм. фаз. объёма, 1/ед. времени",
        # f"PV_{NAMES[m]}.png", "скорость изменения фазового объёма", exact=True)
        plot(m, res, ref, 2, "фазовый объём V = Δx·Δy·Δz",
             f"PV_box_{NAMES[m]}.png", "фазовый объём (объём аттрактора)")
    print(f"Готово за {time.time() - t0:.0f} с")