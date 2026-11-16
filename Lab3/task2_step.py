import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lle_core import A, V0, NAMES, TITLES, METHODS, RK4, scan_step, log_steps

H_MIN, H_MAX = 1e-4, 0.1
RES = 200
T_SKIP = 1000.0
CT = 4000.0
S = A[0]

HS = log_steps(H_MIN, H_MAX, RES)
OUT = os.path.join("plots", "task2")


def run(m):
    order = np.random.default_rng(0).permutation(RES)
    out = np.empty(RES)
    out[order] = scan_step(METHODS[m], HS[order], V0, T_SKIP, CT, A, S)
    bad = np.isnan(out)
    print(f"  {NAMES[m]}: разошлось {bad.sum()} из {RES}, первый расходящийся шаг "
          f"{HS[bad][0] if bad.any() else '-'}")
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = {m: run(m) for m in [RK4, 0, 1, 2, 3]}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for m, ax in zip(range(4), axes.flat):
        ax.plot(HS, res[RK4], color="tab:red", lw=1.2, label="РК4")
        ax.plot(HS, res[m], color="tab:blue", lw=1.0, label=TITLES[m])
        bad = np.isnan(res[m])
        if bad.any():
            ax.axvspan(HS[bad][0], H_MAX, color="0.85", zorder=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(H_MIN, H_MAX)
        ax.set_xlabel("Шаг h")
        ax.set_ylabel("LLE")
        ax.set_title(TITLES[m])
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left")
    fig.suptitle("Bouali: зависимость LLE от шага (серым — решение метода разошлось)")
    fig.tight_layout()
    path = os.path.join(OUT, "LLE_step.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("сохранено:", path)
    print(f"Готово за {time.time() - t0:.0f} с")