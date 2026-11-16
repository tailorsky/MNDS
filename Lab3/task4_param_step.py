import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lle_core import A, V0, NAMES, TITLES, METHODS, RK4, scan_param_step, log_steps
from plot2d import make_norm, draw

RES_P = 30
RES_H = 30
H_MIN, H_MAX = 1e-4, 0.1
T_SKIP = 500.0
CT = 2000.0
S = A[0]

PAR, P_RANGE = 1, (0.0, 6.0)
OUT = os.path.join("plots", "task4")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    values = np.linspace(*P_RANGE, RES_P)
    hs = log_steps(H_MIN, H_MAX, RES_H)
    order = np.random.default_rng(0).permutation(RES_P * RES_H)
    cache = os.path.join(OUT, "data.npz")
    maps = {}
    if "--replot" in sys.argv and os.path.exists(cache):
        d = np.load(cache)
        maps = {m: d[NAMES[m]] for m in range(5)}
    for m in [RK4, 0, 1, 2, 3]:
        if m in maps:
            continue
        maps[m] = scan_param_step(METHODS[m], PAR, values, hs, order, V0, T_SKIP, CT, A, S)
        print(f"{NAMES[m]}: разошлось {np.isnan(maps[m]).sum()} из {RES_P * RES_H} "
              f"({time.time() - t0:.0f} с)")
    np.savez(cache, **{NAMES[m]: maps[m] for m in range(5)})
    norm = make_norm(list(maps.values()))
    for m in range(5):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = draw(ax, maps[m], (*P_RANGE, H_MIN, H_MAX), norm, log_y=True, y=hs)
        fig.colorbar(im, ax=ax, label="LLE")
        ax.set_xlabel(f"a{PAR}")
        ax.set_ylabel("Шаг h")
        ax.set_title(f"Bouali: LLE от a{PAR} и шага, {TITLES[m]}\n"
                     "красный — хаос, белый — цикл, синий — устойчивая точка, серый — разошлось",
                     fontsize=10)
        fig.tight_layout()
        path = os.path.join(OUT, f"LLE_param_step_{NAMES[m]}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("  сохранено:", path)
    print(f"Готово за {time.time() - t0:.0f} с")