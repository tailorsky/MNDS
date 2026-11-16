import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lle_core import A, V0, NAMES, TITLES, METHODS, RK4, scan_2d
from plot2d import make_norm, draw

RES = 60
H = 0.01
T_SKIP = 1000.0
CT = 4000.0
S = A[0]

I_PAR, I_RANGE = 1, (0.0, 6.0)
J_PAR, J_RANGE = 2, (0.0, 5.0)
OUT = os.path.join("plots", "task3")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    vi = np.linspace(*I_RANGE, RES)
    vj = np.linspace(*J_RANGE, RES)
    cache = os.path.join(OUT, "data.npz")
    maps = {}
    if "--replot" in sys.argv and os.path.exists(cache):
        d = np.load(cache)
        maps = {m: d[NAMES[m]] for m in range(5)}
    for m in [RK4, 0, 1, 2, 3]:
        if m in maps:
            continue
        maps[m] = scan_2d(METHODS[m], I_PAR, J_PAR, vi, vj, V0, H, T_SKIP, CT, A, S)
        print(f"{NAMES[m]}: разошлось {np.isnan(maps[m]).sum()} из {RES * RES}")
    np.savez(cache, **{NAMES[m]: maps[m] for m in range(5)})
    norm = make_norm(list(maps.values()))
    for m in range(5):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = draw(ax, maps[m], (*I_RANGE, *J_RANGE), norm)
        fig.colorbar(im, ax=ax, label="LLE")
        ax.plot(A[I_PAR], A[J_PAR], "k+", ms=12, mew=2)
        ax.set_xlabel(f"a{I_PAR}")
        ax.set_ylabel(f"a{J_PAR}")
        ax.set_title(f"Bouali: LLE, {TITLES[m]} (h = {H}, {RES}x{RES})\n"
                     "красный — хаос (LLE > 0), белый — цикл (LLE ≈ 0), синий — устойчивая точка (LLE < 0)",
                     fontsize=10)
        fig.tight_layout()
        path = os.path.join(OUT, f"LLE_2d_{NAMES[m]}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("  сохранено:", path)
    print(f"Готово за {time.time() - t0:.0f} с")