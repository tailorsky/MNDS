import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from main import A, V0, NAMES, TITLES, METHODS, RK4, scan_2d

RES = 100
H = 0.01
T = 8000.0
T_SKIP = 2000.0
MAX_PEAKS = 8000
TOL = 1e-3
S = A[0]

I_PAR, I_RANGE = 1, (0.0, 6.0)
J_PAR, J_RANGE = 2, (0.0, 5.0)

VARS = ["x", "y", "z"]
OUT = os.path.join("plots", "task4")


def plot(m, res, vmax):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.8")
    for c, (ax, var) in enumerate(zip(axes, VARS)):
        data = np.ma.masked_less(res[c], 0)
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap,
                       extent=(*I_RANGE, *J_RANGE), interpolation="nearest",
                       vmin=0, vmax=vmax[c])
        fig.colorbar(im, ax=ax, extend="max", label="Число кластеров")
        ax.plot(A[I_PAR], A[J_PAR], "r+", ms=12, mew=2)
        ax.set_xlabel(f"a{I_PAR}")
        ax.set_ylabel(f"a{J_PAR}")
        ax.set_title(f"по максимумам {var}")
    fig.suptitle(f"Bouali: {TITLES[m]}, число кластеров максимумов "
                 f"(h = {H}, {RES}x{RES}; серым — разошлось, + — номинальные параметры)")
    fig.tight_layout()
    path = os.path.join(OUT, f"map_{NAMES[m]}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("  сохранено:", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    vi = np.linspace(*I_RANGE, RES)
    vj = np.linspace(*J_RANGE, RES)
    n, skip = int(T / H), int(T_SKIP / H)
    maps = {}
    for m in [RK4, 0, 1, 2, 3]:
        maps[m] = scan_2d(METHODS[m], I_PAR, J_PAR, vi, vj, V0, H, n, skip, A, S, MAX_PEAKS, TOL)
        print(f"{NAMES[m]}: разошлось {np.count_nonzero(maps[m][0] < 0)} из {RES * RES}")
    vmax = [max(1, int(np.percentile(maps[RK4][c][maps[RK4][c] >= 0], 95))) for c in range(3)]
    for m in range(5):
        plot(m, maps[m], vmax)
    for m in range(4):
        same = np.mean(maps[m][2] == maps[RK4][2])
        print(f"{NAMES[m]}: совпадение карты z с РК4 {100 * same:.0f}%")
    print(f"Готово за {time.time() - t0:.0f} с")