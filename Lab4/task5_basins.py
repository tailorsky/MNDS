import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN
from core4 import A, NAMES, TITLES, METHODS, basins

RES = 100
X_RANGE = (-4.0, 4.0)
Y_RANGE = (-4.0, 4.0)
Z0 = 0.1
H = 0.01
T_SKIP = 1000.0
T_WIN = 2000.0
AMP_SCALE = 0.15
PER_SCALE = 600.0
EPS = 0.01
MIN_SAMPLES = 10
S = A[0]
OUT = os.path.join("plots", "task5")


def clusters(amp, per):
    lab = np.full(amp.shape, -2)
    ok = ~np.isnan(amp)
    if ok.sum() == 0:
        return lab, 0
    f = np.column_stack([amp[ok] / AMP_SCALE, per[ok] / PER_SCALE])
    lab[ok] = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(f)
    return lab, lab.max() + 1


def plot(m, amp, per):
    lab, k = clusters(amp, per)
    ext = (*X_RANGE, *Y_RANGE)
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.75")
    im = ax[0].imshow(np.ma.masked_invalid(amp), origin="lower", aspect="auto", extent=ext, cmap=cmap,
                      interpolation="nearest")
    fig.colorbar(im, ax=ax[0], label="средняя амплитуда (среднее значение пиков z)")
    ax[0].set_title("средняя амплитуда")
    im = ax[1].imshow(np.ma.masked_invalid(per), origin="lower", aspect="auto", extent=ext, cmap=cmap,
                      interpolation="nearest")
    fig.colorbar(im, ax=ax[1], label="средний межпиковый интервал")
    ax[1].set_title("средний межпиковый интервал")
    base = plt.get_cmap("tab20")
    colors = ["0.75", "black"] + [base(i % 20) for i in range(max(k, 1))]
    shown = np.where(lab == -2, 0, np.where(lab == -1, 1, lab + 2))
    im = ax[2].imshow(shown, origin="lower", aspect="auto", extent=ext, cmap=ListedColormap(colors),
                      vmin=-0.5, vmax=len(colors) - 0.5, interpolation="nearest")
    ax[2].set_title(f"бассейны, кластеров: {k} (серый — уход в ∞, чёрный — не попал в кластер)")
    for a_ in ax:
        a_.plot(1, 1, "r+", ms=12, mew=2)
        a_.set_xlabel("x0")
        a_.set_ylabel("y0")
    fig.suptitle(f"Bouali: бассейны притяжения, {TITLES[m]} (h = {H}, z0 = {Z0:g})")
    fig.tight_layout()
    path = os.path.join(OUT, f"basins_{NAMES[m]}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    sizes = np.bincount(lab[lab >= 0]) if k else []
    print(f"  {NAMES[m]}: кластеров {k}, размеры {sorted(sizes, reverse=True)[:10]}, "
          f"не в кластере {(lab == -1).sum()}, разошлось {(lab == -2).sum()}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    xs = np.linspace(*X_RANGE, RES)
    ys = np.linspace(*Y_RANGE, RES)
    cache = os.path.join(OUT, "data.npz")
    data = {}
    if "--replot" in sys.argv and os.path.exists(cache):
        d = np.load(cache)
        data = {m: d[NAMES[m]] for m in range(5) if NAMES[m] in d}
    for m in range(5):
        if m not in data:
            data[m] = basins(METHODS[m], xs, ys, Z0, H, T_SKIP, T_WIN, A, S)
            print(f"{NAMES[m]}: посчитано ({time.time() - t0:.0f} с)")
            np.savez(cache, **{NAMES[k]: v for k, v in data.items()})
    for m in range(5):
        plot(m, data[m][0], data[m][1])
    print(f"Готово за {time.time() - t0:.0f} с")