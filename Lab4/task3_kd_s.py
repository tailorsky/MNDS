import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core4 import A, V0, METHODS, KD, RK4, scan_s, phase_volume

H = 0.01
T_SKIP = 1000.0
T_AVG = 1000.0

RANGES = [(0.0, 1.0, 101), (-100.0, 100.0, 201)]
OUT = os.path.join("plots", "task3")


def plot(results, ref, col, ylabel, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (lo, hi, k), (sv, r) in zip(axes, RANGES, results):
        bad = np.isnan(r[:, 0])
        for sb in sv[bad]:
            w = (hi - lo) / (2 * (k - 1))
            ax.axvspan(sb - w, sb + w, color="0.85", lw=0, zorder=0)
        ax.axhline(ref[col], color="tab:red", lw=1.2, label="РК4 (h = 0.01)")
        if col == 0:
            ax.axhline(ref[1], color="k", lw=1, ls="--", label="точно: среднее div f")
        ax.plot(sv, r[:, col], "o-", ms=2.5, lw=0.8, color="tab:blue", label="КД")
        if lo < 0:
            ax.axvspan(0, 1, color="tab:green", alpha=0.2, lw=0)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("коэффициент симметрии s")
        ax.set_ylabel(ylabel)
        ax.set_title(f"s ∈ [{lo:g}, {hi:g}]")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"Bouali: {title} для КД в зависимости от s (h = {H}); серым — разошлось")
    fig.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("сохранено:", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    ref = phase_volume(METHODS[RK4], V0, H, T_SKIP, T_AVG, A, 0.5)
    results = []
    for lo, hi, k in RANGES:
        sv = np.linspace(lo, hi, k)
        r = scan_s(METHODS[KD], sv, V0, H, T_SKIP, T_AVG, A)
        print(f"s ∈ [{lo}, {hi}]: разошлось {np.isnan(r[:, 0]).sum()} из {k}")
        results.append((sv, r))
    # скорость изменения фазового объёма пока не выводим
    # plot(results, ref, 0, "скорость изм. фаз. объёма, 1/ед. времени", "PV_KD_s.png",
    # "скорость изменения фазового объёма")
    plot(results, ref, 2, "фазовый объём V = Δx·Δy·Δz", "PV_box_KD_s.png", "фазовый объём (объём аттрактора)")
    print(f"Готово за {time.time() - t0:.0f} с")