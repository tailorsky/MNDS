import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core4 import A, V0, NAMES, TITLES, METHODS, continuation_many

RUNS = 100
T_RUN = 1000.0
HS = np.array([0.002, 0.005, 0.01, 0.02, 0.04])
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
MAX_PEAKS = 2000
S = A[0]
OUT = os.path.join("plots", "task4")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    runs = np.arange(1, RUNS + 1)
    for m in range(5):
        bufs, cnts = continuation_many(METHODS[m], V0, HS, T_RUN, RUNS, A, S, MAX_PEAKS)
        fig, ax = plt.subplots(figsize=(14, 7))
        for j, h in enumerate(HS):
            ok = cnts[j] >= 0
            k = np.repeat(runs[ok], cnts[j][ok])
            v = np.concatenate([bufs[j, i, :cnts[j, i]] for i in np.nonzero(ok)[0]]) if ok.any() else np.empty(0)
            ax.plot(k, v, "o", ms=3, color=COLORS[j], alpha=0.8, label=f"h = {h:g}")
            if not ok.all():
                ax.axvline(runs[~ok][0], color=COLORS[j], lw=1, ls="--")
                print(f"  {NAMES[m]}, h = {h}: разошлось на прогоне {runs[~ok][0]}")
            elif len(v):
                print(f"  {NAMES[m]}, h = {h}: пики z от {v.min():.4f} до {v.max():.4f}, "
                      f"последний прогон {bufs[j, -1, :cnts[j, -1]].round(4)}")
        ax.set_xlim(0, RUNS + 1)
        ax.set_xlabel(f"номер прогона (каждый прогон {T_RUN:g} ед. времени, старт из конечной точки предыдущего)")
        ax.set_ylabel("максимумы z")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", markerscale=2)
        ax.set_title(f"Bouali: диаграмма продолжения, {TITLES[m]}, разные шаги "
                     "(пунктир цвета шага — решение разошлось)")
        fig.tight_layout()
        path = os.path.join(OUT, f"continuation_{NAMES[m]}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"  сохранено: {path} ({time.time() - t0:.0f} с)")
    print(f"Готово за {time.time() - t0:.0f} с")