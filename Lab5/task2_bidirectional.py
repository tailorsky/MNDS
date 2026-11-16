import os
import numpy as np
import matplotlib.pyplot as plt

from core5 import A, H, VARS, start_points, bidirectional, scan_bidirectional
from plotting import folder, plot_traj, plot_err, COMP

T = 3000.0
T_TAIL = 500.0
KS = np.logspace(-2, 2, 41)

out = folder("task2")
m0, s0 = start_points()
n = int(T / H)
tail = int(T_TAIL / H)
t = H * np.arange(n + 1)

fig, ax = plt.subplots(figsize=(9, 5))
lines = []
for d, name in enumerate(VARS):
    err = scan_bidirectional(m0, s0, d, KS, H, n, tail, A)
    ax.loglog(KS, err, "o-", ms=3, color=COMP[d], label=f"связь по {name}")
    k = KS[np.argmin(err)]
    m, sl = bidirectional(m0, s0, d, k, H, n, A)
    e = np.sqrt(((m - sl) ** 2).sum(1))
    lines.append(f"{name}: лучший k = {k:.3g}, ошибка на хвосте = {err.min():.3e}, ||e(T)|| = {e[-1]:.3e}")
    print(lines[-1])
    plot_traj(t, m, sl, f"Двунаправленная синхронизация по {name}, k = {k:.3g}",
              os.path.join(out, f"BI_traj_{name}.png"))
    plot_err(t, m, sl, f"Двунаправленная синхронизация по {name}, k = {k:.3g}: ошибка",
             os.path.join(out, f"BI_err_{name}.png"))
ax.set_xlabel("коэффициент связи k")
ax.set_ylabel(f"max ||e|| на последних {T_TAIL:.0f} ед. времени")
ax.set_title("Двунаправленная синхронизация: ошибка от силы связи")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out, "BI_scan_k.png"), dpi=150)
plt.close(fig)

with open(os.path.join(out, "results.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
