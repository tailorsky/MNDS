import os
import numpy as np
import matplotlib.pyplot as plt

from core5 import A, H, VARS, start_points, trajectory, time_reversible
from plotting import folder, plot_traj, log_limits, COMP, FLOOR

T_WIN = 200.0
K = 10.0
KS = [1.0, 3.0, 10.0, 30.0, 100.0]
ITERS = 40
KEEP = np.array([0, 1, 2, 5, 10, ITERS - 1])

out = folder("task3")
m0, s0 = start_points()
n = int(T_WIN / H)
t = H * np.arange(n + 1)
master = trajectory(m0, H, n, A)

fig_all, ax_all = plt.subplots(figsize=(9, 5))
all_err = []
lines = []
for d, name in enumerate(VARS):
    err_f, err_b, passes = time_reversible(master, s0, d, K, H, ITERS, A, KEEP)
    ax_all.semilogy(np.arange(1, ITERS + 1), np.maximum(err_f, FLOOR), "o-", ms=3, color=COMP[d], label=f"по {name}")
    all_err.append(np.maximum(err_f, FLOOR))
    lines.append(f"{name}: K = {K}, ||e|| после итераций 1/5/10/{ITERS}: "
                 + ", ".join(f"{err_f[i]:.2e}" for i in [0, 4, 9, ITERS - 1]))
    print(lines[-1])

    plot_traj(t, master, passes[-1], f"Возвратная синхронизация по {name}: итерация {ITERS}",
              os.path.join(out, f"TR_traj_{name}.png"))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(KEEP)))
    curves = []
    for c, it, p in zip(cmap, KEEP, passes):
        e = np.sqrt(((p - master) ** 2).sum(1))
        curves.append(np.maximum(e, FLOOR))
        ax.semilogy(t[::10], np.maximum(e[::10], FLOOR), color=c, lw=1.2, label=f"итерация {it + 1}")
    log_limits(ax, *curves)
    ax.set_xlabel("t")
    ax.set_ylabel("||e||")
    ax.set_title(f"Возвратная синхронизация по {name}: ошибка вдоль окна, K = {K:g}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"TR_err_{name}.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    curves = []
    for c, k in zip(plt.cm.plasma(np.linspace(0, 0.85, len(KS))), KS):
        ef, _, _ = time_reversible(master, s0, d, k, H, ITERS, A, KEEP[:0])
        curves.append(np.maximum(ef, FLOOR))
        ax.semilogy(np.arange(1, ITERS + 1), np.maximum(ef, FLOOR), "o-", ms=3, color=c, label=f"K = {k:g}")
    log_limits(ax, *curves)
    ax.set_xlabel("номер итерации")
    ax.set_ylabel("||e|| в конце окна")
    ax.set_title(f"Возвратная синхронизация по {name}: ошибка по итерациям")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"TR_iter_K_{name}.png"), dpi=150)
    plt.close(fig)

log_limits(ax_all, *all_err)
ax_all.set_xlabel("номер итерации")
ax_all.set_ylabel("||e|| в конце окна")
ax_all.set_title(f"Возвратная синхронизация: ошибка по итерациям, K = {K:g}, окно {T_WIN:g}")
ax_all.grid(alpha=0.3, which="both")
ax_all.legend()
fig_all.tight_layout()
fig_all.savefig(os.path.join(out, "TR_iter.png"), dpi=150)
plt.close(fig_all)

with open(os.path.join(out, "results.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
