import os
import numpy as np

from core5 import A, H, VARS, start_points, pecora_carroll, conditional_lyapunov
from plotting import folder, plot_traj, plot_err

T = 3000.0
T_SKIP = 500.0
T_AVG = 10000.0

out = folder("task1")
m0, s0 = start_points()
n = int(T / H)
t = H * np.arange(n + 1)

lines = []
for d, name in enumerate(VARS):
    m, sl = pecora_carroll(m0, s0, d, H, n, A)
    lam = conditional_lyapunov(m0, d, H, T_SKIP, T_AVG, A)
    e = np.sqrt(((m - sl) ** 2).sum(1))
    lines.append(f"{name}: CLE = {lam:+.5f}, ||e(0)|| = {e[0]:.3e}, "
                 f"||e(T)|| = {e[-1]:.3e}, max||e|| = {e.max():.3e}")
    print(lines[-1])
    plot_traj(t, m, sl, f"Пекора-Кэррол, синхронизация по {name}",
              os.path.join(out, f"PC_traj_{name}.png"))
    plot_err(t, m, sl, f"Пекора-Кэррол, синхронизация по {name}: ошибка",
             os.path.join(out, f"PC_err_{name}.png"), skip=d)

with open(os.path.join(out, "results.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
