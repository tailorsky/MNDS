import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import numpy as np
import ExplictEulerMethod as euler
import EulerCromerMethod as eulerCromer
import EMP as emp
import RungeKuttaMethod4th as rk4
import MethodKD as kd

A = [0.5, 3, 2.2, 1, 0.001]


def system(t, v, a=A):
    x, y, z = v
    return np.array([a[1] * x * (1 - y) - a[2] * z,
                     -a[3] * y * (1 - x ** 2),
                     a[4] * x])


V0 = [1.0, 1.0, 0.0]
T_END = 1000
H = 0.01
OUT = "plots"

METHODS = {
    "Явный Эйлер": euler.explicit_euler,
    "Эйлер-Кромер": eulerCromer.euler_cromer,
    "EMP": emp.emp,
    "КД": kd.KD,
    "РК4": rk4.RK4,
}


def plot_trajectory(name, t, v):
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"{name}, h = {H}")

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(v[:, 0], v[:, 1], v[:, 2], lw=0.3, color="k")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Фазовый портрет")

    for i, label in enumerate("xyz"):
        ax = fig.add_subplot(3, 2, 2 * i + 2)
        ax.plot(t, v[:, i], lw=0.5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    ax.set_xlabel("t")

    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=120)

def error_analysis(t_end=1.0):
    ref = solve_ivp(system, (0, t_end), V0, method="DOP853",
                    rtol=1e-13, atol=1e-14).y[:, -1]
    hs = 0.1 / 2 ** np.arange(8)

    print(f"Глобальная ошибка в t = {t_end}")
    print(f"{'Метод':<14}{'порядок':>8}   ошибки для h = " + " ".join(f"{h:.1e}" for h in hs))

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, method in METHODS.items():
        errs = np.array([np.linalg.norm(method(system, 0, t_end, V0, h)[1][-1] - ref)
                         for h in hs])
        ok = errs > 1e-12
        order = np.polyfit(np.log(hs[ok]), np.log(errs[ok]), 1)[0]

        ax.loglog(hs, errs, "o-", label=f"{name}: p ≈ {order:.2f}")
        print(f"{name:<14}{order:>8.3f}   " + " ".join(f"{e:.1e}" for e in errs))

    for p, style in [(1, ":"), (2, "--"), (4, "-.")]:
        ax.loglog(hs, 1e-3 * (hs / hs[0]) ** p, style, color="gray", lw=1, label=f"~h^{p}")

    ax.set_xlabel("шаг h")
    ax.set_ylabel("глобальная ошибка")
    ax.set_title("Зависимость глобальной ошибки от шага")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/errors.png", dpi=120)

def error_over_time(t_end=120.0, hs=(0.1, 0.05, 0.025, 0.0125)):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
 
    for ax, (name, method) in zip(axes, METHODS.items()):
        for h in hs:
            t, v = method(system, 0, t_end, V0, h)
            ref = solve_ivp(system, (0, t_end), V0, method="DOP853", t_eval=t,
                            rtol=1e-13, atol=1e-14).y.T
            err = np.linalg.norm(v - ref, axis=1)
            ax.semilogy(t[1:], err[1:], label=f"h = {h}")
 
        ax.set_title(name)
        ax.set_xlabel("t")
        ax.set_ylabel("ошибка")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
 
    axes[-1].axis("off")
    fig.suptitle("Зависимость глобальной ошибки от времени")
    fig.tight_layout()
    fig.savefig(f"{OUT}/errors_time.png", dpi=120)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    results = {}
    for name, method in METHODS.items():
        print(f"Считаю: {name}")
        results[name] = method(system, 0, T_END, V0, H)
        plot_trajectory(name, *results[name])

    error_analysis()
    error_over_time()
    plt.show()