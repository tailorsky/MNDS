import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core5 import VARS

MASTER = "#1f4e9c"
SLAVE = "#d9622b"
COMP = ["#1f4e9c", "#2a9d5c", "#c0392b"]


FLOOR = 1e-17


def log_limits(ax, *arrays):
    vals = np.concatenate([np.ravel(a) for a in arrays])
    vals = vals[np.isfinite(vals)]
    ax.set_ylim(max(FLOOR, vals.min()) / 3, vals.max() * 3)


def folder(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", name)
    os.makedirs(path, exist_ok=True)
    return path


def plot_traj(t, m, sl, title, path, every=10):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t[::every], m[::every, j], color=MASTER, lw=1.6, label="ведущая")
        ax.plot(t[::every], sl[::every, j], color=SLAVE, lw=1.2, ls="--", label="ведомая")
        ax.set_ylabel(VARS[j])
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_err(t, m, sl, title, path, skip=None, every=10):
    e = np.maximum(np.abs(m - sl)[::every], FLOOR)
    tt = t[::every]
    total = np.maximum(np.sqrt((e ** 2).sum(1)), FLOOR)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for j in range(3):
        if j != skip:
            ax.semilogy(tt, e[:, j], color=COMP[j], lw=1, label=f"|e_{VARS[j]}|")
    ax.semilogy(tt, total, color="k", lw=1.6, label="||e||")
    log_limits(ax, total, *[e[:, j] for j in range(3) if j != skip])
    ax.set_xlabel("t")
    ax.set_ylabel("ошибка синхронизации")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left", ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
