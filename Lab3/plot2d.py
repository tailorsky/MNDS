import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def make_norm(maps):
    # симметричная шкала: белый = 0, иначе шум около нуля выглядит как сильный минус
    v = np.concatenate([m[np.isfinite(m)] for m in maps])
    lim = max(np.percentile(np.abs(v), 99), 1e-4)
    return Normalize(vmin=-lim, vmax=lim)


def draw(ax, data, extent, norm, log_y=False, y=None):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.6")
    masked = np.ma.masked_invalid(data)
    if log_y:
        x = np.linspace(extent[0], extent[1], data.shape[1])
        im = ax.pcolormesh(x, y, masked, cmap=cmap, norm=norm, shading="nearest")
        ax.set_yscale("log")
    else:
        im = ax.imshow(masked, origin="lower", aspect="auto", extent=extent,
                       cmap=cmap, norm=norm, interpolation="nearest")
    return im