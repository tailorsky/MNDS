import numpy as np
from numba import njit, prange

import MethodKD as kd

A = np.array([0.5, 3.0, 2.2, 1.0, 0.001])
V0 = np.array([1.0, 1.0, 0.0])
VARS = ["x", "y", "z"]

H = 0.01
S = 0.5
T_TRANSIENT = 2000.0
SLAVE_SHIFT = np.array([0.1, 0.1, 0.05])


@njit
def system(t, v, a):
    x, y, z = v
    return np.array([a[1] * x * (1 - y) - a[2] * z,
                     -a[3] * y * (1 - x ** 2),
                     a[4] * x])


_kd = njit(kd.KD)


@njit
def step(v, h, a):
    _, out = _kd(system, 0.0, h, v, h, a, S)
    return out[1].copy()


@njit
def trajectory(v0, h, n, a):
    out = np.empty((n + 1, 3))
    out[0] = v0
    for i in range(n):
        out[i + 1] = step(out[i], h, a)
    return out


def start_points(a=A):
    master = trajectory(V0, H, int(T_TRANSIENT / H), a)[-1]
    return master, master + SLAVE_SHIFT


@njit
def norm(v):
    return np.sqrt((v ** 2).sum())


@njit
def pecora_carroll(m0, s0, d, h, n, a):
    m = np.empty((n + 1, 3))
    sl = np.empty((n + 1, 3))
    m[0] = m0
    sl[0] = s0
    sl[0, d] = m0[d]
    for i in range(n):
        m[i + 1] = step(m[i], h, a)
        sl[i + 1] = step(sl[i], h, a)
        sl[i + 1, d] = m[i + 1, d]
    return m, sl


@njit
def conditional_lyapunov(m0, d, h, t_skip, t_avg, a):
    eps = 1e-8
    m = m0.copy()
    for i in range(int(t_skip / h)):
        m = step(m, h, a)
    sl = m.copy()
    for j in range(3):
        if j != d:
            sl[j] += eps / np.sqrt(2.0)
    acc = 0.0
    n = int(t_avg / h)
    for i in range(n):
        m = step(m, h, a)
        sl = step(sl, h, a)
        sl[d] = m[d]
        dist = norm(sl - m)
        acc += np.log(dist / eps)
        sl = m + (sl - m) * (eps / dist)
    return acc / (n * h)


@njit
def bidirectional(m0, s0, d, k, h, n, a):
    m = np.empty((n + 1, 3))
    sl = np.empty((n + 1, 3))
    m[0] = m0
    sl[0] = s0
    q = 0.5 * (1 - np.exp(-2 * k * h))
    for i in range(n):
        m[i + 1] = step(m[i], h, a)
        sl[i + 1] = step(sl[i], h, a)
        diff = sl[i + 1, d] - m[i + 1, d]
        m[i + 1, d] += q * diff
        sl[i + 1, d] -= q * diff
    return m, sl


@njit(parallel=True)
def scan_bidirectional(m0, s0, d, ks, h, n, tail, a):
    out = np.empty(len(ks))
    for j in prange(len(ks)):
        m, sl = bidirectional(m0, s0, d, ks[j], h, n, a)
        e = 0.0
        for i in range(n + 1 - tail, n + 1):
            e = max(e, norm(m[i] - sl[i]))
        out[j] = e if e == e else np.inf
    return out


@njit
def time_reversible(master, s0, d, k, h, iters, a, keep):
    n = master.shape[0] - 1
    c = 1 - np.exp(-k * h)
    s = s0.copy()
    err_f = np.empty(iters)
    err_b = np.empty(iters)
    passes = np.empty((len(keep), n + 1, 3))
    slot = 0
    for it in range(iters):
        store = slot < len(keep) and keep[slot] == it
        if store:
            passes[slot, 0] = s
        for i in range(n):
            s = step(s, h, a)
            s[d] += c * (master[i + 1, d] - s[d])
            if store:
                passes[slot, i + 1] = s
        if store:
            slot += 1
        err_f[it] = norm(s - master[n])
        for i in range(n, 0, -1):
            s = step(s, -h, a)
            s[d] += c * (master[i - 1, d] - s[d])
        err_b[it] = norm(s - master[0])
    return err_f, err_b, passes
