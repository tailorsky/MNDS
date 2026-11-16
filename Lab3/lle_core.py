import numpy as np
from numba import njit, prange

import ExplictEulerMethod as euler
import EulerCromerMethod as eulerCromer
import EMP as emp
import MethodKD as kd
import RungeKuttaMethod4th as rk4

A = np.array([0.5, 3.0, 2.2, 1.0, 0.001])
V0 = np.array([1.0, 1.0, 0.0])

NAMES = ["Euler", "Euler-Cromer", "EMP", "KD", "RK4"]
TITLES = ["Явный Эйлер", "Эйлер-Кромер", "EMP", "КД", "РК4"]

EPS = 1e-8
NT = 0.4
DIVERGE = 1e6
CHUNK = 100_000


@njit
def system(t, v, a):
    x, y, z = v
    return np.array([a[1] * x * (1 - y) - a[2] * z,
                     -a[3] * y * (1 - x ** 2),
                     a[4] * x])


_euler = njit(euler.explicit_euler)
_cromer = njit(eulerCromer.euler_cromer)
_emp = njit(emp.emp)
_kd = njit(kd.KD)
_rk4 = njit(rk4.RK4)


@njit
def m_euler(f, t0, t1, v0, h, a, s):
    return _euler(f, t0, t1, v0, h, a)


@njit
def m_cromer(f, t0, t1, v0, h, a, s):
    return _cromer(f, t0, t1, v0, h, a)


@njit
def m_emp(f, t0, t1, v0, h, a, s):
    return _emp(f, t0, t1, v0, h, a)


@njit
def m_kd(f, t0, t1, v0, h, a, s):
    return _kd(f, t0, t1, v0, h, a, s)


@njit
def m_rk4(f, t0, t1, v0, h, a, s):
    return _rk4(f, t0, t1, v0, h, a)


METHODS = [m_euler, m_cromer, m_emp, m_kd, m_rk4]
KD, RK4 = 3, 4


@njit
def bad(v):
    return not (abs(v[0]) < DIVERGE and abs(v[1]) < DIVERGE and abs(v[2]) < DIVERGE)


@njit
def lle(method, v0, h, t_skip, ct, a, s):
    # 1. переходный процесс (по кускам, чтобы не держать всю траекторию в памяти)
    x = v0.copy()
    n_skip = int(round(t_skip / h))
    done = 0
    while done < n_skip:
        k = min(CHUNK, n_skip - done)
        _, v = method(system, 0.0, k * h, x, h, a, s)
        x = v[k].copy()
        if bad(x):
            return np.nan
        done += k

    # 2. вторая траектория на расстоянии eps в случайном направлении
    p = np.random.rand(3) + 1e-12
    x2 = x + EPS * p / np.linalg.norm(p)

    # 3. окна по NT: считаем обе траектории, копим ln(d/eps), возвращаем вторую на расстояние eps
    nw = max(1, int(round(NT / h)))
    windows = max(1, int(round(ct / (nw * h))))
    total = 0.0
    for w in range(windows):
        _, v1 = method(system, 0.0, nw * h, x, h, a, s)
        _, v2 = method(system, 0.0, nw * h, x2, h, a, s)
        x = v1[nw].copy()
        y = v2[nw]
        if bad(x) or bad(y):
            return np.nan
        d = np.linalg.norm(y - x)
        if d == 0.0:
            d = 1e-300
        total += np.log(d / EPS)
        x2 = x + (y - x) / d * EPS
    return total / (windows * nw * h)


@njit(parallel=True)
def scan_param(method, which, values, v0, h, t_skip, ct, a, s):
    out = np.empty(values.shape[0])
    for i in prange(values.shape[0]):
        aa = a.copy()
        aa[which] = values[i]
        out[i] = lle(method, v0, h, t_skip, ct, aa, s)
    return out


@njit(parallel=True)
def scan_step(method, hs, v0, t_skip, ct, a, s):
    out = np.empty(hs.shape[0])
    for i in prange(hs.shape[0]):
        out[i] = lle(method, v0, hs[i], t_skip, ct, a, s)
    return out


@njit(parallel=True)
def scan_2d(method, i_par, j_par, vi, vj, v0, h, t_skip, ct, a, s):
    ni, nj = vi.shape[0], vj.shape[0]
    out = np.empty((nj, ni))
    for k in prange(ni * nj):
        jj, ii = k // ni, k % ni
        aa = a.copy()
        aa[i_par] = vi[ii]
        aa[j_par] = vj[jj]
        out[jj, ii] = lle(method, v0, h, t_skip, ct, aa, s)
    return out


@njit(parallel=True)
def scan_param_step(method, which, values, hs, order, v0, t_skip, ct, a, s):
    ni, nj = values.shape[0], hs.shape[0]
    out = np.empty((nj, ni))
    for q in prange(ni * nj):
        k = order[q]
        jj, ii = k // ni, k % ni
        aa = a.copy()
        aa[which] = values[ii]
        out[jj, ii] = lle(method, v0, hs[jj], t_skip, ct, aa, s)
    return out


def log_steps(h_min, h_max, n):
    return h_min * (h_max / h_min) ** (np.arange(n) / (n - 1))