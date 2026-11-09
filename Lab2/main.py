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
def peaks(method, v0, h, n, skip, a, s, out):
    cnt = np.zeros(3, dtype=np.int64)
    maxp = out.shape[1]
    state = v0.copy()
    p2 = v0.copy()
    p1 = v0.copy()
    done = 0
    while done < n:
        k = min(CHUNK, n - done)
        _, v = method(system, 0.0, k * h, state, h, a, s)
        for r in range(1, k + 1):
            i = done + r
            for c in range(3):
                if not abs(v[r, c]) < DIVERGE:
                    cnt[:] = -1
                    return cnt
            if i >= skip + 2:
                for c in range(3):
                    if p1[c] > p2[c] and p1[c] > v[r, c] and cnt[c] < maxp:
                        out[c, cnt[c]] = p1[c]
                        cnt[c] += 1
            p2[:] = p1
            p1[:] = v[r]
        state = v[k].copy()
        done += k
    return cnt


@njit(parallel=True)
def scan_param(method, which, values, v0, h, n, skip, a, s, maxp):
    k = values.shape[0]
    buf = np.empty((k, 3, maxp))
    cnt = np.zeros((k, 3), dtype=np.int64)
    for i in prange(k):
        aa = a.copy()
        aa[which] = values[i]
        cnt[i] = peaks(method, v0, h, n, skip, aa, s, buf[i])
    return buf, cnt


@njit(parallel=True)
def scan_step(method, hs, v0, t_total, t_skip, a, s, maxp):
    k = hs.shape[0]
    buf = np.empty((k, 3, maxp))
    cnt = np.zeros((k, 3), dtype=np.int64)
    for i in prange(k):
        h = hs[i]
        cnt[i] = peaks(method, v0, h, int(t_total / h), int(t_skip / h), a, s, buf[i])
    return buf, cnt


@njit(parallel=True)
def scan_s(method, svals, v0, h, n, skip, a, maxp):
    k = svals.shape[0]
    buf = np.empty((k, 3, maxp))
    cnt = np.zeros((k, 3), dtype=np.int64)
    for i in prange(k):
        cnt[i] = peaks(method, v0, h, n, skip, a, svals[i], buf[i])
    return buf, cnt


@njit
def clusters(p, tol):
    if p.shape[0] == 0:
        return 0
    q = np.sort(p)
    c = 1
    for i in range(1, q.shape[0]):
        if q[i] - q[i - 1] > tol:
            c += 1
    return c


@njit(parallel=True)
def scan_2d(method, i_par, j_par, vi, vj, v0, h, n, skip, a, s, maxp, tol):
    ni, nj = vi.shape[0], vj.shape[0]
    res = np.zeros((3, nj, ni), dtype=np.int64)
    for k in prange(ni * nj):
        jj = k // ni
        ii = k % ni
        aa = a.copy()
        aa[i_par] = vi[ii]
        aa[j_par] = vj[jj]
        buf = np.empty((3, maxp))
        cnt = peaks(method, v0, h, n, skip, aa, s, buf)
        for c in range(3):
            res[c, jj, ii] = -1 if cnt[c] < 0 else clusters(buf[c, :cnt[c]], tol)
    return res


def flatten(values, buf, cnt, c):
    ok = cnt[:, c] > 0
    p = np.repeat(values[ok], cnt[ok, c])
    v = np.concatenate([buf[i, c, :cnt[i, c]] for i in np.nonzero(ok)[0]]) if ok.any() else np.empty(0)
    return p, v


def cloud(ax, p, v, color, label=None, alpha=1.0):
    if len(v):
        ax.plot(p, v, ".", ms=0.8, mew=0, color=color, alpha=alpha, label=label, rasterized=True)