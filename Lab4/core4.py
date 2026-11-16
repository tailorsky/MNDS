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
DELTA = 1e-7


@njit
def system(t, v, a):
    x, y, z = v
    return np.array([a[1] * x * (1 - y) - a[2] * z,
                     -a[3] * y * (1 - x ** 2),
                     a[4] * x])


@njit
def divergence(v, a):
    return a[1] * (1 - v[1]) - a[3] * (1 - v[0] ** 2)


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
def advance(method, x, h, t, a, s):
    # интегрирует t единиц времени по кускам; NaN в ответе = решение разошлось
    n = int(round(t / h))
    done = 0
    while done < n:
        k = min(CHUNK, n - done)
        _, v = method(system, 0.0, k * h, x, h, a, s)
        x = v[k].copy()
        if bad(x):
            x[:] = np.nan
            return x
        done += k
    return x


@njit
def one_step(method, x, h, a, s):
    _, v = method(system, 0.0, h, x, h, a, s)
    return v[1].copy()


@njit
def phase_volume(method, v0, h, t_skip, t_avg, a, s):
    # [0] скорость изменения фазового объёма у метода: (1/T)·Σ ln|det J_k|,
    #     где J_k — матрица Якоби одного шага метода (считается разностями)
    # [1] то же для исходной системы: среднее div f вдоль траектории
    # [2] объём параллелепипеда, в который помещается аттрактор
    out = np.full(3, np.nan)
    x = advance(method, v0.copy(), h, t_skip, a, s)
    if np.isnan(x[0]):
        return out
    n = int(round(t_avg / h))
    J = np.empty((3, 3))
    lo = x.copy()
    hi = x.copy()
    sum_log = 0.0
    sum_div = 0.0
    for k in range(n):
        fx = one_step(method, x, h, a, s)
        for j in range(3):
            xp = x.copy()
            xp[j] += DELTA
            J[:, j] = (one_step(method, xp, h, a, s) - fx) / DELTA
        det = np.linalg.det(J)
        if bad(fx) or not abs(det) > 0.0:
            return out
        sum_log += np.log(abs(det))
        sum_div += divergence(x, a)
        x = fx
        for c in range(3):
            lo[c] = min(lo[c], x[c])
            hi[c] = max(hi[c], x[c])
    out[0] = sum_log / (n * h)
    out[1] = sum_div / n
    out[2] = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    return out


@njit(parallel=True)
def scan_param(method, which, values, v0, h, t_skip, t_avg, a, s):
    out = np.empty((values.shape[0], 3))
    for i in prange(values.shape[0]):
        aa = a.copy()
        aa[which] = values[i]
        out[i] = phase_volume(method, v0, h, t_skip, t_avg, aa, s)
    return out


@njit(parallel=True)
def scan_step(method, hs, v0, t_skip, t_avg, a, s):
    out = np.empty((hs.shape[0], 3))
    for i in prange(hs.shape[0]):
        out[i] = phase_volume(method, v0, hs[i], t_skip, t_avg, a, s)
    return out


@njit(parallel=True)
def scan_s(method, svals, v0, h, t_skip, t_avg, a):
    out = np.empty((svals.shape[0], 3))
    for i in prange(svals.shape[0]):
        out[i] = phase_volume(method, v0, h, t_skip, t_avg, a, svals[i])
    return out


@njit
def z_maxima(method, x, h, t, a, s, out):
    # считает t единиц времени, пишет максимумы z в out; возвращает (конечная точка, их число)
    n = int(round(t / h))
    cnt = 0
    z2 = x[2]
    z1 = x[2]
    done = 0
    while done < n:
        k = min(CHUNK, n - done)
        _, v = method(system, 0.0, k * h, x, h, a, s)
        for r in range(1, k + 1):
            if bad(v[r]):
                x = v[r].copy()
                x[:] = np.nan
                return x, -1
            if z1 > z2 and z1 > v[r, 2] and cnt < out.shape[0]:
                out[cnt] = z1
                cnt += 1
            z2 = z1
            z1 = v[r, 2]
        x = v[k].copy()
        done += k
    return x, cnt


@njit
def continuation(method, v0, h, t_run, runs, a, s, maxp):
    # прогон за прогоном: каждый следующий стартует из конечной точки предыдущего, параметры не меняются
    buf = np.empty((runs, maxp))
    cnt = np.zeros(runs, dtype=np.int64)
    x = v0.copy()
    for i in range(runs):
        x, ci = z_maxima(method, x, h, t_run, a, s, buf[i])
        cnt[i] = ci
        if ci < 0:
            cnt[i:] = -1
            break
    return buf, cnt


@njit(parallel=True)
def continuation_many(method, v0, hs, t_run, runs, a, s, maxp):
    bufs = np.empty((hs.shape[0], runs, maxp))
    cnts = np.zeros((hs.shape[0], runs), dtype=np.int64)
    for j in prange(hs.shape[0]):
        b, c = continuation(method, v0, hs[j], t_run, runs, a, s, maxp)
        bufs[j] = b
        cnts[j] = c
    return bufs, cnts


@njit
def z_peaks(method, x, h, t, a, s, vals, times):
    # как z_maxima, но запоминает ещё и моменты времени пиков
    n = int(round(t / h))
    cnt = 0
    z2 = x[2]
    z1 = x[2]
    done = 0
    while done < n:
        k = min(CHUNK, n - done)
        _, v = method(system, 0.0, k * h, x, h, a, s)
        for r in range(1, k + 1):
            if bad(v[r]):
                return -1
            if z1 > z2 and z1 > v[r, 2] and cnt < vals.shape[0]:
                vals[cnt] = z1
                times[cnt] = (done + r - 1) * h
                cnt += 1
            z2 = z1
            z1 = v[r, 2]
        x = v[k].copy()
        done += k
    return cnt


@njit(parallel=True)
def basins(method, xs, ys, z0, h, t_skip, t_win, a, s):
    # out[0] — средняя амплитуда (среднее значение пиков z), out[1] — средний межпиковый интервал; NaN = разошлось
    nx, ny = xs.shape[0], ys.shape[0]
    out = np.empty((2, ny, nx))
    for q in prange(nx * ny):
        j, i = q // nx, q % nx
        x = np.array([xs[i], ys[j], z0])
        x = advance(method, x, h, t_skip, a, s)
        if np.isnan(x[0]):
            out[:, j, i] = np.nan
            continue
        vals = np.empty(20000)
        times = np.empty(20000)
        cnt = z_peaks(method, x, h, t_win, a, s, vals, times)
        if cnt < 0:
            out[:, j, i] = np.nan
        elif cnt == 0:
            out[0, j, i] = 0.0
            out[1, j, i] = t_win
        else:
            out[0, j, i] = vals[:cnt].mean()
            out[1, j, i] = (times[cnt - 1] - times[0]) / (cnt - 1) if cnt > 1 else t_win
    return out


def log_steps(h_min, h_max, n):
    return h_min * (h_max / h_min) ** (np.arange(n) / (n - 1))


def flatten(values, buf, cnt):
    ok = cnt > 0
    p = np.repeat(values[ok], cnt[ok])
    v = np.concatenate([buf[i, :cnt[i]] for i in np.nonzero(ok)[0]]) if ok.any() else np.empty(0)
    return p, v