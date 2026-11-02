import numpy as np

def emp(f, t0, t1, v0, h):
    n = int(round((t1 - t0) / h))
    t = t0 + h * np.arange(n + 1)
    v = np.zeros((n + 1, len(v0)))
    v[0] = v0
    
    for i in range(n):
        k1 = f(t[i], v[i])
        v_mid = v[i] + (h/2) * k1
        k2 = f(t[i] + h/2, v_mid)
        v[i + 1] = v[i] + h * k2
        
    return t, v