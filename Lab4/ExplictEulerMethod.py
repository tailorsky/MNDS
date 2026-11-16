import numpy as np

def explicit_euler (f, t0, t1, v0, h, a):
    n = int(round((t1 - t0) / h))
    t = t0 + h * np.arange(n + 1)
    v = np.zeros((n + 1, len(v0)))
    v[0] = v0
    
    for i in range(n):
        v[i + 1] = v[i] + h * f(t[i], v[i], a)
        
    return t, v