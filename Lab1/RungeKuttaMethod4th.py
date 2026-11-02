import numpy as np

def RK4(f, t0, t1, v0, h):
    n = int(round((t1 - t0) / h))
    t = t0 + h * np.arange(n + 1)
    v = np.zeros((n + 1, len(v0)))
    v[0] = v0
    
    for i in range(n):
        k1 = f(t[i], v[i])
        k2 = f(t[i] + h/2, v[i] + (h/2) * k1)
        k3 = f(t[i] + h/2, v[i] + (h/2) * k2)
        k4 = f(t[i] + h, v[i] + h * k3)
        
        v[i+1] = v[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
    return t, v