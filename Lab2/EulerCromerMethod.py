import numpy as np

def euler_cromer (f, t0, t1, v0, h, a):
    n = int(round((t1 - t0) / h))
    t = t0 + h * np.arange(n + 1)
    v = np.zeros((n + 1, len(v0)))
    v[0] = v0
    
    for i in range(n):
        x, y, z = v[i]
        
        x_new = x + h * (a[1] * x * (1 - y) - a[2] * z)
        y_new = y + h * (-a[3] * y * (1 - x_new ** 2))
        z_new = z + h * (a[4] * x_new)
        
        v[i+1] = [x_new, y_new, z_new]
    
    return t, v