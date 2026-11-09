import numpy as np

def KD(f, t0, t1, v0, h, a, s = 0.5):
    n = int(round((t1 - t0) / h))
    t = t0 + h * np.arange(n + 1)
    v = np.zeros((n + 1, len(v0)))
    v[0] = v0
    
    h1 = h * s
    h2 = h * (1 - s)
    
    for i in range(n):
        x, y, z = v[i]
        
        x_s = x + h1 * (a[1] * x * (1-y) - a[2] * z)
        y_s = y + h1 * (-a[3] * y * (1 - x_s ** 2))
        z_s = z + h1 * (a[4] * x_s)
        
        z1 = z_s + h2 * a[4] * x_s
        y1 = y_s / (1 + h2 * a[3] * (1 - x_s**2))
        x1 = (x_s - h2 * a[2] * z1) / (1 - h2 * a[1] * (1 - y1))
        
        v[i+1] = [x1, y1, z1]
        
    return t, v
        