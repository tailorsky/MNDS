import numpy as np

def KD(f, t0, t1, v0, h, s = 0.5):
    n = int(round((t1 - t0) / h))
    t = t0 + h * np.arange(n + 1)
    v = np.zeros((n + 1, len(v0)))
    v[0] = v0
    
    h1 = h * s
    h2 = h * (1 - s)
    
    for i in range(n):
        x, y, z = v[i]
        
        x_s = x + h1 * (3 * x * (1-y) - 2.2 * z)
        y_s = y + h1 * (-1 * y * (1 - x_s ** 2))
        z_s = z + h1 * (0.001 * x_s)
        
        z1 = z_s + h2 * 0.001 * x_s
        y1 = y_s / (1 + h2 * (1 - x_s**2))
        x1 = (x_s - h2 * 2.2 * z1) / (1 - h2 * 3 * (1 - y1))
        
        v[i+1] = [x1, y1, z1]
        
    return t, v
        