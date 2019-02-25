import numpy as np

# Clamp v s.t a <= v <= b
def clamp(v, a, b):
    return max(min(v, b), a)

# Simple grayscale blitting on a 2D buffer
def blit(buffer, rect):
    x, y, w, h = rect
    shape = np.shape(buffer)
    buf_w = shape[0]
    buf_h = shape[1]

    x1 = int(clamp(x, 0, buf_w))
    y1 = int(clamp(y, 0, buf_h))
    x2 = int(clamp(x + w, 0, buf_w))
    y2 = int(clamp(y + h, 0, buf_h))

    for w_ in range(x1, x2):
        for h_ in range(y1, y2):
            buffer[w_][h_] = 1.0