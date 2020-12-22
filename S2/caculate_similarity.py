import math
import numpy as np
def caculate():
    angel90=[math.pi/2-i*(math.pi / 25) for i in range(13)]
    angel0=angel90[::-1]
    angel0.pop()
    fd1=[]
    r=[6.65,6.95,7.25]
    for layer in range(3):
        for j in range(12):
            f1 = 3.7 * (50 - r[layer] * np.sin(angel90[j])) * np.sin(angel90[j]) - 924.528 * np.cos(angel90[j])
            f2 = 3.7 * (50 - r[layer] * np.sin(angel90[j + 1])) * np.sin(angel90[j + 1]) - 924.528 * np.cos(
                angel90[j + 1])
            f1 = np.abs(f1)
            f2 = np.abs(f2)
            if f2 > f1:
                fd1.append(f2 / f1)
            else:
                fd1.append(f1 / f2)
        fd1.append(1)
        for j in range(13, 24):
            f1 = 3.7 * (50 - r[layer] * np.sin(angel0[j - 13])) * np.sin(angel0[j - 13]) + 626.088 * np.cos(
                angel0[j - 13])
            f2 = 3.7 * (50 - r[layer]* np.sin(angel0[j - 12])) * np.sin(angel0[j - 12]) + 626.088 * np.cos(
                angel0[j - 12])
            f1 = np.abs(f1)
            f2 = np.abs(f2)
            if f2 > f1:
                fd1.append(f2 / f1)
            else:
                fd1.append(f1 / f2)
        fd1.append(1)
        for j in range(25, 37):
            f1 = 3.7 * (50 - r[layer] * np.sin(angel90[j - 25])) * np.sin(angel90[j - 25]) + 626.088 * np.cos(
                angel90[j - 25])
            f2 = 3.7 * (50 - r[layer] * np.sin(angel90[j - 24])) * np.sin(angel90[j - 24]) + 626.088 * np.cos(
                angel90[j - 24])
            f1 = np.abs(f1)
            f2 = np.abs(f2)
            if f2 > f1:
                fd1.append(f2 / f1)
            else:
                fd1.append(f1 / f2)
        fd1.append(1)
        for j in range(38, 49):
            f1 = 3.7 * (50 - r[layer]  * np.sin(angel0[j - 38])) * np.sin(angel0[j - 38]) - 924.528 * np.cos(
                angel0[j - 38])
            f2 = 3.7 * (50 - r[layer]  * np.sin(angel0[j - 37])) * np.sin(angel0[j - 37]) - 924.528 * np.cos(
                angel0[j - 37])
            f1 = np.abs(f1)
            f2 = np.abs(f2)
            if f2 > f1:
                fd1.append(f2 / f1)
            else:
                fd1.append(f1 / f2)
    fd2=fd1[49:98]
    fd3=fd1[98:]
    fd1=fd1[:49]
    return fd1,fd2,fd3

