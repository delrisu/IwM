import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
from numba import jit, cuda 

@jit 
def bresenhamLine(x1, y1, x2, y2):
    d,dx,dy,ai,bi,xi,yi = (0,)*7
    x,y = x1,y1
    
    coords = [(x1,y1)]
    
    if (x1 < x2):
        xi =1
        dx = x2 - x1
    else:
        xi = -1
        dx = x1 - x2
    
    if (y1 < y2):
        yi = 1
        dy = y2 - y1
    else:
        yi = -1
        dy = y1 - y2
    
    if (dx > dy):
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx
        
        while (x != x2):
            if (d >= 0):
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi
            coords.append((x, y))
    else:
        ai = (dx - dy) * 2
        bi = dx * 2
        d = bi - dy
        
        while (y != y2):
            if (d >=0):
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi
            coords.append((x, y))
    coords.append((x2,y2))
    return coords

def getEmiterPosition(r, alpha):
    x = r * np.cos( np.radians(alpha)) + r
    y = r * np.sin( np.radians(alpha)) + r
    return (int(x),int(y))

def getSensorPosition(r, alpha, fi, i, numberOfSensors):
    x = r * np.cos (np.radians(alpha) + np.pi - np.radians(fi)/2 + (i * ( np.radians(fi) / (numberOfSensors-1) )) ) + r
    y = r * np.sin (np.radians(alpha) + np.pi - np.radians(fi)/2 + (i * ( np.radians(fi) / (numberOfSensors-1) )) ) + r
    return (int(x),int(y))   

def getAllSensors(r, alpha, fi, numberOfSensors):
    sensors = []
    for i in range(numberOfSensors):
        sensors.append(getSensorPosition(r, alpha, fi, i, numberOfSensors))
    return sensors

"""
Sinogram X -> sensor
         Y -> emiter
         
         To jest zrobione addytywnie, ale nie wiem czy to dobrze działa, trzeba sprawdzić
"""   
def sinogram(ALPHA,DETECTORS,r, L, image, ifFiltr):
    NUMBER_OF_EMITERS = int(360/ALPHA)
    sinogram = np.zeros([NUMBER_OF_EMITERS,DETECTORS])
    
    
    for i in range(NUMBER_OF_EMITERS):
        emiter = getEmiterPosition(r, ALPHA * i)
        sensors = getAllSensors(r, ALPHA * i, L, DETECTORS)
        for j,sensor in enumerate(sensors):
            misc = 0
            coords = bresenhamLine(emiter[0],emiter[1],sensor[0],sensor[1])
            for coord in  coords:
                misc += image[coord[0]-1][coord[1]-1]
            misc = misc/len(coords)
            sinogram[i][j] += misc
    if(ifFiltr):
        for k in range(sinogram.shape[0]):
            sinogram[k]=np.convolve(sinogram[k],get_filtr(), mode='same')

    
    sinogram = (sinogram - np.amin(sinogram)) / (np.amax(sinogram) - np.amin(sinogram))
    plt.imshow(sinogram, cmap='gray')
    return sinogram
    
def masked_normal(x):
    z = x * 1
    minval = np.amin(x[np.nonzero(x)])
    maxval = np.amax(x[np.nonzero(x)])
    print(minval, maxval)

    for i, k in enumerate(x):
        for j, l in enumerate(k):
            if l != 0:
                z[i][j] = (l-minval)/(maxval-minval)
            else:
                z[i][j] = 0
    return z

def get_filtr():
    filtr = [1]

    for i in range(1,21):
        if (i)%2:
            filtr.append(0)
        else:
            x=(-4/(math.pi)**2)/((i)**2)
            filtr.append(x)

    filtr =  filtr[:0:-1]+filtr
    return filtr


