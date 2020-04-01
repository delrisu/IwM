from timeit import default_timer as timer
import matplotlib.pyplot as pltimer 
import numpy as np
import cv2
from sinogram import *
def reverseSinogram(ALPHA,DETECTORS,r,R_org, L, sinogram,im):
    NUMBER_OF_EMITERS = int(360/ALPHA)
    start = timer()
    gif = []
    x_range = range(int(R_org/2),im.shape[0]-int(R_org/2))
    y_range = range(int(R_org/2),im.shape[1]-int(R_org/2))
    revSin = np.zeros([im.shape[0], im.shape[1]])
    counter = np.zeros([im.shape[0], im.shape[1]])
    for i in range(NUMBER_OF_EMITERS):
        emiter = getEmiterPosition(r, ALPHA * i)
        sensors = getAllSensors(r, ALPHA * i, L, DETECTORS)
        for j,sensor in enumerate(sensors):
            coords = bresenhamLine(sensor[0],sensor[1],emiter[0],emiter[1])
            for coord in  coords:
                if coord[0]-1 in x_range and coord[1]-1 in y_range:
                    revSin[coord[0]-1][coord[1]-1] += sinogram[i][j]
                    counter[coord[0]-1][coord[1]-1]  += 1
        
        if(i%90==0):
            gif.append(revSin*1)
            cv2.imwrite("gif"+str(i)+".png",revSin[int(R_org/2):-int(R_org/2),int(R_org/2):-int(R_org/2)])

    gif.append(revSin[int(R_org/2):-int(R_org/2),int(R_org/2):-int(R_org/2)]*1)
    cv2.imwrite("gif"+str(NUMBER_OF_EMITERS)+".png",revSin[int(R_org/2):-int(R_org/2),int(R_org/2):-int(R_org/2)])
    for x in range(revSin.shape[0]):
        if x in x_range:
            for y in range(revSin.shape[1]):
                if y in y_range:
                    if counter[x][y]!=0:
                        revSin[x][y] = revSin[x][y]/counter[x][y]

    img = revSin[int(R_org/2):-int(R_org/2),int(R_org/2):-int(R_org/2)]*1
    blur = cv2.GaussianBlur(img,(7,7),0)
    normalize = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #enhanced = cv2.equalizeHist(normalize.astype(np.uint8))

    gif.append(normalize)
    cv2.imwrite("gif_FILTERED.png",normalize)
    plt.imshow(normalize, cmap='gray')
    print("time: ",timer()-start)
    return gif 
