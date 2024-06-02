import numpy as np
import cv2 #, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
#import pandas as pd
#%matplotlib nbagg

def create_aruco_markers_board():
    # REF: https://www.makeuseof.com/python-aruco-marker-generator-how-create/

    #aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    fig = plt.figure()
    nx = 5
    ny = 4
    for i in range(1, nx*ny+1):
        ax = fig.add_subplot(ny,nx, i)
        #img = aruco.drawMarker(aruco_dict,i, 700)
        img = aruco.generateImageMarker(dictionary=aruco_dict, id=i, sidePixels=8)
        plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
        ax.axis("off")

    plt.savefig("markers5x4.jpeg")
    plt.show()

def create_aruco_markers():
    # REF: https://www.makeuseof.com/python-aruco-marker-generator-how-create/

    #aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    #img = aruco.drawMarker(aruco_dict, 1, 700)
    img = aruco.generateImageMarker(dictionary=aruco_dict, id=1, sidePixels=8)

    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    plt.axis("off")

    plt.savefig("marker.jpeg")
    plt.show()

if __name__ == '__main__':
    create_aruco_markers_board()
    create_aruco_markers()