'''
 # @ Author: Sofia Condesso - A50308
 # @ Description:
 '''


import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt



# REF: https://www.makeuseof.com/python-aruco-marker-generator-how-create/


def create_aruco_marker(
        dictionary=aruco.DICT_6X6_250, 
        marker_id=1, 
        sidePixels=8
    ):
    """Create an ArUco marker with the given parameters."""

    # Get the current date and time
    date = dt.datetime.now().strftime("%d%m-%H:%M:%S")
    # Get the predefined dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    # Generate the image marker
    img = aruco.generateImageMarker(dictionary=aruco_dict, id=marker_id, sidePixels=sidePixels)
    # Display and save the marker
    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    plt.axis("off")
    plt.savefig(f"markers/marker_{date}.jpeg")
    plt.show()

def create_aruco_markers_board(
        dictionary=aruco.DICT_7X7_50,#aruco.DICT_6X6_250,
        nx=5,
        ny=3,
        sidePixels=9
    ):
    """Create an ArUco marker board with the given parameters."""
    # Get the current date and time
    date = dt.datetime.now().strftime("%d%m-%H:%M:%S")
    # Get the predefined dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    # Create the board with ArUco markers
    fig = plt.figure()
    for i in range(1, nx*ny+1):
        ax = fig.add_subplot(ny,nx, i)
        img = aruco.generateImageMarker(dictionary=aruco_dict, id=i, sidePixels=sidePixels)
        plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
        ax.axis("off")

    plt.savefig(f"markers/markers{nx}x{ny}_{date}.jpeg")
    plt.show()

# TODO: Implement this function
# def create_charuco_markers():
#     # REF: https://www.makeuseof.com/python-aruco-marker-generator-how-create/

#     #aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#     aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

#     #img = aruco.drawMarker(aruco_dict, 1, 700)
#     img = aruco.generateImageMarker(dictionary=aruco_dict, id=1, sidePixels=8)

#     plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
#     plt.axis("off")

#     plt.savefig("marker.jpeg")
#     plt.show()

if __name__ == '__main__':
    """Running this script will create an ArUco marker and an ArUco marker board.
    TODO: Interaction in the shell"""
    create_aruco_markers_board()
    create_aruco_marker()