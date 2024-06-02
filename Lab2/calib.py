import numpy as np
import cv2 as cv
import glob
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


def calibrate_camera(frame):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
 
    # Draw and display the corners
    cv.drawChessboardCorners(frame, (7,6), corners2, ret)
 
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save to file
    np.savez('calib_image.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    print("Camera matrix: ", mtx)
    print("Distortion coefficients: ", dist)
    print("rvecs: ", rvecs)
    print("tvecs: ", tvecs)

    # Save fx, fy and cx, cy
    print("fx: ", mtx[0,0])
    print("fy: ", mtx[1,1])
    print("cx: ", mtx[0,2])
    print("cy: ", mtx[1,2])

    # Save distortion coefficients
    print("k1: ", dist[0,0])
    print("k2: ", dist[0,1])
    print("p1: ", dist[0,2])
    print("p2: ", dist[0,3])
    print("k3: ", dist[0,4])

    
 
    # Undistort the frame
    corrected_frame = cv.undistort(frame, mtx, dist, None, mtx)
 
    return corrected_frame


images = glob.glob('*.jpeg')
 
for fname in images:
    img = cv.imread(fname)
    corrected_img = calibrate_camera(img)
    cv.imshow(fname, img)
    cv.waitKey(2000)
 
cv.destroyAllWindows()