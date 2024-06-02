import numpy as np
import cv2 as cv
import glob
 
h,w = 5,4

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


def calibrate_camera(frame):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (h,w), None)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(frame, (h,w), corners2, ret)
 
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Save to file
        np.savez('calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

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
        frame = cv.undistort(frame, mtx, dist, None, mtx)
 
    return frame

 


# Run when script is executed (not imported)
if __name__ == "__main__":

    # Initialize the camera and start capturing real-time images
    cap = cv.VideoCapture(0)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        
        # Detect faces in the captured frame
        corrected_frame = calibrate_camera(frame)
        
        # Display the detected frame
        cv.imshow('Camera Calibration', corrected_frame)
        
        # Check for the 'q' key to exit the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv.destroyAllWindows()