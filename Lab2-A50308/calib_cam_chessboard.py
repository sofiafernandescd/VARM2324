import numpy as np
import cv2 as cv
import datetime as dt

def undistort_image(frame, mtx, dist):
    # Undistort the frame
    frame_undistorted = cv.undistort(frame, mtx, dist, None, mtx)
    return frame_undistorted

def calibrate_camera(
        frames,
        h=5,
        w=4,
        criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    ):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    
    for i,frame in enumerate(frames):
        print(f"Calibrating camera with image {i}...")
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (h,w), None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            #corners2 = cv.cornerSubPix(gray, corners, (h*2,w*2), (-1,-1), criteria)
            corners2 = cv.cornerSubPix(gray, corners, (20,20), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(frame, (h,w), corners2, ret)

            #imgpoints.append(corners)
            #cv.drawChessboardCorners(frame, (h,w), corners, ret)
        else:
            print(f"Image {i} failed for calibration.")
 
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save to file
    np.savez('calibrations/calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    # Save results to txt file
    with open(f'calibrations/calib_{dt.datetime.now().strftime("%d%m-%H:%M:%S")}.txt', 'w') as f:
        f.write(f"Camera matrix: {mtx}\n")
        f.write(f"Distortion coefficients: {dist}\n")
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
    frames_undistorted = [undistort_image(frame, mtx, dist) for frame in frames]

    return frames_undistorted




# Run when script is executed (not imported)
if __name__ == "__main__":
    h,w = 3,3

    calib_imgs, all_imgs = [], []

    # Initialize the camera and start capturing real-time images
    cap = cv.VideoCapture(0)

    while len(calib_imgs) < 10 and len(all_imgs) < 100:
        # Capture a frame from the camera
        ret, frame = cap.read()
        all_imgs.append(frame)
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (h,w), None)
        # Make a copy of the frame
        detect_frame = frame.copy()

        # If found, add object points, image points (after refining them)
        if ret == True:
            calib_imgs.append(frame)
            print(f"Image {len(calib_imgs)} captured.")
            
            # Draw and display the corners
            detect_frame = cv.drawChessboardCorners(detect_frame, (h,w), corners, ret)
        cv.imshow('Camera Calibration', detect_frame)
        # Check for the 'q' key to exit the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
                
    # Calibrate the camera
    corrected_frames = calibrate_camera(all_imgs, h, w)
    # Read parameters from file
    data = np.load('calibrations/calib.npz')
    mtx, dist = data['mtx'], data['dist']
        
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        
        # Detect faces in the captured frame
        corrected_frame = undistort_image(frame, mtx, dist)
        
        # Display the detected frame
        cv.imshow('Undistorted Camera', corrected_frame)
        
        # Check for the 'q' key to exit the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv.destroyAllWindows()