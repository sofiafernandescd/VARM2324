import cv2
import cv2.aruco as aruco
import numpy as np

def load_calibration_parameters(file_path='calib_image_aruco_board1.npz'):
    with np.load(file_path) as data:
        mtx = data['mtx']
        dist = data['dist']
    return mtx, dist

def draw_virtual_objects(frame, rvecs, tvecs, mtx, dist, ids):
    axis_length = 0.05

    for i in range(len(ids)):
        # Draw a simple cube for marker ID 0
        if ids[i] == 11:
            points = np.float32([
                [0, 0, 0], [axis_length, 0, 0], [axis_length, axis_length, 0], [0, axis_length, 0],
                [0, 0, -axis_length], [axis_length, 0, -axis_length], [axis_length, axis_length, -axis_length], [0, axis_length, -axis_length]
            ]).reshape(-1, 3)
            

            imgpts, _ = cv2.projectPoints(points, rvecs[i], tvecs[i], mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Draw ground floor in green
            frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), -3)

            # Draw pillars in blue
            for j in range(4):
                frame = cv2.line(frame, tuple(imgpts[j]), tuple(imgpts[j + 4]), (255, 0, 0), 3)

            # Draw top layer in red
            frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 3)

        # Add more virtual objects for different marker IDs as needed

    return frame

def detect_and_register_virtual_objects(mtx, dist):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.04, mtx, dist)

            for i in range(len(ids)):
                aruco.drawDetectedMarkers(frame, corners)
                #aruco.drawAxis(frame, mtx, dist, rvecs[i], tvecs[i], 0.05)
                frame = draw_virtual_objects(frame, rvecs, tvecs, mtx, dist, ids)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load calibration parameters
mtx, dist = load_calibration_parameters()

# Detect ArUco markers and register virtual objects
detect_and_register_virtual_objects(mtx, dist)
