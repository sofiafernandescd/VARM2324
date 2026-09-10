import numpy as np
import cv2
import cv2.aruco as aruco
import datetime as dt


def calibrate_aruco(
        images, 
        marker_length=0.8, marker_separation=0.3,
        h=5, w=4, dictionary=aruco.DICT_6X6_250
        ):
    '''Apply camera calibration using aruco boards.
    The dimensions are in cm.
    '''
    aruco_dict = aruco.getPredefinedDictionary(dictionary) 
    arucoParams = aruco.DetectorParameters()
    board = aruco.GridBoard((h, w), marker_length, marker_separation, aruco_dict)

    counter, corners_list, id_list = [], [], []
    first = 0
    final_images = []

    # Find the ArUco markers inside each image
    for image in images:
        img_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            img_gray, 
            aruco_dict, 
            parameters=arucoParams
        )
        if ids is not None:
            if first == 0:
                corners_list = corners
                id_list = ids
            else:
                corners_list = np.vstack((corners_list, corners))
                id_list = np.vstack((id_list,ids))
            first = first + 1
            counter.append(len(ids))
            final_images.append(image)

            
        cv2.imshow('frame', image)

    counter = np.array(counter)
    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
        corners_list, 
        id_list,
        counter, 
        board, 
        img_gray.shape, 
        None, 
        None 
    )

    for img, corners, ids in zip(final_images, corners_list, id_list):
        #img = aruco.drawDetectedMarkers(img, corners, ids)
        # Estimate pose of each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        
        #for rvec, tvec in zip(rvecs, tvecs):
            #aruco.drawAxis(img, mtx, dist, rvec, tvec, 0.1)
            #aruco.drawDetectedMarkers(img, corners)
        
        #cv2.imshow('frame', img)
        print(ids, rvecs, tvecs)

    # Save to file
    np.savez('calibrations/calib_image_aruco_board.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    # Save to txt file
    date = dt.datetime.now()
    with open(f'calibrations/calib_image_aruco_board_{date}.txt', 'w') as f:
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

    return [ret, mtx, dist, rvecs, tvecs]



if __name__ == '__main__':

    h, w = 5,4
    dict_name = aruco.DICT_6X6_250

    images = []
    aruco_dict = aruco.getPredefinedDictionary(dict_name)#(aruco.DICT_6X6_250)
    arucoParams = aruco.DetectorParameters()

    # Initialize the camera and start capturing real-time images
    cap = cv2.VideoCapture(0)

    while len(images) < 100:
        # Capture a frame from the camera
        ret, frame = cap.read()
        images.append(frame)

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            img_gray, 
            aruco_dict, 
            parameters=arucoParams
        )
        # Draw the detected markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        # Show webcam stream
        cv2.imshow('Webcam stream', frame)
        
        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    calibrate_aruco(
        images, marker_length = 1.2,#0.7, #1.8, #0.8,
        marker_separation = 0.5,#0.3, #1.2, #0.5,
        h = h, w = w,
        dictionary = dict_name
        )

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()