import numpy as np
import cv2
import cv2.aruco as aruco
import pathlib


def calibrate_charuco_images(dirpath, image_format, marker_length, square_length):
    '''Apply camera calibration using aruco.
    The dimensions are in cm.
    '''
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000) # 1000
    board = aruco.CharucoBoard_create(4, 3, square_length, marker_length, aruco_dict)
    arucoParams = aruco.DetectorParameters_create()

    counter, corners_list, id_list = [], [], []
    img_dir = pathlib.Path(dirpath)
    first = 0
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*{image_format}'):
        print(f'using image {img}')
        image = cv2.imread(str(img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            img_gray, 
            aruco_dict, 
            parameters=arucoParams
        )

        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_gray,
            board=board
        )
        # If a Charuco board was found, let's collect image/corner points
        # Requiring at least 20 squares
        if resp > 20:
            # Add these corners and ids to our calibration arrays
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)
        else: 
            print(f'Not enough corners found in {img}')

    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_list, 
        charucoIds=id_list, 
        board=board, 
        imageSize=img_gray.shape, 
        cameraMatrix=None, 
        distCoeffs=None)
    
    # Save to file
    np.savez('calib_image_charuco.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    
    return [ret, mtx, dist, rvecs, tvecs]

def calibrate_charuco(frame, square_length=4.2, marker_length=3.1):

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) # 1000
    counter = np.zeros((2*2,3), np.float32)
    #board = aruco.CharucoBoard_create(4, 3, square_length, marker_length, aruco_dict)
    #arucoParams = aruco.DetectorParameters_create()
    #board = aruco.CharucoBoard((2, 2), square_length, marker_length, aruco_dict)
    board = aruco.GridBoard((2, 2), square_length, marker_length, aruco_dict)
    arucoParams = aruco.DetectorParameters()

    # Convert the frame to grayscale
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(
        img_gray, 
        aruco_dict, 
        parameters=arucoParams
    )
    # draw corners
    frame = aruco.drawDetectedMarkers(frame, corners, ids)

    print(f'{len(corners)} corners: {corners}')
    

    if len(corners) > 3:
        #resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        #     markerCorners=corners,
        #     markerIds=ids,
        #     image=img_gray,
        #     board=board
        # )


        #print(f'charuco_corners: {charuco_corners}')
        #print(f'charuco_ids: {resp}')

        
        
        # If a Charuco board was found, let's collect image/corner points
        # Requiring at least 20 squares
        # if resp > 4:
        #     # Add these corners and ids to our calibration arrays
        #     corners_list.append(charuco_corners)
        #     id_list.append(charuco_ids)

        # Actual calibration
        # ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        #     charucoCorners=corners,#corners_list, 
        #     charucoIds=ids, #id_list, 
        #     board=board, 
        #     imageSize=img_gray.shape, 
        #     cameraMatrix=None, 
        #     distCoeffs=None)

        
        
        # Draw and display the corners
        #cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
        #aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('frame', frame)

        return corners, ids

        
    else: 
        print(f'Not enough corners found')
        return None, None

 
    #return frame


if __name__ == '__main__':

    corners_list, id_list = [], []

    # Initialize the camera and start capturing real-time images
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        
        # Detect faces in the captured frame
        corners, ids = calibrate_charuco(frame, )

        if corners is not None:
            if len(corners) > 3:
                corners_list.append(corners)
                id_list.append(ids)

                marker_length, square_length = 3.1, 4.2
                aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
                counter = np.zeros((2*2,3), np.float32)
                board = aruco.GridBoard((2, 2), square_length, marker_length, aruco_dict)

                ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
                    corners=corners_list, 
                    ids=np.array(id_list), 
                    counter=counter,
                    board=board, 
                    imageSize=board.getGridSize(), 
                    cameraMatrix=None, 
                    distCoeffs=None
                )
    
                # Save to file
                np.savez('calib_image_charuco.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            
            
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
                #frame = cv2.undistort(frame, mtx, dist, None, mtx)
                
                # Display the detected frame
                #cv2.imshow('Camera Calibration ChArUco', corrected_frame)
        
        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()