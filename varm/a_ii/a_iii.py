import cv2
import dlib # conda install -c conda-forge dlib   

# Load pre-trained face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.xml')

# Read an image and detect faces with facial landmarks
img = cv2.imread('image.png')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
detector.setImage(rgb)
facelocs = detector.run()
for loc in facelocs:
    shape = predictor(rgb, loc)
    for pt in shape.part(dlib.shape_pixel):
        cv2.circle(img, (pt.x, pt.y), 1, (0, 255, 0), -1)

# Display the result
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)