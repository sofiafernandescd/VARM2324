import cv2

def superimpose_hat(image, x, y, w, h, hat='hat'):
    
    hat_img = cv2.imread(f'./masks/{hat}.png')
    #plt.imshow(hat_img)

    hat_h, hat_w, _ = hat_img.shape
    scale_factor = 1.5  # Adjust the scale factor as needed
    hat_width = int(w * scale_factor)
    hat_height = int(hat_width * (hat_h / hat_w))
    hat_img = cv2.resize(hat_img, (hat_width, hat_height))


    # Calculate the new y-coordinate for placing the hat on the top of the face
    new_y = y - int(0.60 * h)  # Adjust the percentage as needed
    new_x = x - int(0.25 * w)  # Adjust the percentage as needed

    # Iterate over the pixels of the hat image and copy them to the face image
    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if hat_img[i][j][k] > 0:
                    image[new_y + i][new_x + j][k] = hat_img[i][j][k]

    return image

def superimpose_glasses(image, x, y, w, h, glasses='eyeglasses'):

    glasses_img = cv2.imread(f'./masks/{glasses}.png', cv2.IMREAD_UNCHANGED)[:,:,:3]
    print(glasses_img.shape)
    #plt.imshow(glasses_img)

    glasses_h, glasses_w, _ = glasses_img.shape
    scale_factor = 1.1  # Adjust the scale factor as needed
    glasses_width = int(w * scale_factor)
    glasses_height = int(glasses_width * (glasses_h / glasses_w))
    glasses_img = cv2.resize(glasses_img, (glasses_width, glasses_height))

    # Calculate the new y-coordinate for placing the glasses on the face
    new_y = y + int(0.15 * h)  # Adjust the percentage as needed
    new_x = x - int(0.15 * w)  # Adjust the percentage as needed

    # Iterate over the pixels of the glasses image and copy them to the face image
    for i in range(glasses_height):
        for j in range(glasses_width):
            for k in range(3):
                #if glasses_img[i][j][k] < 235:
                if glasses_img[i][j][k] > 0:
                    image[new_y + i][new_x + j][k] = glasses_img[i][j][k]

    return image

def main():
    # Load the face image
    face_img = cv2.imread('image.png')

    # Detect faces in the image
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(face_img, 1.3, 5)

    for (x, y, w, h) in faces:
        # Superimpose the hat on the face
        face_img = superimpose_hat(face_img, x, y, w, h, hat='hat')

        # Superimpose the glasses on the face
        face_img = superimpose_glasses(face_img, x, y, w, h, glasses='blue-glasses')

    # Display the result
    cv2.imshow('Result', face_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the camera and start capturing real-time images
    cap = cv2.VideoCapture(0)

    while True:

        ret, face_img = cap.read()

        faces = face_detector.detectMultiScale(face_img, 1.3, 5)

        for (x, y, w, h) in faces:
            # Superimpose the hat on the face
            face_img = superimpose_hat(face_img, x, y, w, h, hat='hat')

            # Superimpose the glasses on the face
            face_img = superimpose_glasses(face_img, x, y, w, h, glasses='blue-glasses')

        cv2.imshow('Result', face_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
