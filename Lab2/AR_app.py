'''
 # @ Author: Group 1
 # @ Description:
 Develop a DIP application for face images that deals with images in gray-scale levels and with 
 color images, as follows:
(i) The input image is provided by a file or from the PC Webcam. 
(ii) Display the input and the output images and their histograms.
(iii) The output image is written on a PNG and on a JPEG output file. 

The application provides the following functionalities:
(i) Identity hiding - The output image is a face image, such that the presence of a person is noticeable, but not possible to identify it.
(ii) Contrast adjustment - Automatic contrast adjustment on the input image.
(iii) Negative version - The output image is the negative version of the input image.
(iv) Eyes and mouth detection - The output image exibits squares or rectangles superimposed over the eyes and the mouth.
(v) JPEG lossy compression - The output image is JPEG encoded with different levels of quality.
 '''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time


def read_image(img_file):

    # read image
    img = cv2.imread(img_file)
    # convert to RGB
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def take_batch_pictures():
    # initialize the camera
    cap = cv2.VideoCapture(0)
    batch_imgs = []
    while len(batch_imgs) < 10:
        time.sleep(1)
        # capture frame-by-frame
        ret, frame = cap.read()
        # convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_imgs.append(frame)
        # display the resulting frame
        cv2.imshow('frame', frame)
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return batch_imgs


def take_picture():
    # initialize the camera
    cap = cv2.VideoCapture(0)
    while(True):
        # capture frame-by-frame
        ret, frame = cap.read()
        # display the resulting frame
        cv2.imshow('frame', frame)
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame



def process_image(img):
    if img is not None:
        st.subheader("Input Image")
        st.image(img, channels="BGR", use_column_width=True)
        # histograms
        st.subheader("Histogram of Input Image")
        show_histogram(img, "Input Image")

        st.subheader("Identity Hidden")
        img_blurred = blur_face(img)
        st.image(img_blurred, channels="BGR", use_column_width=True)
        # buttons to save image
        if st.button("Save Identity Hidden to PNG"):
            save_png(img, "output.png", 9)
        if st.button("Save Identity Hidden to JPEG"):
            save_jpeg(img, "output.jpg", 50)
        # histograms
        st.subheader("Histogram of Identity Hidden")
        show_histogram(img_blurred, "Identity Hidden")
        
        st.subheader("Negative Image")
        img_neg = negative(img)
        st.image(img_neg, channels="BGR", use_column_width=True)
        # buttons to save image
        if st.button("Save Negative Image to PNG"):
            save_png(img, "output.png", 9)
        if st.button("Save Negative Image to JPEG"):
            save_jpeg(img, "output.jpg", 50)
        # histograms
        st.subheader("Histogram of Negative Image")
        show_histogram(img_neg, "Negative Image")

        st.subheader("Eyes and Mouth Detection")
        img_eyes_mouth = detect_eyes_mouth(img)
        st.image(img_eyes_mouth, channels="BGR", use_column_width=True)
        # buttons to save image
        if st.button("Save Eyes and Mouth to PNG"):
            save_png(img, "output.png", 9)
        if st.button("Save Eyes and Mouth to JPEG"):
            save_jpeg(img, "output.jpg", 50)
        # histograms
        st.subheader("Histogram of Eyes and Mouth Detection")
        show_histogram(img_eyes_mouth, "Eyes and Mouth Detection")

        st.subheader("Contrast Adjusted")
        img_cont = contrast(img)
        st.image(img_cont, channels="BGR", use_column_width=True)
        # buttons to save image
        if st.button("Save Contrast Adjusted to PNG"):
            save_png(img, "output.png", 9)
        if st.button("Save Contrast Adjusted to JPEG"):
            save_jpeg(img, "output.jpg", 50)
        # histograms
        st.subheader("Histogram of Contrast Adjusted")
        show_histogram(img_cont, "Contrast Adjusted")

        st.subheader("Contrast and Brightness Adjusted")
        img_bright_contrast = automatic_brightness_and_contrast(img)
        st.image(img_bright_contrast, channels="BGR", use_column_width=True)
        # buttons to save image
        if st.button("Save Auto Bright-Contrast to PNG"):
            save_png(img, "output.png", 9)
        if st.button("Save Auto Bright-Contrast to JPEG"):
            save_jpeg(img, "output.jpg", 50)
        # histograms
        st.subheader("Histogram of Contrast and Brightness Adjusted")
        show_histogram(img_bright_contrast, "Contrast and Brightness Adjusted")

        st.subheader("JPEG Compression")
        # quality slider
        quality = st.slider("JPEG Compression Quality", min_value=0, max_value=100, value=50)
        img_jpeg = jpeg_compression(img, quality)
        st.image(img_jpeg, channels="BGR", use_column_width=True)
        # buttons to save image
        if st.button("Save JPEG Compression to PNG"):
            save_png(img, "output.png", 9)
        if st.button("Save JPEG Compression to JPEG"):
            save_jpeg(img, "output.jpg", quality)
        # histograms
        st.subheader("Histogram of JPEG Compression")
        show_histogram(img_jpeg, "JPEG Compression")

        st.subheader("PNG Compression")
        # quality slider
        quality = st.slider("PNG Compression Quality", min_value=0, max_value=9, value=0)
        img_png = png_compression(img, quality)
        st.image(img_png, channels="BGR", use_column_width=True)
        # buttons to save image
        if st.button("Save PNG Compression to PNG"):
            save_png(img, "output.png", quality)
        if st.button("Save PNG Compression to JPEG"):
            save_jpeg(img, "output.jpg", 50)
        # histograms
        st.subheader("Histogram of PNG Compression")
        show_histogram(img_png, "PNG Compression")

def main():

    """Digital Image Processing App with Streamlit
    """
    st.title("Digital Image Processing App")
    st.sidebar.header("Select an option")

    # option of uploading an image or taking a picture with webcam
    option = st.sidebar.selectbox(
        "Choose an option:",
        ("Calibrate with Chessboard", "Calibrate with ArUco board" , "Upload Image", "Take a Picture with Webcam")
    )

    # calibrate with chessboard
    if option == "Calibrate with Chessboard":
        st.write("Click the 'Take Picture' button to capture an image from your webcam.")
        if st.button("Calibrate with Chessboard"):
            batch_imgs = take_batch_pictures()
            process_image(img)
    # calibrate with ArUco board
    elif option == "Calibrate with ArUco board":
        pass
    # file uploader
    elif option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_image is not None:
            img = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
            process_image(img)
    # webcam picture
    elif option == "Take a Picture with Webcam":
        st.write("Click the 'Take Picture' button to capture an image from your webcam.")
        if st.button("Take Picture"):
            img = take_picture()
            process_image(img)

if __name__ == "__main__":
    main()

