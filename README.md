# Artificial Vision & Mixed Reality

> Practical computer vision and mixed-reality projects developed during the MSc in Computer Engineering and Multimedia.

This repository contains laboratory work and project implementations developed for the **Artificial Vision and Mixed Reality** course during my Master's degree.

The work explores several practical computer vision problems, from camera calibration and fiducial markers to real-time face detection and facial feature processing.

Rather than being a single application, the repository documents a collection of experiments and implementations developed throughout the course.

---

## Highlights

### Camera calibration & fiducial markers

The `Lab2` work explores camera calibration and marker-based computer vision, including:

* Camera calibration
* Calibration image processing
* **ArUco boards and markers**
* ChArUco calibration
* Camera parameter estimation
* Marker generation and detection

The repository includes both Python implementations and calibration data/results used during the experiments.

### Face detection

The `varm` project includes real-time face detection using **OpenCV** and Haar Cascade classifiers.

The implementation captures frames directly from a camera, converts them to grayscale and detects faces using OpenCV's `detectMultiScale` pipeline before displaying the detected regions in real time.

### Facial analysis

The project also contains material related to facial image processing, including normalized face images, facial landmark models and supporting datasets/assets.

---

## Selected Results

<!-- Replace these placeholders with screenshots from the repository -->

### Camera calibration & ArUco

![Camera calibration and ArUco markers](./assets/camera-calibration.png)

> **Screenshot to add:** Show the calibration setup, ArUco/ChArUco board or an example of detected markers. This is probably the strongest visual for demonstrating the computer-vision work in `Lab2`.

### Real-time face detection

![Real-time face detection](./assets/face-detection.png)

> **Screenshot to add:** Run the face-detection implementation with a webcam and take a screenshot showing the detected face bounding boxes. This makes the project immediately understandable without reading the code.

---

## Technical Work

The repository demonstrates practical experience with:

* **Python**
* **OpenCV**
* **NumPy**
* Computer vision pipelines
* Camera calibration
* ArUco / ChArUco markers
* Fiducial marker detection
* Real-time image processing
* Face detection
* Facial image normalization
* Jupyter notebooks
* Camera-based computer vision

The implementations combine standalone Python scripts with notebooks and supporting image/calibration data.

---

## Repository Structure

```text
VARM2324/
│
├── Lab2/
│   ├── Camera calibration
│   ├── ArUco / ChArUco calibration
│   ├── Marker generation
│   ├── Calibration data
│   └── lab2.ipynb
│
├── varm/
│   ├── Face detection
│   ├── Facial image processing
│   ├── Facial landmarks
│   ├── Computer vision exercises
│   └── project1.ipynb
│
└── README.md
```

The repository structure reflects the progression from individual laboratory exercises towards larger computer-vision experiments.


---

## Academic Context

**Course:** Artificial Vision and Mixed Reality
**Degree:** MSc in Computer Engineering and Multimedia
**Institution:** ISEL
**Academic year:** 2023–2024

This repository contains coursework developed as part of the Master's programme.

---

## Author

**Sofia Fernandes Condesso**

Data Scientist / ML Engineer

* GitHub: [@sofiafernandescd](https://github.com/sofiafernandescd)

---

## License & Academic Attribution

This repository is primarily an **academic record and portfolio of coursework**.

Some materials originate from or build upon course-provided resources and external libraries, datasets, models or assets. Their respective authors and licenses remain applicable.

