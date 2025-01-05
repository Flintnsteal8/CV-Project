version https://git-lfs.github.com/spec/v1
oid sha256:2ab4eb96d7bce406e426df4dacb958cf039b568d6e989f7379680e217cbd9bd0
size 734
---

# Drowsiness Detection System

This project is a real-time drowsiness detection application that uses computer vision techniques to monitor a user's eye activity. It determines whether the user is awake or drowsy based on their eye aspect ratio (EAR). The system is implemented using Python and utilizes libraries such as OpenCV, dlib, and Tkinter.

## Features

- **Real-Time Drowsiness Detection**: Continuously monitors and calculates the EAR from a live video feed to detect signs of drowsiness.
- **Customizable Thresholds**: Allows users to adjust the thresholds for detecting awake and drowsy states based on individual differences in eye sizes.
- **Visual and Audio Alerts**: Displays warnings and plays an alert sound when drowsiness is detected.
- **Interactive GUI**: Provides a user-friendly interface with real-time analytics and controls to start, pause, or resume detection.
- **Graphical Analytics**: Displays a live graph of the drowsiness score over time for better visualization.
- **Event Logging**: Logs timestamps and events (e.g., drowsiness detected) into a CSV file for future reference.

## Installation

### Prerequisites

Ensure you have Python installed along with the required libraries:

```bash
pip install scipy imutils pygame opencv-python-headless dlib matplotlib pillow
```

### Additional Files
1. **Pre-trained Model**: Download the `shape_predictor_68_face_landmarks.dat` file from the `models` directory.(shape_predictor_68_face_landmarks.dat.bz2)
2. **Alert Sound**: Add a `music.wav` file to the project directory. This file will be played as an alert sound when drowsiness is detected.

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Flintnsteal8/Drowsiness-Detection.git
   cd Drowsiness-Detection
   ```
2. Run the script:
   ```bash
   python drowsiness_detection.py
   ```

## User Interface

### Key Components
- **Live Video Feed**: Displays a real-time video feed with eye contours highlighted.
- **Drowsiness Score**: Indicates the user's eye aspect ratio. A higher score suggests alertness, while a lower score indicates potential drowsiness.
- **State Display**: Shows the current state as "Awake" (in green) or "Drowsy" (in red).
- **Threshold Settings**: Allows customization of drowsiness and awake thresholds.
- **Graph Analytics**: A live plot of drowsiness scores over time.

### Controls
- **Start**: Begin drowsiness detection.
- **Pause**: Temporarily pause detection.
- **Resume**: Resume detection after pausing.

## Technical Details

### Eye Aspect Ratio (EAR)
The EAR is calculated using the Euclidean distances between vertical and horizontal landmarks of the eyes:

\[
EAR = \frac{\text{(Distance1 + Distance2)}}{2 \times \text{Horizontal Distance}}
\]

### Detection Logic
- **Awake**: EAR > Awake Threshold
- **Drowsy**: EAR < Drowsy Threshold for a sustained period (default: 20 frames)

### Libraries Used
- **dlib**: Facial landmark detection.
- **OpenCV**: Video capture and frame processing.
- **Tkinter**: Graphical User Interface.
- **pygame**: Audio alert playback.
- **matplotlib**: Real-time graph plotting.

## Logging
Events are logged in `drowsiness_log.csv` with the following structure:
- **Timestamp**: Date and time of the event.
- **Event**: Description of the event (e.g., "Drowsy state detected").

## Customization
- **Thresholds**: Adjust awake and drowsy thresholds in the GUI.
- **Alert Sound**: Replace `music.wav` with any desired audio file.

## Notes
- Ensure proper lighting for accurate detection.
- Calibrate thresholds if the system produces false positives or negatives.

