from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
from datetime import datetime


mixer.init()
mixer.music.load('music.wav')


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 30

        # creates a toplevel window
        self.tooltip = tk.Toplevel(self.widget)

        # Leaves only the label and removes the app window
        self.tooltip.wm_overrideredirect(True)

        self.tooltip.wm_geometry(f"+{x}+{y}")

        tk.Label(self.tooltip, text=self.text, background="#ffffff", relief='solid', borderwidth=1).grid(row=0, column=0)

    def on_leave(self, _):
        if self.tooltip:
            self.tooltip.destroy()


class DrowsinessApp:
    def __init__(self, root, detect, predict):
        self.root = root
        self.root.title("Drowsiness Detection")

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.ear_label = tk.Label(root, text="Drowsiness Score(higher score= more awake): ", font=("Helvetica", 12))
        self.ear_label.pack()
        ToolTip(self.ear_label, "The Drowsiness Score is a measure of eye openness.It indicates the drowsiness level of the driver.The average indication for awake state is 0.3 and above and 0.25 and below for drowsy state")




        self.state_label = tk.Label(root, text="State: Awake", font=("Helvetica", 14, "bold"), fg="green")
        self.state_label.pack()
        ToolTip(self.state_label, "Current state of the driver, indicating whether they are awake or drowsy.")

        self.start_button = tk.Button(root, text="Start Drowsiness Detection", command=self.start_detection)
        self.start_button.pack()

        self.pause_button = tk.Button(root, text="Pause Detection", command=self.pause_detection)
        self.pause_button.pack()

        self.resume_button = tk.Button(root, text="Resume Detection", command=self.resume_detection)
        self.resume_button.pack()

        self.threshold_frame = ttk.LabelFrame(root, text="Range Settings. Adjust if your eyes are smaller or bigger")
        self.threshold_frame.pack(padx=10, pady=10)

        self.drowsy_thresh_var = tk.DoubleVar(value=0.25)
        self.awake_thresh_var = tk.DoubleVar(value=0.3)

        ttk.Label(self.threshold_frame, text="Drowsy Range:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        drowsy_entry = ttk.Entry(self.threshold_frame, textvariable=self.drowsy_thresh_var)
        drowsy_entry.grid(row=0, column=1, padx=5, pady=5)
        ToolTip(drowsy_entry, "Drowsy Threshold: Set the warning level for detecting signs of drowsiness. "
                              "Lower values make it more sensitive, while higher values make it less sensitive.")

        ttk.Label(self.threshold_frame, text="Awake Range:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        awake_entry = ttk.Entry(self.threshold_frame, textvariable=self.awake_thresh_var)
        awake_entry.grid(row=1, column=1, padx=5, pady=5)
        ToolTip(awake_entry, "Awake Threshold: Set the safety level to recognize when the user is fully awake. "
                             "Adjust based on personal comfort. Extremely low values may trigger false alerts, and extremely high "
                             "values may not detect early signs of tiredness.")

        self.cap = cv2.VideoCapture(0)
        self.detect = detect
        self.predict = predict
        self.flag = 0
        self.detection_active = False
        self.is_paused = False

        # Data Logging
        self.log_file = open('drowsiness_log.csv', 'a', newline='')
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(['Timestamp', 'Event'])

        # Real-time Graph
        self.fig, self.ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_title('Drowsiness score Over Time graph analytics')
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Score')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.canvas.draw()

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def start_detection(self):
        self.detection_active = True
        self.is_paused = False
        self.update()

    def pause_detection(self):
        self.is_paused = True

    def resume_detection(self):
        self.is_paused = False

    def update(self):
        if not self.detection_active or self.is_paused:
            self.root.after(10, self.update)
            return

        ret, frame = self.cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 0)

        for subject in subjects:
            shape = self.predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            self.ear_label.config(text=f"Drowsiness Score: {ear:.2f}")

            if ear < self.drowsy_thresh_var.get() and not self.is_paused:
                self.flag += 1
                if self.flag >= 20:
                    self.state_label.config(text="State: Drowsy", fg="red")
                    cv2.putText(frame, "****************DROWSY!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************DROWSY!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            elif ear > self.awake_thresh_var.get() and not self.is_paused:
                self.flag = 0
                self.state_label.config(text="State: Awake", fg="green")

                # Update the graph
                self.line.set_xdata([*self.line.get_xdata(), len(self.line.get_xdata()) + 1])
                self.line.set_ydata([*self.line.get_ydata(), ear])
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw()

        # Convert the frame to RGB for displaying in tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = imutils.resize(img, width=400)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        self.video_label.img = img
        self.video_label.config(image=img)

        self.root.after(10, self.update)

    def log_event(self, event):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_writer.writerow([timestamp, event])
        self.log_file.flush()

    def __del__(self):
        # Release resources
        self.cap.release()
        self.log_file.close()


if __name__ == "__main__":
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    root = tk.Tk()
    app = DrowsinessApp(root, detect, predict)
    root.mainloop()
