import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import math
import time
import threading
import winsound

# =========================
# GLOBALS
# =========================
CHIN_THRESHOLD = 25
SHOULDER_THRESHOLD = 20
ANGLE_THRESHOLD = 15
ALERT_DELAY = 1.5

BASELINE_CHIN = None
BASELINE_SHOULDER = None
BASELINE_ANGLE = None

# >>> ADDED: TIMERS
program_start_time = None
bad_posture_total_time = 0.0
bad_posture_active_start = None
# <<< ADDED


# =========================
# ANGLE FUNCTION
# =========================
def angle(a, b, c):
    ang = math.degrees(
        math.atan2(a[1]-b[1], a[0]-b[0]) -
        math.atan2(c[1]-b[1], c[0]-b[0])
    )
    ang = abs(ang)
    return 360-ang if ang > 180 else ang


# =========================
# GUI WITH PREVIEW + OVERLAY
# =========================
def launch_gui():

    global BASELINE_CHIN, BASELINE_SHOULDER, BASELINE_ANGLE
    global CHIN_THRESHOLD, SHOULDER_THRESHOLD, ANGLE_THRESHOLD, ALERT_DELAY

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            label.after(10, update_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            nose = lm[mp_pose.PoseLandmark.NOSE]
            mouth = lm[mp_pose.PoseLandmark.MOUTH_LEFT]

            chin_y = ((nose.y + mouth.y) / 2) * h
            shoulder_y = shoulder.y * h

            neck_angle = angle(
                [ear.x, ear.y],
                [shoulder.x, shoulder.y],
                [elbow.x, elbow.y]
            )

            def px(p): return (int(p.x * w), int(p.y * h))

            ear_p = px(ear)
            shoulder_p = px(shoulder)
            elbow_p = px(elbow)
            chin_p = (int((nose.x + mouth.x)/2 * w), int(chin_y))

            # Draw nodes
            cv2.circle(frame, ear_p, 6, (255,0,0), -1)
            cv2.circle(frame, shoulder_p, 6, (0,255,0), -1)
            cv2.circle(frame, elbow_p, 6, (0,0,255), -1)
            cv2.circle(frame, chin_p, 6, (0,255,255), -1)

            # Draw lines
            cv2.line(frame, elbow_p, shoulder_p, (255,255,255), 2)
            cv2.line(frame, shoulder_p, ear_p, (255,255,255), 2)
            cv2.line(frame, shoulder_p, chin_p, (0,255,255), 2)
            cv2.line(frame, chin_p, ear_p, (0,255,255), 2)

            # Debug box
            cv2.rectangle(frame, (10,100), (260,200), (0,0,0), -1)

            # Debug text
            cv2.putText(frame, f"Chin Y: {int(chin_y)}", (20,130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            cv2.putText(frame, f"Shoulder Y: {int(shoulder_y)}", (20,160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.putText(frame, f"Angle: {int(neck_angle)}", (20,190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        frame = cv2.resize(frame, (900, 550))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        label.imgtk = imgtk
        label.configure(image=imgtk)

        label.after(10, update_frame)


    def calibrate_and_start():
        global BASELINE_CHIN, BASELINE_SHOULDER, BASELINE_ANGLE

        CHIN_THRESHOLD = chin_slider.get()
        SHOULDER_THRESHOLD = shoulder_slider.get()
        ANGLE_THRESHOLD = angle_slider.get()
        ALERT_DELAY = delay_slider.get()

        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            BASELINE_CHIN = ((lm[mp_pose.PoseLandmark.NOSE].y +
                              lm[mp_pose.PoseLandmark.MOUTH_LEFT].y) / 2) * h

            BASELINE_SHOULDER = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h

            BASELINE_ANGLE = angle(
                [lm[mp_pose.PoseLandmark.LEFT_EAR].x,
                 lm[mp_pose.PoseLandmark.LEFT_EAR].y],
                [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                 lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                 lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            )

            print("✅ Calibrated from GUI")

        cap.release()
        root.destroy()


    # Window setup
    root = tk.Tk()
    root.title("Posture Monitor Setup")

    root.state("zoomed")
    root.geometry("1200x800")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    tk.Label(root, text="Sit straight, then press Continue",
             font=("Arial", 16)).grid(row=0, column=0, pady=10)

    label = tk.Label(root)
    label.grid(row=1, column=0, sticky="nsew")

    controls = tk.Frame(root)
    controls.grid(row=2, column=0, sticky="ew", pady=10)
    controls.columnconfigure((0,1), weight=1)

    chin_slider = tk.Scale(controls, from_=10, to=50, orient="horizontal", label="Chin Threshold")
    chin_slider.set(25)
    chin_slider.grid(row=0, column=0, sticky="ew", padx=10)

    shoulder_slider = tk.Scale(controls, from_=10, to=50, orient="horizontal", label="Shoulder Threshold")
    shoulder_slider.set(20)
    shoulder_slider.grid(row=0, column=1, sticky="ew", padx=10)

    angle_slider = tk.Scale(controls, from_=5, to=30, orient="horizontal", label="Angle Threshold")
    angle_slider.set(15)
    angle_slider.grid(row=1, column=0, sticky="ew", padx=10)

    delay_slider = tk.Scale(controls, from_=1, to=5, resolution=0.5,
                            orient="horizontal", label="Alert Delay")
    delay_slider.set(1.5)
    delay_slider.grid(row=1, column=1, sticky="ew", padx=10)

    ttk.Button(root, text="Continue / Calibrate",
               command=calibrate_and_start).grid(row=3, column=0, pady=15)

    update_frame()
    root.mainloop()


# Run GUI first
launch_gui()


# =========================
# ORIGINAL POSTURE MONITOR
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

baseline_chin_y = BASELINE_CHIN
baseline_shoulder_y = BASELINE_SHOULDER
baseline_angle = BASELINE_ANGLE

slouch_start = None


program_start_time = time.time()


def beep():
    winsound.Beep(1200, 300)

print("📏 Monitoring started (already calibrated)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.pose_landmarks:

        lm = res.pose_landmarks.landmark

        ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
        shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        nose = lm[mp_pose.PoseLandmark.NOSE]
        mouth = lm[mp_pose.PoseLandmark.MOUTH_LEFT]

        chin_y = ((nose.y + mouth.y) / 2) * h
        shoulder_y = shoulder.y * h

        neck_angle = angle(
            [ear.x, ear.y],
            [shoulder.x, shoulder.y],
            [elbow.x, elbow.y]
        )

        def px(p): return (int(p.x * w), int(p.y * h))

        ear_p = px(ear)
        shoulder_p = px(shoulder)
        elbow_p = px(elbow)
        chin_p = (int((nose.x + mouth.x)/2 * w), int(chin_y))

        chin_bad = baseline_chin_y is not None and chin_y > baseline_chin_y + CHIN_THRESHOLD
        shoulder_bad = baseline_shoulder_y is not None and shoulder_y > baseline_shoulder_y + SHOULDER_THRESHOLD
        angle_bad = baseline_angle is not None and neck_angle < baseline_angle - ANGLE_THRESHOLD

        chin_color = (0,255,255)
        shoulder_color = (0,255,0)
        angle_color = (255,255,255)

        if chin_bad:
            chin_color = (0,0,255)
        if shoulder_bad:
            shoulder_color = (0,0,255)
        if angle_bad:
            angle_color = (0,0,255)

        cv2.circle(frame, ear_p, 6, (255,0,0), -1)
        cv2.circle(frame, shoulder_p, 6, shoulder_color, -1)
        cv2.circle(frame, elbow_p, 6, (0,0,255), -1)
        cv2.circle(frame, chin_p, 6, chin_color, -1)

        cv2.line(frame, elbow_p, shoulder_p, angle_color, 2)
        cv2.line(frame, shoulder_p, ear_p, angle_color, 2)
        cv2.line(frame, shoulder_p, chin_p, chin_color, 2)
        cv2.line(frame, chin_p, ear_p, chin_color, 2)

        cv2.putText(frame, f"Chin Y: {int(chin_y)}", (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, chin_color, 2)

        cv2.putText(frame, f"Shoulder Y: {int(shoulder_y)}", (20,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, shoulder_color, 2)

        cv2.putText(frame, f"Angle: {int(neck_angle)}", (20,180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, angle_color, 2)

        bad_posture = chin_bad or shoulder_bad or angle_bad

        #POSTURE TIMER LOGIC (non-invasive)
        if bad_posture:
            if bad_posture_active_start is None:
                bad_posture_active_start = time.time()
        else:
            if bad_posture_active_start is not None:
                bad_posture_total_time += time.time() - bad_posture_active_start
                bad_posture_active_start = None


        if bad_posture:
            cv2.putText(frame, "BAD POSTURE", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            if slouch_start is None:
                slouch_start = time.time()
            elif time.time() - slouch_start > ALERT_DELAY:
                threading.Thread(target=beep).start()
                slouch_start = None
        else:
            cv2.putText(frame, "GOOD POSTURE", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            slouch_start = None

    cv2.imshow("Posture Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break


#finalize timer on exit
if bad_posture_active_start is not None:
    bad_posture_total_time += time.time() - bad_posture_active_start


cap.release()
cv2.destroyAllWindows()


#SUMMARY WINDOW
def show_summary():
    root = tk.Tk()
    root.title("Session Summary")
    root.geometry("400x250")

    runtime = time.time() - program_start_time
    bad_time = bad_posture_total_time
    percent = (bad_time / runtime * 100) if runtime > 0 else 0

    tk.Label(root, text="Posture Summary", font=("Arial", 16, "bold")).pack(pady=10)
    tk.Label(root, text=f"Total Runtime: {runtime:.1f} sec").pack(pady=5)
    tk.Label(root, text=f"Bad Posture Time: {bad_time:.1f} sec").pack(pady=5)
    tk.Label(root, text=f"Bad Posture %: {percent:.1f}%").pack(pady=10)

    ttk.Button(root, text="Close", command=root.destroy).pack(pady=10)

    root.mainloop()

show_summary()
