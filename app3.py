import customtkinter as ctk
from tkinter import filedialog
import cv2
import threading
from ultralytics import YOLO
import os
import sys

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
model_path = os.path.join(base_path, "yolov5n.pt")

model = YOLO(model_path)

def process_video(video_path):
    status_label.configure(text="Processing video...")

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    paused = False

    cv2.namedWindow("Flood Victim Detection")

    def on_trackbar(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    cv2.createTrackbar("Timeline", "Flood Victim Detection", 0, total_frames, on_trackbar)

    while True:

        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Timeline", "Flood Victim Detection", current_frame)

            frame = cv2.resize(frame, (640, 360))

            results = model(frame)
            victim_detected = False

            for r in results:
                for box in r.boxes:

                    cls = int(box.cls[0])

                    if cls == 0:
                        victim_detected = True

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        cv2.putText(frame, f"Victim {conf:.2f}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2)

            if victim_detected:
                cv2.putText(frame, "VICTIM DETECTED!",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time = current_time_ms / 1000

        total_time = total_frames / fps if fps > 0 else 0

        time_text = f"{int(current_time//60):02}:{int(current_time%60):02} / {int(total_time//60):02}:{int(total_time%60):02}"

        cv2.putText(frame, time_text,
                    (20, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        if paused:
            cv2.putText(frame, "PAUSED",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        cv2.imshow("Flood Victim Detection", frame)


        key = cv2.waitKey(delay) & 0xFF

        if key == 27:  # ESC
            break

        elif key == 32:  # SPACE
            paused = not paused

        elif key == 83:  # RIGHT
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time + 5000)

        elif key == 81:  # LEFT
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, current_time - 5000))

    cap.release()
    cv2.destroyAllWindows()
    status_label.configure(text="Done ✔")

def upload_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    if file_path:
        status_label.configure(text="Loading video...")
        threading.Thread(target=process_video, args=(file_path,)).start()


app = ctk.CTk()
app.title("Flood Victim Detection System")
app.geometry("420x320")

title = ctk.CTkLabel(app,
                     text="Flood Victim Detection",
                     font=("Arial", 22, "bold"))
title.pack(pady=20)

upload_btn = ctk.CTkButton(app,
                           text="Upload Video",
                           command=upload_video,
                           width=200,
                           height=40)
upload_btn.pack(pady=10)

status_label = ctk.CTkLabel(app,
                            text="Waiting for video...",
                            font=("Arial", 14))
status_label.pack(pady=10)

info = ctk.CTkLabel(app,
                    text="Controls:\nSPACE = Pause\nESC = Exit\n← / → = Skip 5 sec\nDrag timeline to seek",
                    font=("Arial", 12))
info.pack(pady=20)

app.mainloop()