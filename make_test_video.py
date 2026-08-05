import cv2
import numpy as np

W, H, FPS, SEC = 960, 540, 25, 6
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
cap = cv2.VideoWriter("test_input.mp4", fourcc, FPS, (W, H))

plate_texts = ["Dhaka Metro La 12-3456", "Chattogram Kha 11-2290", "Sylhet Da 24-1001"]

for f in range(FPS * SEC):
    frame = np.full((H, W, 3), 90, np.uint8)
    car_id = (f // (FPS * 2)) % len(plate_texts)
    x = 120 + (f * 3) % (W - 600)
    cv2.rectangle(frame, (x, 260), (x + 420, 420), (30, 30, 30), -1)
    cv2.rectangle(frame, (x + 60, 200), (x + 280, 260), (40, 40, 45), -1)
    cv2.circle(frame, (x + 90, 420), 30, (20, 20, 20), -1)
    cv2.circle(frame, (x + 330, 420), 30, (20, 20, 20), -1)
    px, py, pw, ph = x + 130, 300, 180, 40
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (200, 200, 20), -1)
    cv2.putText(frame, plate_texts[car_id], (px + 8, py + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cap.write(frame)

cap.release()
print("test_input.mp4 written")
