import json

import cv2
from screeninfo import get_monitors
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv as c

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
               (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
               (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
               (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
               (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
               (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
               (29, 31), (30, 32), (27, 31), (28, 32)]
CSV_KEYS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


visible_side = None
points = []

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = PoseLandmarker.create_from_options(options)

csv = []

cap = cv2.VideoCapture('input/input2.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0
frame_index = 0
SCREEN_WIDTH, SCREEN_HEIGHT = get_monitors()[0].width, get_monitors()[0].height
video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if video_w > 0 and video_h > 0:
    target_w = max(1, int(video_w * 0.5))
    target_h = max(1, int(video_h * 0.5))
else:
    target_w, target_h = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2


if target_w > SCREEN_WIDTH or target_h > SCREEN_HEIGHT:
    fit_scale = min(SCREEN_WIDTH / target_w, SCREEN_HEIGHT / target_h)
    target_w = max(1, int(target_w * fit_scale))
    target_h = max(1, int(target_h * fit_scale))

cv2.namedWindow('Stridelytics', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('Stridelytics', target_w, target_h)
cv2.moveWindow('Stridelytics', (SCREEN_WIDTH-target_w) //
               2, (SCREEN_HEIGHT-target_h)//2)
if not cap.isOpened():
    print("Error: could not open video")
    exit()


def get_visible_side(points):
    left_sum = sum(points[i][2] for i in [1, 2, 3, 7, 11,
                   13, 15, 17, 19, 21, 23, 25, 27, 29, 31])
    right_sum = sum(points[i][2] for i in [4, 5, 6, 8, 10,
                    12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    return "left" if left_sum <= right_sum else "right"


def draw_lines(result, frame):
    global visible_side
    global points
    height, width = frame.shape[:2]

    if len(result.pose_landmarks) > 0:

        for i, lm in enumerate(result.pose_landmarks[0]):

            x, y, z = int(lm.x*width), int(lm.y*height), lm.z
            points.append((x, y, z))
            cv2.circle(frame, (x, y), 5, (255, 255, 255))
            cv2.putText(frame, str(i), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for (start, end) in CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start][:2],
                         points[end][:2], (255, 255, 255), 2)

        if visible_side is None:
            visible_side = get_visible_side(points)

        cv2.putText(frame, visible_side, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


while True:
    points = []
    ret, frame = cap.read()

    if ret:
        ts_ms = int((frame_index / fps) * 1000.0)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, ts_ms)
        draw_lines(result, frame)
        frame_index += 1
        cv2.imshow('Stridelytics', frame)
    else:
        # Transpose: convert from list-of-frames to dict-of-landmarks
        output = {key: [] for key in CSV_KEYS}
        for frame in csv:
            for i, landmark_name in enumerate(CSV_KEYS):
                output[landmark_name].append(frame[i])

        print(output)
        with open('output/output.csv', 'w', newline='', encoding='utf-8') as f:
            writer = c.DictWriter(f, ["landmark", "points"])
            writer.writeheader()
            for landmark_name, points_list in output.items():
                writer.writerow(
                    {"landmark": landmark_name, "points": json.dumps(points_list)})
        break

    if frame_index % 10 == 0:
        csv.append(points)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
