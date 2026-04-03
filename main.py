import cv2
from screeninfo import get_monitors
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
               (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
               (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
               (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
               (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
               (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
               (29, 31), (30, 32), (27, 31), (28, 32)]
POINT_LANDMARKS = [
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

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = PoseLandmarker.create_from_options(options)


cap = cv2.VideoCapture('input/input.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0
frame_index = 0
SCREEN_WIDTH, SCREEN_HEIGHT = get_monitors()[0].width, get_monitors()[0].height
target_w, target_h = SCREEN_WIDTH//2, SCREEN_HEIGHT//2
cv2.namedWindow('Stridelytics', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stridelytics', target_w, target_h)
cv2.moveWindow('Stridelytics', (SCREEN_WIDTH-target_w) //
               2, (SCREEN_HEIGHT-target_h)//2)
if not cap.isOpened():
    print("Error: could not open video")
    exit()

def angle_between(a, b, c):
    # Angle between triangle ABC where A, B, and C are indexes of points on the body
    ab = math.dist(a, b)
    bc = math.dist(b, c)
    ca = math.dist(c, a)
    return math.acos((ab**2+bc**2-ca**2)/(2*ab*bc))*(180/math.pi)



def draw_lines(result, frame):
    global visible_side

    points = {}
    height, width = frame.shape[:2]

    if len(result.pose_landmarks) > 0:
        
        for i, lm in enumerate(result.pose_landmarks[0]):
            
            x, y = int(lm.x*width), int(lm.y*height)
            points[POINT_LANDMARKS[i]] = (x,y)
            cv2.circle(frame, (x, y), 5, (255, 255, 255))
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        for (start, end) in CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points.keys()[start],
                        points.keys()[end], (255, 255, 255), 2)
        if visible_side is None:
            visible_side = min(points["left_hip"].z, points["right_hip"].z)
        cv2.putText(frame, str(visible_side), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


while True:

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
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
