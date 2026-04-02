import cv2
from screeninfo import get_monitors
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
frame_index = 0
SCREEN_WIDTH, SCREEN_HEIGHT = get_monitors()[0].width, get_monitors()[0].height
target_w, target_h = SCREEN_WIDTH//2, SCREEN_HEIGHT//2
cv2.namedWindow('Stridelytics', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stridelytics', target_w, target_h)
cv2.moveWindow('Stridelytics', (SCREEN_WIDTH-target_w)//2, (SCREEN_HEIGHT-target_h)//2)
if not cap.isOpened():
    print("Error: could not open video")
    exit()

def draw_lines(result, frame):
    if result.pose_landmarks:
        for landmark in result.pose_landmarks[0]:
            cv2.circle(frame, (landmark.x, landmark.y), 2, (0,255,0), -1)


while True:

    ret, frame = cap.read()

    if ret:
        cv2.imshow('Stridelytics', frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    ts_ms = (frame_index / fps) * 1000.0

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, ts_ms)
    draw_lines(result, frame)
    frame_index += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()