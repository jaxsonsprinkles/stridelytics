import cv2
from screeninfo import get_monitors

SCREEN_WIDTH, SCREEN_HEIGHT = get_monitors()[0].width, get_monitors()[0].height
print(SCREEN_WIDTH, SCREEN_HEIGHT)
cap = cv2.VideoCapture('input/input.mp4')
cv2.namedWindow('Stridelytics', cv2.WINDOW_NORMAL)
_, _, w, h = cv2.getWindowImageRect('Stridelytics')
cv2.resizeWindow('Stridelytics', w//2, h//2)
cv2.moveWindow('Stridelytics', SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
if not cap.isOpened():
    print("Error: could not open video")
    exit()

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow('Stridelytics', frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()