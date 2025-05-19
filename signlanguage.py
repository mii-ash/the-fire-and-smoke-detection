import cv2
import numpy as np

def count_fingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        a = np.linalg.norm(contour[e][0] - contour[s][0])
        b = np.linalg.norm(contour[f][0] - contour[s][0])
        c = np.linalg.norm(contour[e][0] - contour[f][0])
        if b * c == 0:
            continue
        angle = np.arccos((b*2 + c - a*2) / (2 * b * c)) * (180 / np.pi)
        if angle < 90 and d > 3000:
            count += 1
    return count + 1 if count > 0 else 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Webcam not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    label = "No hand"
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 3000:
            fingers = count_fingers(c)
            if fingers == 0:
                label = "Fist"
            elif fingers == 1:
                label = "Thumbs Up"
            elif fingers == 2:
                label = "Peace"
            elif fingers >= 4:
                label = "Hi-Fi"
            else:
                label = f"{fingers} fingers"

    cv2.putText(frame, f'Sign: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()