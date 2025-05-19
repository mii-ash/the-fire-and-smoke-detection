import cv2
import numpy as np

# Load the image
image = cv2.imread('tomatoes.jpg')
resized = cv2.resize(image, (640, 480))

# Convert to HSV color space
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

# Define color ranges
red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])

green_lower = np.array([35, 40, 40])
green_upper = np.array([85, 255, 255])

# Create masks
red_mask = cv2.inRange(hsv, red_lower, red_upper)
green_mask = cv2.inRange(hsv, green_lower, green_upper)

# Find contours
def detect_color_objects(mask, color_name, draw_color):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(resized, (x, y), (x+w, y+h), draw_color, 2)
            cv2.putText(resized, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)

# Detect ripe and unripe
detect_color_objects(red_mask, 'Ripe', (0, 0, 255))
detect_color_objects(green_mask, 'Unripe', (0, 255, 0))

# Show the result
cv2.imshow('Tomato Quality Inspection', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()