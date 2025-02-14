import cv2
from object_detector import *
import numpy as np

# Load aruco detector
parameters = cv2.aruco.DetectorParameters()
# detector_params = cv2.aruco.getDefaultParameters(dictionary=cv2.aruco.DICT_APRILTAG_36h11)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)


# Load object detector
detector = HomogeneousBgDetector()

# Load the image
img = cv2.imread("phone_aruco_marker.jpg")


# get aruco marker
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

int_corners = np.int_(corners)

# draw polygon around the marker

cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

# Aruco Perimeter
aruco_perimeter = cv2.arcLength(int_corners[0], True)

# Pixel to centimeter ratio
pixel_to_cm_ratio=aruco_perimeter/20
print(pixel_to_cm_ratio)


contours = detector.detect_objects(img)

# Draw object boundaries

for cnt in contours:
    # Get rect
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    
    # Get width and height of objects in cm
    obj_width=w/pixel_to_cm_ratio
    obj_height=h/pixel_to_cm_ratio

    box = cv2.boxPoints(rect)

    box = np.int_(box)

    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.polylines(img, [box], True, (255, 0, 0), 2)

    cv2.putText(img, "Width {} cm".format(round(obj_width, 1)), (int(x)-100, int(y)-15),
                cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(img, "Height {} cm".format(round(obj_height, 1)), (int(x)-100, int(y)+15),
                cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)


cv2.imshow("image", img)

cv2.waitKey(1)
