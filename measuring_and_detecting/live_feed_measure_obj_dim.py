import os
import cv2
import numpy as np
import streamlit as st
from object_detector import HomogeneousBgDetector
from helpers import *


# Initialize session state variables
if 'start' not in st.session_state:
    st.session_state.start = False
if 'stop' not in st.session_state:
    st.session_state.stop = False
if 'save' not in st.session_state:
    st.session_state.save = False

# Directory to save object images
save_dir = 'saved_images'
os.makedirs(save_dir, exist_ok=True)

st.title("MEASUREMENT OF OBJECT DIMENSIONS - OPENCV")

frame_placeholder = st.empty()
col1, col2, col3, col4 = st.columns(4)

with col2:
    st.button("Start", on_click=lambda: st.session_state.update({'start': True, 'stop': False}))
with col3:
    st.button("Stop", on_click=lambda: st.session_state.update({'stop': True, 'start': False}))
with col4:
    st.button("Save", on_click=lambda: st.session_state.update({'save': True}))

# Load ArUco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# Load object detector
detector = HomogeneousBgDetector()

# Load the Cap
cap = cv2.VideoCapture('http://192.168.124.176:8080/video')
# cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened() and st.session_state.start:
    ret, img = cap.read()
    
    if not ret:
        st.write("The video capture has ended")
        break

    # Get ArUco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if corners:
        # Draw polygon around the marker
        int_corners = np.int_(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # ArUco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(img)

        # Draw objects boundaries and dimensions
        for i, cnt in enumerate(contours):
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.int_(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width: {:.1f} cm".format(object_width), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(img, "Height: {:.1f} cm".format(object_height), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            
            if st.session_state.save:
                # Extract and save each detected object
                object_img = four_point_transform(img, box)
                filename = f"object_{i}_width_{round(object_width, 1)}_height_{round(object_height, 1)}.png"
                cv2.imwrite(os.path.join(save_dir, filename), object_img)
                
        if st.session_state.save:
            st.session_state.save = False  # Reset save flag after saving
        
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB")

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
