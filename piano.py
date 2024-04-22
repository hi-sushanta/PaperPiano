# Import necessary libraries
import pickle  # Pickle library for serializing Python objects
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy library for numerical operations
from cvzone.HandTrackingModule import HandDetector
import time
from utils import Utils
import pygame

######################################
cam_id = 1
cam_id2 = 0 # Laptop Webcamera detect
width, height = 1280, 720
map_file_path = "GetCornerPoint/corner.p"
countries_file_path = "GetPollygonPoint/piano.p"
######################################
 
file_obj = open(map_file_path, 'rb')
key_point = pickle.load(file_obj)
file_obj.close()
 
# Load previously defined Regions of Interest (ROIs) polygons from a file
if countries_file_path:
    file_obj = open(countries_file_path, 'rb')
    polygons = pickle.load(file_obj)
    file_obj.close()
else:
    polygons = []
 
# Open a connection to the webcam
cap = cv2.VideoCapture(cam_id)  # For Webcam
cap2 = cv2.VideoCapture(cam_id2)

# Set the width and height of the webcam frame
cap.set(3, width)
cap.set(4, height)
cap2.set(3, width)
cap2.set(4, height)
win_name = "Output Image"
# cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(win_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# Counter to keep track of how many polygons have been created
counter = 0
 
# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False,
                        maxHands=1,
                        modelComplexity=1,
                        detectionCon=0.5,
                        minTrackCon=0.5)
 
ut = Utils(detector)

def play_sound(sound_path):
    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    success2, img2 = cap2.read()
    imgWarped, matrix = ut.warp_image(img, key_point)
    imgOutput = img.copy()
 
    # Find the hand and its landmarks
    warped_point = ut.get_finger_location(img, imgWarped,matrix)
    
    h, w, _ = imgWarped.shape
    imgOverlay = np.zeros((h, w, 3), dtype=np.uint8)
    gname = None
    if warped_point:
        imgOverlay,gname = ut.create_overlay_image(polygons, warped_point, imgOverlay)
        imgOutput = ut.inverse_warp_image(img, imgOverlay, key_point)
        

    cv2.imshow(win_name, imgOutput)
    cv2.imshow("YourFace",img2)
    
    if gname != None:
            my_sound = play_sound(f"song/{gname}.wav")
    
    key = cv2.waitKey(1)

    if key == 27:
        break
  
cap.release()
cv2.destroyAllWindows()