# Import necessary libraries
import pickle  
import cv2  
import numpy as np  
from cvzone.HandTrackingModule import HandDetector
from utils import Utils
import pygame

######################################
cam_id = 1
cam_id2 = 0 # Laptop Webcamera detect
width, height = 1280, 720
corner_file_path = "GetCornerPoint/corner.p"
key_file_path = "GetPollygonPoint/piano.p"
######################################
 
file_obj = open(corner_file_path, 'rb')
key_point = pickle.load(file_obj)
file_obj.close()
 
# Load previously defined Regions of Interest (ROIs) polygons from a file
if key_file_path:
    file_obj = open(key_file_path, 'rb')
    polygons = pickle.load(file_obj)
    file_obj.close()
else:
    polygons = []
 
# Open a connection to the webcam
cap = cv2.VideoCapture(cam_id)  # For extranal Webcam
cap2 = cv2.VideoCapture(cam_id2) # For Internal laptop webcam

# Set the width and height of the webcam frame
cap.set(3, width)
cap.set(4, height)
cap2.set(3, width)
cap2.set(4, height)
win_name = "Output Image"

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
    # while pygame.mixer.music.get_busy():
    #     continue

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    success2, img2 = cap2.read()
    hands , img = detector.findHands(img,draw=True,flipType=True)
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

    if (key == 27) or (key == ord("Q")) or (key == ord("q")):
        break
  
cap.release()
cap2.release()
cv2.destroyAllWindows()