import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands=mp_hands.Hands(min_detection_confidence=0.8,
                     min_tracking_confidence=0.5) 
def rec_coords(hand,w,h):
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
    for lm in hand.landmark:
        x, y = int(lm.x * (w*1.09)), int(lm.y *( h*1.09))
        if  x < w and x > x_max :
            x_max = int(x)
        if x < x_min and x > 0:
            x_min = int(x-(x*0.43))
        if y > y_max and y < h:
            y_max = int(y)
        if y < y_min and y>0:
            y_min = int(y-(y*0.48))
    return x_min,y_min,x_max,y_max
def to_edge(img):
      img=cv2.resize(img, (250,250))
      Gray_Img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
      _,threshold_Img = cv2.threshold(Gray_Img,240,255,cv2.THRESH_BINARY_INV)
      canny_Img = cv2.Canny(threshold_Img,90,100)
    #   Mask = cv2.bitwise_not(threshold_Img)
      return canny_Img

def ExtractHandRegion(img):
    # BGR 2 RGB
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Set flag
    image.flags.writeable = False
    # Detections
    results = hands.process(image)
    # Set flag to true
    image.flags.writeable = True
    # RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Detections
    h,w,_=image.shape
    # Rendering results
    if results.multi_hand_landmarks:
        mult_coord=[]
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(255,255,255), thickness=3,circle_radius=0),
            mp_drawing.DrawingSpec(color=(255,255,255), thickness=3,circle_radius=0) )
            if len(results.multi_handedness)==2:
                mult_coord.append(rec_coords(hand,w,h))
            else:
                x_min, y_min, x_max, y_max=rec_coords(hand,w,h)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                both_hands=image[y_min:y_max,x_min:x_max]
                tr_img=cv2.resize(to_edge(both_hands), (250,250))
                return image
        if len(mult_coord)!=0:
            x_min=min(mult_coord[0][0],mult_coord[1][0])
            y_min=min(mult_coord[0][1],mult_coord[1][1])
            x_max=max(mult_coord[0][2],mult_coord[1][2])
            y_max=max(mult_coord[0][3],mult_coord[1][3])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            single_hand=image[y_min:y_max,x_min:x_max]
            tr_img=cv2.resize(to_edge(single_hand), (250,250))
            return image
