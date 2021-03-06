import cv2
import mediapipe as mp
import numpy as np
import time 
import serial
import osc 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

# class for the coordinate of 
class coordinate:
  def __init__(self, x, y, id):
    self.x = x
    self.y = y
    self.cord = np.array([x, y])
    self.id = id 
  def __str__(self):
    str_rep = f'ID: {self.id}\nx coordinate: {self.x}\ny coordinate: {self.y}'
    return str_rep


colors = [(i,100,100) for i in range(255)]
color_counter = 0
cur_color = (np.random.randint(255), np.random.randint(255), np.random.433randint(255))

# tracks which points we want graphed and their line colors >:)
graph_landmarks = [ 8, 12, 16, 20]
landmark_colors = [[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)] for _ in range(len(graph_landmarks))] 
landmark_enabled = [1 for _ in range(len(graph_landmarks))]

# output mode and cooresponding output values
output_mode = 0
# 2 ^ 4 possible values
output_values = [[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19]]

output_mode_colors = [[0, 0, 0] for _ in range(len(graph_landmarks))] 

key_change_mode = 1
key = 0
#keys = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb/F#", "B", "E", "A", "D", "G"]
keys = ["A", "F", "D", "G"]
key_color = [[255, 0, 0],
              [0, 255, 0],
              [0, 0, 255],
              [255, 255, 0]]

for c in osc.clients:
  c.send_message("note", keys[key])

hand_num = 0

with mp_hands.Hands(
    min_detection_confidence=0.82,
    min_tracking_confidence=0.82) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue


    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    
    # get the iamge shape 
    img_shape = image.shape

    # store font 
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # line to split screen in half
    # cv2.line(image,(image.shape[1]//2, 0),(image.shape[1]//2, image.shape[0]) , (0,0,0), 1)

    if results.multi_hand_landmarks:
      if len(results.multi_hand_landmarks) == 1:
        output_mode = 0
      
      for hand_landmarks in results.multi_hand_landmarks:
        cur_hand_cords = []
        
        for i, landmark in enumerate(hand_landmarks.landmark):
          cur_x = int(landmark.x * img_shape[1])
          cur_y = int(landmark.y * img_shape[0])
          cur_cord = coordinate(cur_x, cur_y, i)

          cur_hand_cords.append(cur_cord)

          if i in graph_landmarks:
            cv2.circle(image, (cur_x, cur_y), radius=1, color=(225, 0, 100), thickness=10)
        
        if cur_hand_cords[0].x > image.shape[1]//2:
          hand_num = 0
        else:
          hand_num = 1
        if hand_num == 0:
          for i, landmark in enumerate(graph_landmarks):
            dist = np.linalg.norm(cur_hand_cords[4].cord - cur_hand_cords[landmark].cord)
            cv2.line(image,cur_hand_cords[4].cord, cur_hand_cords[landmark].cord, landmark_colors[i], 10)

            
            if dist < 30 and landmark_enabled[i]:
              print(f'serial output: {output_values[output_mode][i]}')

              rand_val = np.random.randint(3)
              landmark_colors[i][rand_val] += 40
              landmark_colors[i][rand_val] %= 255
              landmark_enabled[i] = 0
              for c in osc.clients:
                c.send_message("note", int(output_values[output_mode][i]))
            
            elif dist > 35 and not landmark_enabled[i]:
              print(f'serial output: {output_values[output_mode][i]}')
              landmark_enabled[i] = 1
              for c in osc.clients:
                c.send_message("note", int(output_values[output_mode][i]))
              
        elif hand_num == 1:
          distances = []
          for i, landmark in enumerate(graph_landmarks):
            dist = np.linalg.norm(cur_hand_cords[4].cord - cur_hand_cords[landmark].cord)
            cv2.line(image,cur_hand_cords[4].cord, cur_hand_cords[landmark].cord, output_mode_colors[i], 10)
            
            distances.append(dist)

            if dist < 50 and not output_mode:
              output_mode = i + 1
              output_mode_colors[i] = [255, 255, 255]

            elif dist > 80 and i == output_mode - 1:
              output_mode_colors[i] = [0, 0, 0]
              output_mode = 0

          if sum(distances) < 100 and key_change_mode == 1:
            key_change_mode = 0
            key += 1
            key %= len(keys)
            for c in osc.clients:
             c.send_message("key", keys[key])


          elif sum(distances) > 90:
            key_change_mode = 1

          cv2.circle(image, cur_hand_cords[4].cord, 2, key_color[key], 10)
                
        #cv2.putText(image, f'Key: {keys[key]}', (0, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        dist = np.linalg.norm(cur_hand_cords[4].cord - cur_hand_cords[8].cord)
        
        


    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()

# bursts of white noise 
# split image in half 