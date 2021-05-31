import cv2
import mediapipe as mp
import numpy as np
import time 
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
cur_color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))

# tracks which points we want graphed and their line colors >:)
graph_landmarks = [ 8, 12, 16, 20]
landmark_colors = [[0, 0, 0] for _ in range(len(graph_landmarks))] 
landmark_enabled = [1 for _ in range(len(graph_landmarks))]

# prior coordinates 
prior_cords = [[],[]]

# tracer list of prior cords
tracer = []

# rate at which hand is moving
roc = [0, 0]

# output mode and cooresponding output values
output_mode = 0
# 2 ^ 4 possible values
output_values = [[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19]]

output_mode_colors = [[0, 0, 0] for _ in range(len(graph_landmarks))] 

with mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:
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
    
    

    if results.multi_hand_landmarks:
      if len(results.multi_hand_landmarks) == 1:
        output_mode = 0
      
      for hand_num, hand_landmarks in enumerate(results.multi_hand_landmarks):
        cur_hand_cords = []
        
        for i, landmark in enumerate(hand_landmarks.landmark):
          cur_x = int(landmark.x * img_shape[1])
          cur_y = int(landmark.y * img_shape[0])
          cur_cord = coordinate(cur_x, cur_y, i)

          cur_hand_cords.append(cur_cord)

          if i in graph_landmarks:
            cv2.circle(image, (cur_x, cur_y), radius=1, color=(225, 0, 100), thickness=10)
        
        # tracer effect
        # for prior in tracer:
        #   for cord in prior:
        #     cv2.circle(image, cord.cord, radius=1, color=colors[color_counter], thickness=10)
        #     color_counter += 1
        #     color_counter = color_counter % len(colors)

        # draw distance bar
        # cv2.line(image,cur_hand_cords[4].cord, cur_hand_cords[8].cord, cur_color, 10)
        # dist = np.linalg.norm(cur_hand_cords[4].cord - cur_hand_cords[8].cord)
        # cv2.putText(image, f'Dist: {dist}',(10,400), font, 1,(255,255,255),2,cv2.LINE_AA)

        # distance bar cycle through patterns
        # if dist < 20:
        #   print(f'change: {color_counter}')
        #   color_counter += 1
        #   color_counter = color_counter % len(colors)
        
        # change distance bar randomly
        # if dist < 50:
        #   print(f'change: {color_counter}')
        #   cur_color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        if hand_num == 0:
          for i, landmark in enumerate(graph_landmarks):
            dist = np.linalg.norm(cur_hand_cords[4].cord - cur_hand_cords[landmark].cord)
            cv2.line(image,cur_hand_cords[4].cord, cur_hand_cords[landmark].cord, landmark_colors[i], 10)

            
            if dist < 20 and landmark_enabled[i]:
              print(f'serial output: {output_values[output_mode][i]}')

              rand_val = np.random.randint(3)
              landmark_colors[i][rand_val] += 40
              landmark_colors[i][rand_val] %= 255
              landmark_enabled[i] = 0
            
            elif dist > 30 and not landmark_enabled[i]:
              landmark_enabled[i] = 1
              
        elif hand_num == 1:
           for i, landmark in enumerate(graph_landmarks):
            dist = np.linalg.norm(cur_hand_cords[4].cord - cur_hand_cords[landmark].cord)
            cv2.line(image,cur_hand_cords[4].cord, cur_hand_cords[landmark].cord, output_mode_colors[i], 10)
            
            if dist < 50 and not output_mode:
              output_mode = i + 1
              output_mode_colors[i] = [255, 255, 255]

            elif dist > 60 and i == output_mode - 1:
              output_mode_colors[i] = [0, 0, 0]
              output_mode = 0


          

        if prior_cords[hand_num]:
          del_x = cur_hand_cords[0].x - prior_cords[hand_num][0].x 
          del_y = cur_hand_cords[0].y - prior_cords[hand_num][0].y
          roc[hand_num] = abs(del_y / del_x) if del_x and del_y else 0
          #cv2.putText(image, f'Roc: {roc[hand_num]}',(90,50), font, 1, (255,255,255),2,cv2.LINE_AA)

        dist = np.linalg.norm(cur_hand_cords[4].cord - cur_hand_cords[8].cord)
        
        prior_cords[hand_num] = cur_hand_cords
        
        #tracer.append(cur_hand_cords)

        #if len(tracer) == 15:
        #  tracer = tracer[1:]

        # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()