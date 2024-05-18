import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import mediapipe as mp
import cv2
import os
from main import mediapipe_detection, draw_styled_landmarks, extract_keypoints

mp_holistic = mp.solutions.holistic  # Holistic Model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
en_actions = np.array(['What is your name','where is ur home','salam alaykum','im fine','how r u','im tired','need a help','how old r u','ur phone num','im from egypt'])

DATA_PATH= os.path.join('..\MP_Data') # Path For Exported Data

no_sequences = 50# number of videos for each word
sequence_length= 40 # num of frames in each video
def collect_data(no_sequences=no_sequences,sequence_length=sequence_length):
    # Creating folder for each class

    for action in en_actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass



    # Set mediapipe model p
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # NEW LOOP
        # Loop through actions
        for action in en_actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()
                    #mirror the frame
                    frame=cv2.flip(frame, 1)
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(1000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    collect_data()