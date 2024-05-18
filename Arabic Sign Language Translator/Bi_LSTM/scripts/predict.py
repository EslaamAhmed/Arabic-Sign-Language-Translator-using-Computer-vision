
import cv2
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.models import load_model 
from main import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Load the saved model
model = load_model("../models/bi_lstm_model.h5")

# Define Arabic actions
ar_actions = np.array(["ما اسمك ؟", "اين منزلك ؟", "السلام عليكم", "تمام الحمدلله", "عامل ايه ؟", 'انا تعبان', "محتاج مساعدة ؟", "عندك كم سنة ؟", "رقم تليفونك", "انا من مصر"])

def predict_actions(threshold=0.5, actions=ar_actions):
    # New detection variables
    sequence = []
    sentence = []

    cap = cv2.VideoCapture(0)

    # Set mediapipe model
    mp_holistic = mp.solutions.holistic  # Holistic Model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            print(keypoints.shape)
            
            # Check for inactivity
            arr = keypoints[98:]
            num_zeros = np.count_nonzero(arr == 0)
            
            if num_zeros == 21*6 and len(sequence) < 10:
                sequence = []
            else:
                sequence.append(keypoints)
                sequence = sequence[-40:]
            
            if len(sequence) == 40:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                sequence.clear()
                print(ar_actions[np.argmax(res)])
                
                if res[np.argmax(res)] > threshold:
                    sentence = ar_actions[np.argmax(res)]
            
            # Display the result
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            
            # Convert Arabic text to Unicode
            ar_font = "arial.ttf"  # Make sure this font file is available
            font = ImageFont.truetype(ar_font, 32)
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((0, 0), get_display(arabic_reshaper.reshape(sentence)), font=font)
            img = np.array(img_pil)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', img)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Call the function
predict_actions()