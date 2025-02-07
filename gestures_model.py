import numpy as np
import cv2
import keras
# from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

model = keras.models.load_model("/Users/poojaraghuram/Documents/VS Code/ASL Translator/gestures_model.keras")

background = None
accum_weight = 0.5

top_bound = 100
bottom_bound = 300
right_bound = 150
left_bound = 350

height = bottom_bound - top_bound
width = left_bound - right_bound 

top_bound = top_bound - height // 2
bottom_bound = bottom_bound + height // 2
right_bound = right_bound - width // 2
left_bound = left_bound + width // 2

alphabet_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
alphabet_list.append('del')
alphabet_list.append('nothing')
alphabet_list.append('space')

def calc_accum_average(frame, accum_weight):
    global background

    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame, background, accum_weight)

def detect_hand(frame, threshold = 25):
    global background 
    diff = cv2.absdiff(background.astype('uint8'), frame)
    _, thresholded_frame = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        max_contours = max(contours, key = cv2.contourArea)
        return (thresholded_frame, max_contours)

video = cv2.VideoCapture(0)
num_frames = 0

while True:
    _, frame = video.read()
    flipFrame = cv2.flip(frame, 1)

    roi = frame[top_bound:bottom_bound, right_bound:left_bound]

    grayscale_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(grayscale_img, (9, 9), 0)

    if num_frames < 70:
        calc_accum_average(blurred_img, accum_weight)
        cv2.putText(flipFrame, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        hand = detect_hand(blurred_img)
        
        if hand is not None:
            thresholded_frame, hand_segment = hand
            cv2.drawContours(flipFrame, [hand_segment + (right_bound, top_bound)], -1, (255, 0, 0), 1)
            resized_thresholded = cv2.resize(thresholded_frame, (64, 64))
            resized_thresholded = cv2.cvtColor(resized_thresholded, cv2.COLOR_GRAY2BGR)
            flipFrame[-160:-10, -160:-10] = cv2.resize(resized_thresholded, (150, 150))
            resized_thresholded = np.reshape(resized_thresholded, (1, resized_thresholded.shape[0], resized_thresholded.shape[1], 3))

            prediction = model.predict(resized_thresholded)
            cv2.putText(flipFrame, alphabet_list[np.argmax(prediction)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.rectangle(flipFrame, (right_bound, top_bound), (left_bound, bottom_bound), (255, 128, 0), 3)
    cv2.putText(flipFrame, "ASL TRANSLATOR", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
    num_frames += 1
    cv2.imshow("Sign Detection", flipFrame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()