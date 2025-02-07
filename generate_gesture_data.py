import warnings
import cv2
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

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
num_images_taken = 0

alphabet_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
current_index = 0
element = alphabet_list[current_index]

while True:
    _, frame = video.read()
    flipFrame = cv2.flip(frame, 1)

    roi = frame[top_bound:bottom_bound, right_bound:left_bound]

    grayscale_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(grayscale_img, (9, 9), 0)

    if num_frames < 60:
        calc_accum_average(blurred_img, accum_weight)
        if num_frames <= 59:
            cv2.putText(flipFrame, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    elif num_frames <= 300:
        hand = detect_hand(blurred_img)
        cv2.putText(flipFrame, "ADJUST HAND...GESTURE FOR " + element, (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if hand is not None:
            thresholded_frame, hand_segment = hand
            cv2.drawContours(flipFrame, [hand_segment + (right_bound, top_bound)], -1, (255, 0, 0), 1)
            cv2.putText(flipFrame, str(num_frames) + " FOR " + element, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            resized_thresholded = cv2.resize(thresholded_frame, (150, 150))
            flipFrame[-160:-10, -160:-10] = cv2.cvtColor(resized_thresholded, cv2.COLOR_GRAY2BGR)
            # cv2.imshow("Thresholded Image", thresholded_frame)
    else:
        hand = detect_hand(blurred_img)
        
        if hand is not None:
            thresholded_frame, hand_segment = hand
            cv2.drawContours(flipFrame, [hand_segment + (right_bound, top_bound)], -1, (255, 0, 0), 1)
            cv2.putText(flipFrame, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(flipFrame, str(num_images_taken) + " IMAGES FOR " + element, (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            resized_thresholded = cv2.resize(thresholded_frame, (150, 150))
            flipFrame[-160:-10, -160:-10] = cv2.cvtColor(resized_thresholded, cv2.COLOR_GRAY2BGR)
            # cv2.imshow("Thresholded Image", thresholded_frame)

            if num_images_taken <= 40:
                cv2.imwrite(r"/Users/poojaraghuram/Documents/VS Code/ASL Translator/test/" + element + "/" + str(num_images_taken + 300) + '.jpg', thresholded_frame)
                time.sleep(1)
            else:
                num_images_taken = 0
                current_index += 1
                if current_index < len(alphabet_list):
                    element = alphabet_list[current_index]
                else:
                    break

            num_images_taken += 1
        else:
            cv2.putText(flipFrame, 'NO HAND DETECTED...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.rectangle(flipFrame, (right_bound, top_bound), (left_bound, bottom_bound), (255, 128, 0), 3)
    cv2.putText(flipFrame, "ASL TRANSLATOR", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
    num_frames += 1
    cv2.imshow("Sign Detection", flipFrame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()