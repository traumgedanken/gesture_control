from time import sleep

import numpy as np
import pyautogui
import cv2
from win32api import GetSystemMetrics

import detection
from gesture_events import UpEvent, DownEvent, DragEvent, DoubleClickEvent, LeftClickEvent, RightClickEvent, \
    process_gesture
from training import predict_gesture


def get_video_source():
    for i in range(0, 5):
        try:
            cap = cv2.VideoCapture(0)
            cap.read()
            cap.release()
            return i
        except:
            ...
    return -1


cap = cv2.VideoCapture(get_video_source())
hist = detection.capture_histogram(cap)
SCREEN_SIZE = (GetSystemMetrics(1), GetSystemMetrics(0))
drag = False
EVENT_MAPPER = {'up': 'Scroll Up', 'down': 'Scroll down', 'left': 'Left Click', 'right': 'Right Click',
                'horizontal': 'Double click', 'vertical': 'Drag', 'fist': 'Nothing'}


def get_cursor_position(position, frame_shape):
    x, y = position[0] * (SCREEN_SIZE[1] / frame_shape[1]), \
           position[1] * (SCREEN_SIZE[0] / frame_shape[0])
    width, height = SCREEN_SIZE
    x_diff = width / 2 - x
    y_diff = height / 2 - y
    scale = 1.2
    return width / 2 - scale * x_diff, \
           height / 2 - scale * y_diff


while True:
    if cv2.waitKey(5) == ord('q'):
        break

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    try:
        # detection.detect_face(frame, block=Trqe)
        hand = detection.detect_hand(frame, hist)
        center = hand.get_center_of_mass()
        cursor_position = get_cursor_position(center, frame.shape)
        pyautogui.moveTo(*cursor_position, _pause=False)
        gesture = predict_gesture(hand.binary)
        cv2.putText(hand.outline, f'{EVENT_MAPPER[gesture]} ({gesture})',
                    (5, hand.outline.shape[0] - 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.circle(hand.outline, center, 5, (0, 0, 255), -1)
        cv2.imshow("Hand", detection.scale_image(hand.outline))

        process_gesture(gesture)
    except UpEvent:
        print('up')
        pyautogui.scroll(300)
    except DownEvent:
        print('down')
        pyautogui.scroll(-300)
    except DragEvent:
        print('drag')
        if drag:
            pyautogui.mouseUp()
        else:
            pyautogui.mouseDown()
        drag = not drag
    except DoubleClickEvent:
        print('double')
        pyautogui.doubleClick()
    except LeftClickEvent:
        print('left')
        pyautogui.click()
    except RightClickEvent:
        print('right')
        pyautogui.rightClick()
    except Exception as e:
        cv2.imshow("Hand", detection.scale_image(frame))
        print(e)

cap.release()
cv2.destroyAllWindows()
