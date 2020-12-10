import cv2

from hand import Hand


def detect_face(frame, block=False, colour=(0, 0, 0)):
    fill = [1, -1][block]
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    area = 0
    X = Y = W = H = 0
    for (x, y, w, h) in faces:
        if w * h > area:
            area = w * h
            X, Y, W, H = x, y, w, h
    cv2.rectangle(frame, (X - 50, Y - 50), (X + W + 50, Y + H + 50), colour, fill)


def scale_image(img, scale=1):
    width = img.shape[0] * scale
    height = img.shape[1] * scale
    return cv2.resize(img, (height, width))


def capture_histogram(cap):
    """Return histogram needed to calibrate detection with color of your hand"""
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Place region of the hand inside box and press `A`",
                    (5, 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (500, 100), (580, 180), (105, 105, 105), 2)
        box = frame[105:175, 505:575]

        cv2.imshow("Capture Histogram", scale_image(frame))
        key = cv2.waitKey(10)
        if key == ord('a'):
            object_color = box
            break

    object_color_hsv = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)
    object_hist = cv2.calcHist([object_color_hsv], [0, 1], None,
                               [12, 15], [0, 180, 0, 256])

    cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.destroyAllWindows()
    return object_hist


def locate_object(frame, object_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # apply back projection to image using object_hist as
    # the model histogram
    object_segment = cv2.calcBackProject(
        [hsv_frame], [0, 1], object_hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cv2.filter2D(object_segment, -1, disc, object_segment)

    _, segment_thresh = cv2.threshold(
        object_segment, 70, 255, cv2.THRESH_BINARY)

    # apply some image operations to enhance image
    kernel = None
    eroded = cv2.erode(segment_thresh, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # masking
    masked = cv2.bitwise_and(frame, frame, mask=closing)

    return closing, masked, segment_thresh


def detect_hand(frame, hist):
    detected_hand, masked, raw = locate_object(frame, hist)
    return Hand(detected_hand, masked, raw, frame)
