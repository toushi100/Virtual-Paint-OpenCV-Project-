import cv2 as cv
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass


cv.namedWindow("HSV")
cv.resizeWindow("HSV", frameWidth, frameHeight)
cv.createTrackbar("HUE min", "HSV", 0, 179, empty)
cv.createTrackbar("HUE max", "HSV", 179, 179, empty)
cv.createTrackbar("SAT min", "HSV", 0, 255, empty)
cv.createTrackbar("SAT max", "HSV", 255, 255, empty)
cv.createTrackbar("VAL min", "HSV", 0, 255, empty)
cv.createTrackbar("VAL max", "HSV", 255, 255, empty)

while True:
    success, img = cap.read()
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos("HUE min", "HSV")
    h_max = cv.getTrackbarPos("HUE max", "HSV")
    s_min = cv.getTrackbarPos("SAT min", "HSV")
    s_max = cv.getTrackbarPos("SAT max", "HSV")
    v_min = cv.getTrackbarPos("VAL min", "HSV")
    v_max = cv.getTrackbarPos("VAL max", "HSV")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(imgHSV, lower, upper)
    result = cv.bitwise_and(img, img, mask=mask)

    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])
    cv.imshow("result", hStack)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
