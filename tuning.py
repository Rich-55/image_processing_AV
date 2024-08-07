import cv2
import numpy as np

frameCounter = 0
cap = cv2.VideoCapture("1.mp4")
width = 640
height = 480
cap.set(3,640) #with
cap.set(4,320) #hight

kernel = 9
thres_min = 69
thres_max = 210
aperture_size = 3 #odd 3-7 (increase in detail)
L2Gradient = False
def region_of_interest(img,w1,h1,w2,h2,w3,h3,w4,h4):

    polygons = np.array([
    [(w1, h1), (w2, h2), (w3, h3), (w4, h4)]
    ], np.int32)
    mask = np.zeros_like(img) #shape of mask look like img
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def nothing(a):
    pass
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 360)
cv2.createTrackbar("Width 1", "Trackbars", 0+20, width, nothing)
cv2.createTrackbar("Height 1", "Trackbars", 360-20, height, nothing)
cv2.createTrackbar("Width 2", "Trackbars", 640-20, width, nothing)
cv2.createTrackbar("Height 2", "Trackbars", 360-20, height, nothing)
cv2.createTrackbar("Width 3", "Trackbars",  640-20, width, nothing)
cv2.createTrackbar("Height 3", "Trackbars", 0+20, height, nothing)
cv2.createTrackbar("Width 4", "Trackbars", 0+20, width, nothing)
cv2.createTrackbar("Height 4", "Trackbars", 0+20, height, nothing)


while True:
    frameCounter += 1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       frameCounter = 0 
    _, img = cap.read()
    img = cv2.resize(img, (width,height))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel,kernel), 0)
    canny = cv2.Canny(blur, thres_max, thres_min, apertureSize=aperture_size, L2gradient = L2Gradient) #threshold
    
    width1 = cv2.getTrackbarPos("Width 1", "Trackbars")
    height1 = cv2.getTrackbarPos("Height 1", "Trackbars")
    width2 = cv2.getTrackbarPos("Width 2", "Trackbars")
    height2 = cv2.getTrackbarPos("Height 2", "Trackbars")
    width3 = cv2.getTrackbarPos("Width 3", "Trackbars")
    height3 = cv2.getTrackbarPos("Height 3", "Trackbars")
    width4 = cv2.getTrackbarPos("Width 4", "Trackbars")
    height4 = cv2.getTrackbarPos("Height 4", "Trackbars")

    pol = np.array([(width1, height1), (width2, height2), (width3, height3), (width4, height4)], np.int32)
    cv2.polylines(img,[pol],True,(0,255,255))
    mask = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    cropped_img = region_of_interest(mask,width1,height1,width2,height2,width3,height3,width4,height4)
    hStack = np.hstack((img,cropped_img))
    print(pol)
    cv2.imshow("Output", hStack)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows