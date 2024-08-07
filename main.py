import cv2
import numpy as np
import time
import RPi.GPIO as GPIO
from gpiozero import AngularServo
from time import sleep
## Servo
from gpiozero.pins.pigpio import PiGPIOFactory
factory = PiGPIOFactory()
servo = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory=factory)
## Motor                                         
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
Ena,In1,In2 = 25,23,24
GPIO.setup(Ena,GPIO.OUT)
GPIO.setup(In1,GPIO.OUT)
GPIO.setup(In2,GPIO.OUT)
pwm = GPIO.PWM(Ena,100) #100 is frequency
pwm.start(0)

width = 320
height = 240
y_ref = 120

kernel = 3   #size of blur variable - odd
thres_min = 95 #color contract range
thres_max = 255
aperture_size = 3 #odd 3-7 (increase in detail)
L2Gradient = False

pos = prevT = e_prev = e_int = target = 0
kp = 1
ki = 0
kd = 0

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
# cap.set(10, brightness)

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel,kernel), 0)
    canny = cv2.Canny(blur, thres_max, thres_min, apertureSize=aperture_size, L2gradient = L2Gradient)
    return canny

def region_of_interest(img):
    w1,h1,w2,h2,w3,h3,w4,h4 = 0,height,width,height,width,0,0,0
    polygons = np.array([
    [(w1,h1), (w2,h2), (w3,h3), (w4,h4)]
    ], np.int32)
    mask = np.zeros_like(img) #shape of mask look like img
    cv2.fillConvexPoly(mask, polygons, 1)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters[0], line_parameters[1]
    y1 = height
    y2 = int(y1*(3/6))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def houghLines(img):
    houghLines = cv2.HoughLinesP(img, 1, np.pi/180, 80, np.array([]), minLineLength=40, maxLineGap=5) #image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]])
    return houghLines

def get_lines(img, lines):
    copyImage = img.copy()
    left_fit = []
    right_fit = []
    roiHeight = height
    line_img = np.zeros_like(img)
    global slope_left, inter_left, slope_right, inter_right, left_line, right_line
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if left_fit:
        slope_left, inter_left = left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(img, left_fit_average)
        try:
            cv2.line(line_img, (left_line[0],left_line[1]), (left_line[2],left_line[3]), (255, 0, 0), 2)
        except Exception as e: 
            print('Error', e)

    if right_fit:
        slope_right, inter_right = right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(img, right_fit_average)
        try:
            cv2.line(line_img, (right_line[0],right_line[1]), (right_line[2],right_line[3]), (255, 0, 0), 2)
        except Exception as e: 
            print('Error', e)

    return cv2.addWeighted(copyImage, 0.8, line_img, 1, 1)

def motor_go(speed):
    GPIO.output(In1, GPIO.LOW)
    GPIO.output(In2, GPIO.HIGH)
    pwm.ChangeDutyCycle(speed) # % of speed

def plant_control(u):
    allow = 7
    fast = 50
    slow = 40
    max_steering = 50
    deviation = u
    startTime = 0
    excTime = 0.01
    endTime = time.time()
    if deviation < allow and deviation > -allow: # do not steer if there is a 10-degree error range
        move = 0
        motor_go(fast)

    elif deviation > allow: # steer right if the deviation is positive
        move = deviation
        if move > max_steering:
            move = max_steering
        motor_go(slow)

    elif deviation < -allow: # steer left if deviation is negative
        move = deviation
        if move < -max_steering:
            move = -max_steering
        motor_go(slow)
    if endTime - startTime > excTime:
        servo.angle = move
        startTime = endTime

while(cap.isOpened()):
    _, img = cap.read()
    frame = cv2.resize(img, (width,height))
    canny_img = canny(frame)
    cropped_img = region_of_interest(canny_img)
    lines = houghLines(cropped_img)
    
    imageWithLines = get_lines(frame, lines)

    x_left = (y_ref-inter_left)/slope_left
    x_right = (y_ref-inter_right)/slope_right
    x_avg = ((x_right - x_left)/2) + x_left
    e = (x_avg - width/2)
    yaw = np.rad2deg(np.arctan(e/(height-y_ref)))
    # PID
    pos = yaw
    currT = time.time()
    deltaT = currT-prevT
    prevT = currT
    e_t = target-pos
    # derivative:
    der_e = (e_t-e_prev)/deltaT
    #integral:
    e_int = (e_int+e_t)*deltaT
    # control signal
    u = kp*e_t+kd*der_e+ki*e_int
    e_prev = e_t

    #print("Pos: ", yaw)
    print("Control: ", u)
    plant_control(u)

    if 0<x_avg<width: 
        turn_point = cv2.circle(imageWithLines, (int(x_avg),y_ref), radius=10, color=(0, 0, 255), thickness=-1)
    mask = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack((imageWithLines,mask))
    cv2.imshow("Result",hStack)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        servo.angle = 0
        motor_go(0)
        break
    
GPIO.cleanup()  
cap.release()
cv2.destroyAllWindows()