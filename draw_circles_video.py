__author__      = "Phelipe Muller e Sabrina Machado"
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# link https://youtu.be/WR6updj8A4c

# If you want to open a video, just change this path
#cap = cv2.VideoCapture('hall_box_battery.mp4')

# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
img1 = cv2.imread('powerpuff-girls.png',0)          # Imagem a procurar
kp1, des1 = sift.detectAndCompute(img1,None)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    # Detect the edges present in the image
    bordas = auto_canny(blur)

    MIN_MATCH_COUNT = 81.5


    img2 = frame # Imagem do cenario

    # find the keypoints and descriptors with SIFT in each image
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 5)

    # Configura o algoritmo de casamento de features
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    circles = []


    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

    # Raposa = None
    # Raposa = cv2.Canny(bordas,cv2.HOUGH_GRADIENT)
    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/

    if len(good) > MIN_MATCH_COUNT:
        print("Powerpuff Girls!!!")

    else:
        print("Nao encontrei nada...")

    if circles is not None:
        circles = np.uint16(np.around(circles))



    # linha azul diagonal
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    # cv2.line(bordas_color,(0,0),(511,511),(255,0,0),5)

    # Quadrado Verde x
    #cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    # cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(bordas_color,'Ninjutsu ;)',(0,50), font, 2,(255,255,255),2,cv2.LINE_AA)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow('Detector de circulos',bordas_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
