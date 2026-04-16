import cv2
import matplotlib.pyplot as plt
import numpy as np
from library.selectBlob import *

img = cv2.imread("Visao_de_maquina/aps3/Figuras_APS3/Fig3_lata5.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("File not found. Bye!")
    exit(0)
h, w = img.shape

params = cv2.SimpleBlobDetector_Params()

# Set blob color (0=black, 255=white)
params.filterByColor = True
params.blobColor = 255

# Filter by Area
params.filterByArea = True
params.minArea = 107000
params.maxArea = 200000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 1
params.maxCircularity = 1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.995
params.maxConvexity = 1

# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
#params.maxInertiaRatio = 1

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

img_bin = np.zeros((h, w), dtype=np.uint8)

img_bin = np.where(img < 50, 0, 255).astype(np.uint8)


#display image (with text)
cv2.imshow("Img binary", img_bin)

# Detect blobs
KP = detector.detect(img_bin)
print("Nro de blobs: ",len(KP))

mask_blobs = selectBlob(img_bin, KP)

contours, hierarchy = cv2.findContours(mask_blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img1_text = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
cv2.drawContours(img1_text, contours, -1, (255,0,0), 2)

contagem = 0
i=1
for cnt in contours:
    ellipse = cv2.fitEllipse(cnt)
    (x, y), (d1, d2), angle = ellipse
    cv2.ellipse(img1_text, ellipse, (0,255,0), 2)
    cv2.putText(img1_text, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    area = np.pi*(d1/2)*(d2/2)
    print("Area_", i, "=", area)
    contagem +=1
    i+=1

if contagem == 1:
    print('A lata está inteira')
else:
    print('A lata está amassada')
#display image (with text)
cv2.imshow("Img1 with texts", img1_text)

#aguarda uma tecla
cv2.waitKey(0)