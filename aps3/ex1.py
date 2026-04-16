import cv2
import matplotlib.pyplot as plt
import numpy as np
from library.selectBlob import *

img = cv2.imread("Visao_de_maquina/aps3/Figuras_APS3/Fig1_Moedas4.png", cv2.IMREAD_GRAYSCALE)
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
params.minArea = 1000
params.maxArea = 30000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.8
params.maxCircularity = 1.2

# Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 0.87
#params.maxConvexity = 1

# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
#params.maxInertiaRatio = 1

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

img_bin = np.zeros((h, w), dtype=np.uint8)

img_bin = np.where(img < 250, 255, 0).astype(np.uint8)


img_close = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

# Detect blobs
KP = detector.detect(img_close)
print("Nro de blobs: ",len(KP))

mask_blobs = selectBlob(img_close, KP)

contours, hierarchy = cv2.findContours(mask_blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img1_text = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
cv2.drawContours(img1_text, contours, -1, (255,0,0), 2)

valor = 0
i=1
for cnt in contours:
    ellipse = cv2.fitEllipse(cnt)
    (x, y), (d1, d2), angle = ellipse
    cv2.ellipse(img1_text, ellipse, (0,255,0), 2)
    cv2.putText(img1_text, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    area = np.pi*(d1/2)*(d2/2)
    print("Area_", i, "=", area)
    if area < 3100:
        valor += 0.1
        cv2.putText(img1_text, "$ 0.10", (int(x), int(y)+15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    elif area < 3400:
        valor += 0.01
        cv2.putText(img1_text, "$ 0.01", (int(x), int(y)+15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    elif area < 4100:
        valor += 0.05
        cv2.putText(img1_text, "$ 0.05", (int(x), int(y)+15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    else:
        valor += 0.25
        cv2.putText(img1_text, "$ 0.25", (int(x), int(y)+15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    i=i+1

print("Valor total: $ {:.2f}".format(valor))

#display image (with text)
cv2.imshow("Img1 with texts", img1_text)

#aguarda uma tecla
cv2.waitKey(0)