import cv2
import matplotlib.pyplot as plt
import numpy as np
from library.selectBlob import *

img = cv2.imread("Visao_de_maquina/Projeto intermediario/Dataset_Projeto1/_Eucalipto_Escolhidos1/Eucalipto1.jpg")
if img is None:
    print("File not found. Bye!")
    exit(0)
h, w, l = img.shape

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

img1_text = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

valor = 0
i=1

plt.figure('img with text')
plt.imshow(img_close, cmap='gray',vmin=0, vmax=255)
plt.show()