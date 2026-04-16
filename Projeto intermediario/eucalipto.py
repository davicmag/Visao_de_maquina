import cv2
import matplotlib.pyplot as plt
import numpy as np
from library.selectBlob import *

img = cv2.imread("Visao_de_maquina/Projeto intermediario/Dataset_Projeto1/_Eucalipto_Escolhidos1/Eucalipto1.jpg")
if img is None:
    print("File not found. Bye!")
    exit(0)
h, w, l = img.shape

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure('Fig Color')
plt.imshow(img_rgb)
plt.axis('off')

(canal_r, canal_g, canal_b) = cv2.split(img_rgb)

img_final = np.where((canal_b > 140 or ()), 255, 0).astype(np.uint8)
plt.figure('Final')
plt.imshow(img_final, cmap='gray')
plt.axis('off')

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

plt.figure('img with text')
plt.imshow(img)
plt.show()