import cv2
import matplotlib.pyplot as plt
import numpy as np
from library.selectBlob import *

img = cv2.imread("Projeto intermediario/Dataset_Projeto1/_Eucalipto_Escolhidos1/Eucalipto1.jpg")
if img is None:
    print("File not found. Bye!")
    exit(0)
h, w, l = img.shape

# Converte imagem para RGB e Mostra
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure('Fig Color')
plt.imshow(img_rgb)
plt.axis('off')

# Separa em 3 canais
(canal_r, canal_g, canal_b) = cv2.split(img_rgb)

# Gera mascara inicial (remove o fundo)
mask_fundo = np.where((canal_b > 140), 0, 255).astype(np.uint8)
plt.figure('Final')
plt.imshow(mask_fundo, cmap='gray')
plt.axis('off')

# --- NOVO: máscara por posição (remove o vaso abaixo de 0.69 da altura) ---
limite_base = int(0.624*h)
mask_posicao = np.zeros((h, w), dtype=np.uint8)
mask_posicao[:limite_base, :] = 255

# Combina a máscara do fundo com a máscara por posição
mask_final = cv2.bitwise_and(mask_fundo, mask_posicao)

plt.figure('Mask Final (sem vaso)')
plt.imshow(mask_final, cmap='gray')
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







plt.show()

