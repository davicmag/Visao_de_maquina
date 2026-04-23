import cv2
import matplotlib.pyplot as plt
import numpy as np
from library.selectBlob import *
from skimage.morphology import skeletonize

img = cv2.imread("Visao_de_maquina/Projeto intermediario/Dataset_Projeto1/_Eucalipto_Escolhidos1/Eucalipto1.jpg")
if img is None:
    print("File not found. Bye!")
    exit(0)
h, w, l = img.shape
# Converte imagem para RGB e Mostra
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.figure('Fig Color')
#plt.imshow(img_rgb)
#plt.axis('off')

# Separa em 3 canais
(canal_r, canal_g, canal_b) = cv2.split(img_rgb)

# Gera mascara inicial (remove o fundo)
mask_fundo = np.where((canal_b > 140), 0, 255).astype(np.uint8)
#plt.figure('Sem Fundo')
#plt.imshow(mask_fundo, cmap='gray')
#plt.axis('off')


# Separando vaso da planta

# kernel um pouco maior que a espessura do caule
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))

# erosão para quebrar a ligação planta-vaso
mask_erodida = cv2.erode(mask_fundo, kernel, iterations=2)

# detectar componentes conectados na imagem erodida
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_erodida, connectivity=8)

# escolhe o componente mais baixo e com área suficiente
melhor_id = -1
melhor_score = -1

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    cy = centroids[i][1]

    if area < 300:

        continue

    # favorece o objeto mais abaixo na imagem
    score = area + cy * 5

    if score > melhor_score:

        melhor_score = score
        melhor_id = i


# gera máscara só do vaso
mascara_vaso = np.uint8(labels == melhor_id) * 255

# fecha pequenos buracos pretos dentro do vaso
kernel_fill = np.ones((15, 15), np.uint8)
mascara_vaso = cv2.morphologyEx(mascara_vaso, cv2.MORPH_CLOSE, kernel_fill)

# Aumenta um pouco a mascara para não ficar com "bordas" na hora de subtrair (maior e Y para não prejudicar o caule)
kernel = np.ones((5, 40), np.uint8)
mascara_vaso_ajustada = cv2.dilate(mascara_vaso, kernel, iterations=1)

#plt.figure("Mascara do vaso")
#plt.imshow(mascara_vaso_ajustada, cmap='gray')
#plt.axis('off')


# Subtraindo vaso da imagem vaso+planta
img_planta = cv2.subtract(mask_fundo, mascara_vaso_ajustada)


# Isolando o caule 

# Esqueletiza a máscara da planta para obter a linha central
img_planta_bool = img_planta.astype(bool)
skeleton = skeletonize(img_planta_bool).astype(np.uint8) * 255

# Analisa componentes conectados do esqueleto
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

#  Seleciona o componente do esqueleto mais vertical e mais longo
melhor_id = -1
melhor_score = -1

for i in range(1, num_labels):
    area  = stats[i, cv2.CC_STAT_AREA]
    w_cc  = stats[i, cv2.CC_STAT_WIDTH]
    h_cc  = stats[i, cv2.CC_STAT_HEIGHT]

    if area < 30:          # ignora ruídos pequenos
        continue

    # quanto mais alto e fino, mais provável de ser o caule
    verticalidade = h_cc / (w_cc + 1)
    score = area * verticalidade

    if score > melhor_score:
        melhor_score = score
        melhor_id = i

# Máscara do esqueleto do caule
skeleton_caule = np.uint8(labels == melhor_id) * 255

# Dilata o esqueleto para reconstruir a espessura real do caule
# Ajuste o tamanho do kernel conforme a espessura do caule na imagem
espessura_caule = 7   # px — aumente se o caule for mais grosso
kernel_caule = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (espessura_caule, espessura_caule))
mascara_caule = cv2.dilate(skeleton_caule, kernel_caule, iterations=2)

# Intersecta com a máscara original para não vazar para fora da planta
mascara_caule = cv2.bitwise_and(mascara_caule, img_planta)

# Fecha pequenos buracos internos
kernel_close = np.ones((5, 5), np.uint8)
mascara_caule = cv2.morphologyEx(mascara_caule, cv2.MORPH_CLOSE, kernel_close)

# Remove pequenos ruídos da imagem
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_planta, connectivity=8)
img_limpa = np.zeros_like(img_planta)

for i in range(1, num_labels):  
    area = stats[i, cv2.CC_STAT_AREA]

    if area > 685:  
        img_limpa[labels == i] = 255

#plt.figure("Imagem Limpa")
#plt.imshow(img_limpa, cmap='gray')
#plt.axis('off')


# Subtrai caule da imagem da planta
mask_folhas = cv2.subtract(img_planta, mascara_caule)

plt.figure("Folhas sem caule")
plt.imshow(mask_folhas, cmap='gray')
plt.axis('off')

# Detecta as folhas
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
#params.minCircularity = 0.8
#params.maxCircularity = 1.2

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

# detecta os blobs nas folhas
keypoints = detector.detect(img_limpa)

# conta
print("Numero de folhas detectadas:", len(keypoints))

# desenha na imagem
img_blob = cv2.drawKeypoints(img_limpa, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#plt.figure("Folhas detectadas")
#plt.imshow(img_blob)
#plt.axis('off')
plt.show()


area = cv2.countNonZero(mask_folhas)

dist = cv2.distanceTransform(mascara_caule, cv2.DIST_L2, 5)

# pega apenas os pontos do esqueleto
raios = dist[skeleton_caule == 255]

# largura local = 2 * raio
distancias = 2 * raios

# média da largura do caule
largura_caule = np.mean(distancias) if len(distancias) > 0 else 0

print("Largura do caule (distance transform):", largura_caule)

print("Area total das folhas:", area, "pixels")
print("Largura total do caule:", largura_caule, "pixels")

plt.figure('mascara caule')
plt.imshow(mascara_caule, cmap='gray')
plt.axis('off')

plt.show()