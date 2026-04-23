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

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

(canal_r, canal_g, canal_b) = cv2.split(img_rgb)

mask_fundo = np.where((canal_b > 140), 0, 255).astype(np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))

mask_erodida = cv2.erode(mask_fundo, kernel, iterations=2)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_erodida, connectivity=8)

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

    score = area + cy * 5

    if score > melhor_score:
        melhor_score = score
        melhor_id = i

mascara_vaso = np.uint8(labels == melhor_id) * 255

kernel_fill = np.ones((15, 15), np.uint8)
mascara_vaso = cv2.morphologyEx(mascara_vaso, cv2.MORPH_CLOSE, kernel_fill)

kernel = np.ones((5, 40), np.uint8)
mascara_vaso_ajustada = cv2.dilate(mascara_vaso, kernel, iterations=1)

img_planta = cv2.subtract(mask_fundo, mascara_vaso_ajustada)

img_planta_bool = img_planta.astype(bool)
skeleton = skeletonize(img_planta_bool).astype(np.uint8) * 255

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

melhor_id = -1
melhor_score = -1

for i in range(1, num_labels):
    area  = stats[i, cv2.CC_STAT_AREA]
    w_cc  = stats[i, cv2.CC_STAT_WIDTH]
    h_cc  = stats[i, cv2.CC_STAT_HEIGHT]

    if area < 30:
        continue

    verticalidade = h_cc / (w_cc + 1)
    score = area * verticalidade

    if score > melhor_score:
        melhor_score = score
        melhor_id = i

skeleton_caule = np.uint8(labels == melhor_id) * 255

espessura_caule = 7
kernel_caule = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (espessura_caule, espessura_caule))
mascara_caule = cv2.dilate(skeleton_caule, kernel_caule, iterations=2)

mascara_caule = cv2.bitwise_and(mascara_caule, img_planta)

kernel_close = np.ones((5, 5), np.uint8)
mascara_caule = cv2.morphologyEx(mascara_caule, cv2.MORPH_CLOSE, kernel_close)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_planta, connectivity=8)
img_limpa = np.zeros_like(img_planta)

for i in range(1, num_labels):  
    area = stats[i, cv2.CC_STAT_AREA]

    if area > 685:  
        img_limpa[labels == i] = 255

mask_folhas = cv2.subtract(img_planta, mascara_caule)

plt.figure("Folhas sem caule")
plt.imshow(mask_folhas, cmap='gray')
plt.axis('off')

params = cv2.SimpleBlobDetector_Params()

params.filterByColor = True
params.blobColor = 255

params.filterByArea = True
params.minArea = 1000
params.maxArea = 30000

params.filterByCircularity = False

params.filterByConvexity = False

params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(img_limpa)

print("Numero de folhas detectadas:", len(keypoints))

img_blob = cv2.drawKeypoints(img_limpa, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.show()

area = cv2.countNonZero(mask_folhas)

dist = cv2.distanceTransform(mascara_caule, cv2.DIST_L2, 5)

y_base = -1
for y in range(h-1, -1, -1):
    if np.any(mascara_caule[y, :] == 255):
        y_base = y
        break

y_alvo = max(0, y_base - 10)

faixa = 2

ys_caule, xs_caule = np.where(mascara_caule == 255)

if len(ys_caule) > 0:
    y_base = np.max(ys_caule)
else:
    y_base = 0

y_alvo = max(0, y_base - 10)

faixa = 2

ys_caule, xs_caule = np.where(mascara_caule == 255)

if len(ys_caule) > 0:
    y_base = np.max(ys_caule)
else:
    y_base = 0

y_alvo = max(0, y_base - 10)

mask = np.abs(ys_caule - y_alvo) <= faixa

if np.sum(mask) > 0:
    raios = dist[ys_caule[mask], xs_caule[mask]]
    largura = 2 * np.mean(raios)
else:
    largura = 0

print("Largura 10px acima da base (linha {}): {}".format(y_alvo, largura))
print("Area total das folhas:", area, "pixels")

plt.figure('mascara caule')
plt.imshow(mascara_caule, cmap='gray')
plt.axis('off')

plt.show()